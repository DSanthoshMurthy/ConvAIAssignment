"""Train the financial QA model on our dataset."""

import torch
from torch.utils.data import DataLoader, Dataset
import json
from pathlib import Path
import wandb
from tqdm import tqdm
from transformers import DistilBertTokenizer, get_linear_schedule_with_warmup
import numpy as np

from models.moe import FinancialMoE
from data.dataset_processor import FinancialDatasetProcessor

class FinancialQADataset(Dataset):
    def __init__(self, qa_pairs, tokenizer, max_length):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-tokenize data to save memory during training
        self.tokenized_data = []
        for qa in qa_pairs:
            encoding = self.tokenizer(
                qa['question'],
                qa['answer'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            self.tokenized_data.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'answer_text': qa['answer']  # Keep original answer for span finding
            })
        
        # Categorize questions
        self.processor = FinancialDatasetProcessor("dummy_path")  # Just for categorization
        for qa in self.qa_pairs:
            qa['category'] = self.processor.categorize_question(qa['question'])
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        item = self.tokenized_data[idx]
        qa = self.qa_pairs[idx]
        
        # Get pre-tokenized tensors
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        
        # Find answer span (using cached answer text)
        answer_tokens = self.tokenizer.encode(item['answer_text'], add_special_tokens=False)
        input_ids_list = input_ids.tolist()
        
        # Find the start position of answer in the combined sequence
        start_position = 0
        end_position = 0
        
        # Use binary search for faster span finding
        input_ids_str = str(input_ids_list)
        answer_str = str(answer_tokens)
        start_idx = input_ids_str.find(answer_str)
        
        if start_idx != -1:
            # Convert string index back to token index
            start_position = input_ids_str[:start_idx].count(',')
            end_position = start_position + len(answer_tokens) - 1
        else:
            # If answer not found, use [SEP] token position as default
            sep_pos = input_ids_list.index(self.tokenizer.sep_token_id)
            start_position = sep_pos
            end_position = sep_pos
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'start_positions': torch.tensor(start_position),
            'end_positions': torch.tensor(end_position),
            'category': qa['category']
        }

def train_model(
    train_loader,
    val_loader,
    model,
    optimizer,
    scheduler,
    num_epochs,
    device,
    checkpoint_dir,
    config,
    gradient_accumulation_steps=4  # Accumulate gradients over multiple steps
):
    """Train the model."""
    best_val_loss = float('inf')
    patience = config.get('patience', 3)
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0
        train_steps = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                start_positions=batch['start_positions'],
                end_positions=batch['end_positions']
            )
            
            loss = outputs[0]  # First element is loss
            
            # Scale loss by gradient accumulation steps
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Update weights if we've accumulated enough gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            train_loss += loss.item()
            train_steps += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Log to wandb
            if config.get('use_wandb', False):
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': scheduler.get_last_lr()[0]
                })
        
        avg_train_loss = train_loss / train_steps
        
        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    start_positions=batch['start_positions'],
                    end_positions=batch['end_positions']
                )
                
                loss = outputs[0]
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        
        print(f"Average train loss: {avg_train_loss:.4f}")
        print(f"Average validation loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config,
            'val_metrics': {'loss': avg_val_loss}
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / 'latest_checkpoint.pt')
        
        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(checkpoint, checkpoint_dir / 'best_checkpoint.pt')
            print("✓ Saved new best checkpoint!")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs!")
            break
        
        # Log to wandb
        if config.get('use_wandb', False):
            wandb.log({
                'epoch': epoch,
                'avg_train_loss': avg_train_loss,
                'avg_val_loss': avg_val_loss
            })
    
    return best_val_loss

def main():
    # Load configuration with memory optimizations
    config = {
        # Improved training configuration
        'batch_size': 8,
        'learning_rate': 1e-4,  # Increased learning rate
        'num_epochs': 20,  # More epochs
        'warmup_steps': 100,
        'max_grad_norm': 1.0,
        'weight_decay': 0.01,
        'hidden_size': 768,  # Full size
        'dropout': 0.2,  # Increased dropout
        'expert_types': [
            'financial_metrics',      # Revenue, income, etc.
            'profitability',         # Profit/loss analysis
            'operational_metrics',   # Operations, expenses
            'segment_performance',   # Segment-wise analysis
            'temporal_analysis'      # Time-based comparisons
        ],
        'use_wandb': False,
        'patience': 5,  # Increased patience
        
        # Better data handling
        'gradient_accumulation_steps': 4,
        'max_length': 512,  # Full sequence length
        'eval_batch_size': 4,
        'num_workers': 0,
        'pin_memory': False,
        'use_gradient_checkpointing': True
    }
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Load dataset
    print("Loading dataset...")
    with open('data/processed/xbrl_qa_pairs.json', 'r') as f:
        qa_pairs = json.load(f)
    
    # Create datasets
    print(f"Creating dataset with max_length={config['max_length']}...")
    full_dataset = FinancialQADataset(qa_pairs, tokenizer, max_length=config['max_length'])
    
    # Split dataset
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders with memory optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['eval_batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['eval_batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    # Initialize model
    print("Initializing model...")
    model = FinancialMoE(
        expert_types=config['expert_types'],
        hidden_size=config['hidden_size'],
        dropout=config['dropout'],
        gradient_checkpointing=config['use_gradient_checkpointing']
    )
    
    model.to(device)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {total_params/1e6:.1f}M parameters")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Initialize scheduler
    num_training_steps = len(train_loader) * config['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=num_training_steps
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Initialize wandb
    if config['use_wandb']:
        wandb.init(
            project="financial-qa-moe",
            config=config
        )
    
    # Train model
    print("Starting training...")
    best_val_loss = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config['num_epochs'],
        device=device,
        checkpoint_dir=checkpoint_dir,
        config=config
    )
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
    
    # Test final model
    print("\nEvaluating on test set...")
    model.eval()
    test_loss = 0
    test_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                start_positions=batch['start_positions'],
                end_positions=batch['end_positions']
            )
            
            loss = outputs[0]
            test_loss += loss.item()
            test_steps += 1
    
    avg_test_loss = test_loss / test_steps
    print(f"Final test loss: {avg_test_loss:.4f}")

if __name__ == "__main__":
    main()
