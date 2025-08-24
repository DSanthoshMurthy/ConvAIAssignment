"""Trainer class for financial QA model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import wandb
from typing import Dict, Optional, Tuple
import os
from pathlib import Path
import json
from tqdm import tqdm

from .metrics import MetricsTracker, compute_metrics
from ..models.moe import FinancialMoE

class FinancialQATrainer:
    def __init__(
        self,
        model: FinancialMoE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize trainer.
        
        Args:
            model: MoE model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Setup scheduler
        num_training_steps = len(train_loader) * config['num_epochs']
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=num_training_steps
        )
        
        # Setup metrics
        self.metrics = MetricsTracker()
        
        # Setup checkpointing
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Setup early stopping
        self.best_val_loss = float('inf')
        self.patience = config.get('patience', 3)
        self.patience_counter = 0
        
        # Initialize wandb
        if config.get('use_wandb', False):
            wandb.init(
                project="financial-qa-moe",
                config=config
            )
    
    def save_checkpoint(
        self,
        epoch: int,
        val_metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_metrics': val_metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(
            checkpoint,
            self.checkpoint_dir / 'latest_checkpoint.pt'
        )
        
        # Save best checkpoint
        if is_best:
            torch.save(
                checkpoint,
                self.checkpoint_dir / 'best_checkpoint.pt'
            )
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch']
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.metrics.reset()
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                start_positions=batch['start_positions'],
                end_positions=batch['end_positions']
            )
            
            loss = outputs[0]
            qa_loss = outputs[1]
            load_balancing_loss = outputs[2]
            expert_weights = outputs[3]
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['max_grad_norm']
            )
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            self.metrics.update(
                loss.item(),
                qa_loss.item(),
                load_balancing_loss.item(),
                expert_weights.detach()
            )
            
            # Update progress bar
            progress_bar.set_postfix(
                loss=f"{self.metrics.get_metrics()['loss']:.4f}"
            )
        
        return self.metrics.get_metrics()
    
    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        self.metrics.reset()
        
        all_predictions = []
        all_truths = []
        all_expert_predictions = []
        
        for batch in tqdm(data_loader, desc='Evaluating'):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            start_logits = outputs[0]
            end_logits = outputs[1]
            expert_weights = outputs[2]
            
            # Get predictions
            start_idx = torch.argmax(start_logits, dim=1)
            end_idx = torch.argmax(end_logits, dim=1)
            
            # Convert predictions to text
            for i in range(len(batch['input_ids'])):
                pred_tokens = batch['input_ids'][i][start_idx[i]:end_idx[i]+1]
                pred_text = self.tokenizer.decode(pred_tokens)
                
                truth_start = batch['start_positions'][i]
                truth_end = batch['end_positions'][i]
                truth_tokens = batch['input_ids'][i][truth_start:truth_end+1]
                truth_text = self.tokenizer.decode(truth_tokens)
                
                all_predictions.append(pred_text)
                all_truths.append(truth_text)
                
                # Get predicted expert
                expert_idx = torch.argmax(expert_weights[i])
                all_expert_predictions.append(self.config['expert_types'][expert_idx])
        
        # Compute metrics
        metrics = compute_metrics(
            all_predictions,
            all_truths,
            all_expert_predictions
        )
        
        return metrics
    
    def train(self) -> Dict[str, float]:
        """Train model for specified number of epochs."""
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Evaluate
            val_metrics = self.evaluate(self.val_loader)
            
            # Log metrics
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    **train_metrics,
                    **{'val_' + k: v for k, v in val_metrics.items()}
                })
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Exact Match: {val_metrics['exact_match']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            
            # Save checkpoint
            is_best = val_metrics['f1'] > self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['f1']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print("Early stopping triggered!")
                break
        
        # Load best model and evaluate on test set
        self.load_checkpoint(self.checkpoint_dir / 'best_checkpoint.pt')
        test_metrics = self.evaluate(self.test_loader)
        
        print("\nTest Results:")
        print(f"Exact Match: {test_metrics['exact_match']:.4f}")
        print(f"F1: {test_metrics['f1']:.4f}")
        print(f"Expert Accuracy: {test_metrics['expert_accuracy']:.4f}")
        
        return test_metrics

