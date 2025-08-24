"""Main training script for financial QA model."""

import torch
from transformers import DistilBertTokenizer
import json
from pathlib import Path
import argparse

from data.dataset_processor import FinancialDatasetProcessor
from training.data_loaders import create_data_loaders
from models.moe import FinancialMoE
from training.trainer import FinancialQATrainer

def load_config(config_path: str) -> dict:
    """Load training configuration."""
    with open(config_path) as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Train Financial QA Model')
    parser.add_argument('--config', type=str, default='configs/training_config.json',
                      help='Path to training configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Load and process data
    print("Loading data...")
    data_processor = FinancialDatasetProcessor('data/processed/xbrl_qa_pairs.json')
    qa_pairs = data_processor.load_data()
    datasets = data_processor.prepare_for_training(qa_pairs)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        datasets['train'],
        datasets['validation'],
        datasets['test'],
        tokenizer,
        batch_size=config['batch_size']
    )
    
    # Initialize model
    print("Initializing model...")
    model = FinancialMoE(
        expert_types=list(data_processor.expert_categories.keys()),
        hidden_size=config['hidden_size'],
        dropout=config['dropout'],
        load_balancing_weight=config['load_balancing_weight']
    )
    
    # Initialize trainer
    print("Setting up trainer...")
    trainer = FinancialQATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )
    
    # Train model
    print("Starting training...")
    test_metrics = trainer.train()
    
    # Save final metrics
    metrics_path = Path('results') / 'test_metrics.json'
    metrics_path.parent.mkdir(exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print("\nTraining completed!")
    print(f"Test metrics saved to {metrics_path}")

if __name__ == '__main__':
    main()

