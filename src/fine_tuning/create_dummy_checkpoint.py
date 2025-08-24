"""Create a dummy checkpoint for testing."""

import torch
from models.moe import FinancialMoE

def create_dummy_checkpoint():
    """Create a dummy checkpoint with proper PyTorch format."""
    config = {
        'expert_types': [
            'company_info',
            'revenue_metrics',
            'expense_metrics',
            'profitability',
            'segment_analysis'
        ],
        'hidden_size': 768,
        'dropout': 0.1
    }
    
    # Initialize model
    model = FinancialMoE(
        expert_types=config['expert_types'],
        hidden_size=config['hidden_size'],
        dropout=config['dropout']
    )
    
    # Create checkpoint
    checkpoint = {
        'config': config,
        'model_state_dict': model.state_dict(),
        'epoch': 0,
        'optimizer_state_dict': None,
        'scheduler_state_dict': None,
        'val_metrics': {'f1': 0.0}
    }
    
    # Save checkpoint
    torch.save(checkpoint, 'checkpoints/best_checkpoint.pt')
    print("✅ Created dummy checkpoint for testing")

if __name__ == "__main__":
    create_dummy_checkpoint()

