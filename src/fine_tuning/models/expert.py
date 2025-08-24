"""Expert model implementation for financial QA."""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig
from typing import Dict, Optional, Tuple

class FinancialExpert(nn.Module):
    def __init__(
        self,
        expert_type: str,
        config: Optional[DistilBertConfig] = None,
        pretrained: bool = True
    ):
        """Initialize a financial expert model.
        
        Args:
            expert_type: Type of financial expertise (e.g., 'revenue_metrics')
            config: DistilBERT configuration (optional)
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        # Initialize DistilBERT with full size
        if config is None:
            config = DistilBertConfig(
                hidden_size=768,  # Full size for better performance
                num_hidden_layers=6,  # Standard layers
                num_attention_heads=12,
                intermediate_size=3072  # 4x hidden size
            )
        
        # Load pre-trained model if specified
        if pretrained:
            self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased', config=config)
        else:
            self.distilbert = DistilBertModel(config)
            
        # Expert-specific layers
        self.expert_type = expert_type
        self.dropout = nn.Dropout(0.2)  # Increased dropout
        
        # Expert-specific head architecture optimized for financial data
        hidden_sizes = {
            'financial_metrics': [768, 512, 256],  # Revenue, income analysis
            'profitability': [768, 512, 256],     # Profit/loss analysis
            'operational_metrics': [768, 512, 256],  # Operations, expenses
            'segment_performance': [768, 512, 256],  # Segment analysis
            'temporal_analysis': [768, 512, 256]     # Time-based analysis
        }
        
        # Add expert-specific processing
        self.expert_preprocessor = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Add output projection layer
        self.output_projection = nn.Linear(256, 768)  # Project back to original dimension
        
        # Build expert head
        layers = []
        sizes = hidden_sizes[expert_type]
        for i in range(len(sizes) - 1):
            layers.extend([
                nn.Linear(sizes[i], sizes[i + 1]),
                nn.LayerNorm(sizes[i + 1]),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
        
        self.expert_head = nn.Sequential(*layers)
        
        # Final QA output layer (using original dimension)
        self.qa_outputs = nn.Linear(768, 2)  # start_logits, end_logits for QA
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass for the expert model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            start_positions: Start positions for training (optional)
            end_positions: End positions for training (optional)
            
        Returns:
            Tuple containing:
            - start_logits: Logits for answer start positions
            - end_logits: Logits for answer end positions
            - loss: Training loss (if start/end positions provided)
        """
        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        
        # Apply expert-specific preprocessing
        sequence_output = self.expert_preprocessor(sequence_output)
        
        # Apply expert-specific processing
        sequence_output = self.dropout(sequence_output)
        sequence_output = self.expert_head(sequence_output)
        
        # Project back to original dimension and add residual connection
        sequence_output = self.output_projection(sequence_output)
        sequence_output = sequence_output + outputs[0]
        
        # Get logits for answer start/end positions
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # [batch_size, seq_len]
        end_logits = end_logits.squeeze(-1)      # [batch_size, seq_len]
        
        outputs = (start_logits, end_logits)
        
        if start_positions is not None and end_positions is not None:
            # Calculate loss
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs
            
        return outputs  # (loss), start_logits, end_logits

