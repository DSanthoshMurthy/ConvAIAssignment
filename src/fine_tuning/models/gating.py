"""Gating network for routing questions to appropriate experts."""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig
from typing import Dict, List, Tuple

class GatingNetwork(nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_size: int = 384,  # Match model's hidden size
        dropout: float = 0.1
    ):
        """Initialize the gating network.
        
        Args:
            num_experts: Number of expert models
            hidden_size: Size of hidden layers
            dropout: Dropout probability
        """
        super().__init__()
        
        # Small DistilBERT for question understanding
        config = DistilBertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=2,
            num_attention_heads=6,
            intermediate_size=hidden_size * 4
        )
        self.question_encoder = DistilBertModel(config)
        
        # Freeze most of the encoder parameters
        for param in self.question_encoder.parameters():
            param.requires_grad = False
        
        # Only train the last layer
        for param in self.question_encoder.transformer.layer[-1].parameters():
            param.requires_grad = True
        
        # Gating layers
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # Keep same dimension initially
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_experts)  # Project to number of experts
        )
        
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the gating network.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Tuple containing:
            - routing_weights: Softmax weights for each expert
            - routing_logits: Pre-softmax logits
        """
        # Get question representation
        outputs = self.question_encoder(
            input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs[0][:, 0]
        
        # Calculate routing logits
        routing_logits = self.gate_net(pooled_output)
        
        # Apply temperature scaling
        routing_logits = routing_logits / self.temperature
        
        # Calculate routing weights with load balancing
        routing_weights = torch.softmax(routing_logits, dim=-1)
        
        return routing_weights, routing_logits
    
    def get_load_balancing_loss(
        self,
        routing_weights: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """Calculate load balancing loss to ensure even expert utilization.
        
        Args:
            routing_weights: Routing weights from forward pass
            eps: Small value for numerical stability
            
        Returns:
            Load balancing loss
        """
        # Calculate mean utilization per expert
        mean_utilization = routing_weights.mean(dim=0)
        
        # Penalize uneven utilization
        cv_squared = torch.var(mean_utilization) / (torch.mean(mean_utilization) + eps)
        
        return cv_squared

