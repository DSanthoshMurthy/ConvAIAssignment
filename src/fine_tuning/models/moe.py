"""Mixture of Experts model for financial QA."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .expert import FinancialExpert
from .gating import GatingNetwork

class FinancialMoE(nn.Module):
    def __init__(
        self,
        expert_types: List[str],
        hidden_size: int = 768,
        dropout: float = 0.1,
        load_balancing_weight: float = 0.01,
        gradient_checkpointing: bool = False
    ):
        """Initialize the Mixture of Experts model.
        
        Args:
            expert_types: List of expert types to create
            hidden_size: Hidden layer size
            dropout: Dropout probability
            load_balancing_weight: Weight for load balancing loss
        """
        super().__init__()
        
        self.num_experts = len(expert_types)
        self.load_balancing_weight = load_balancing_weight
        self.gradient_checkpointing = gradient_checkpointing
        
        # Create expert models
        self.experts = nn.ModuleList([
            FinancialExpert(expert_type=expert_type)
            for expert_type in expert_types
        ])
        
        # Enable gradient checkpointing for experts if requested
        if gradient_checkpointing:
            for expert in self.experts:
                expert.distilbert.gradient_checkpointing_enable()
        
        # Create gating network
        self.gating = GatingNetwork(
            num_experts=self.num_experts,
            hidden_size=hidden_size,
            dropout=dropout
        )
        
        # Expert fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * self.num_experts, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass for the MoE model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            start_positions: Start positions for training (optional)
            end_positions: End positions for training (optional)
            
        Returns:
            Tuple containing:
            - loss: Total loss (if training)
            - start_logits: Logits for answer start positions
            - end_logits: Logits for answer end positions
            - expert_weights: Routing weights for each expert
        """
        # Get routing weights from gating network
        routing_weights, routing_logits = self.gating(input_ids, attention_mask)
        
        # Get outputs from each expert
        expert_start_logits = []
        expert_end_logits = []
        expert_losses = []
        
        for i, expert in enumerate(self.experts):
            expert_out = expert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )
            
            if start_positions is not None:
                # Training mode - expert returns (loss, start_logits, end_logits)
                expert_losses.append(expert_out[0])
                expert_start_logits.append(expert_out[1])
                expert_end_logits.append(expert_out[2])
            else:
                # Inference mode - expert returns (start_logits, end_logits)
                expert_start_logits.append(expert_out[0])
                expert_end_logits.append(expert_out[1])
        
        # Stack expert outputs
        expert_start_logits = torch.stack(expert_start_logits)  # [num_experts, batch_size, seq_len]
        expert_end_logits = torch.stack(expert_end_logits)      # [num_experts, batch_size, seq_len]
        
        # Combine expert outputs using routing weights
        routing_weights = routing_weights.unsqueeze(-1)  # [batch_size, num_experts, 1]
        
        combined_start_logits = torch.sum(
            routing_weights * expert_start_logits.permute(1, 0, 2),  # [batch_size, num_experts, seq_len]
            dim=1
        )
        combined_end_logits = torch.sum(
            routing_weights * expert_end_logits.permute(1, 0, 2),    # [batch_size, num_experts, seq_len]
            dim=1
        )
        

        
        outputs = (combined_start_logits, combined_end_logits, routing_weights)
        
        if start_positions is not None and end_positions is not None:
            # Calculate expert losses
            if expert_losses:
                expert_loss = torch.mean(torch.stack(expert_losses))
            else:
                # Calculate QA loss from combined outputs
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(combined_start_logits, start_positions)
                end_loss = loss_fct(combined_end_logits, end_positions)
                expert_loss = (start_loss + end_loss) / 2
            
            # Calculate load balancing loss
            load_balancing_loss = self.gating.get_load_balancing_loss(routing_weights)
            
            # Combine losses
            total_loss = expert_loss + self.load_balancing_weight * load_balancing_loss
            outputs = (total_loss,) + outputs
        
        return outputs
    
    def get_expert_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get routing weights for experts (useful for analysis).
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Routing weights for each expert
        """
        with torch.no_grad():
            routing_weights, _ = self.gating(input_ids, attention_mask)
        return routing_weights
