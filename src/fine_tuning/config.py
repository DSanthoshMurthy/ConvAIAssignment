"""Configuration for fine-tuning DistilBERT with Mixture of Experts."""

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ExpertConfig:
    hidden_size: int = 768  # DistilBERT's hidden size
    intermediate_size: int = 384
    num_attention_heads: int = 12
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100

@dataclass
class MoEConfig:
    num_experts: int = 5
    expert_hidden_size: int = 768
    gating_hidden_size: int = 384
    gating_dropout: float = 0.1
    expert_dropout: float = 0.1
    capacity_factor: float = 1.2  # Allow each expert to process 20% more tokens than even distribution
    
    # Expert categories and their descriptions
    expert_categories: Dict[str, str] = None
    
    def __post_init__(self):
        self.expert_categories = {
            'income_statement': 'Expert in processing questions about revenue, expenses, profit/loss, and EPS',
            'balance_sheet': 'Expert in handling questions about assets, liabilities, and equity',
            'cash_flow': 'Expert in analyzing cash flow related queries',
            'ratios': 'Expert in calculating and interpreting financial ratios',
            'temporal': 'Expert in handling time-based comparisons and trends'
        }

@dataclass
class GuardrailsConfig:
    min_confidence_threshold: float = 0.7
    max_question_length: int = 512
    min_financial_terms: int = 1
    allowed_currencies: List[str] = None
    
    def __post_init__(self):
        self.allowed_currencies = ['₹', '$', '€', '£']  # Add more as needed

# Create default configurations
expert_config = ExpertConfig()
training_config = TrainingConfig()
moe_config = MoEConfig()
guardrails_config = GuardrailsConfig()

