"""Training metrics and evaluation utilities."""

import torch
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def compute_exact_match(prediction: str, truth: str) -> float:
    """Compute exact match score."""
    return float(prediction.strip() == truth.strip())

def compute_f1(prediction: str, truth: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = prediction.strip().split()
    truth_tokens = truth.strip().split()
    
    # If either is empty, return 0
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0
    
    common = set(pred_tokens) & set(truth_tokens)
    
    # If there are no common tokens, return 0
    if len(common) == 0:
        return 0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def compute_metrics(
    predictions: List[str],
    truths: List[str],
    expert_predictions: List[str]
) -> Dict[str, float]:
    """Compute all evaluation metrics.
    
    Args:
        predictions: List of predicted answers
        truths: List of ground truth answers
        expert_predictions: List of expert categories predicted
        
    Returns:
        Dictionary of metrics
    """
    exact_matches = []
    f1_scores = []
    
    for pred, truth in zip(predictions, truths):
        exact_matches.append(compute_exact_match(pred, truth))
        f1_scores.append(compute_f1(pred, truth))
    
    # Calculate expert prediction accuracy
    expert_accuracy = np.mean([p == t for p, t in zip(expert_predictions, truths)])
    
    metrics = {
        'exact_match': np.mean(exact_matches),
        'f1': np.mean(f1_scores),
        'expert_accuracy': expert_accuracy
    }
    
    return metrics

class MetricsTracker:
    """Track and log training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.total_loss = 0
        self.qa_loss = 0
        self.load_balancing_loss = 0
        self.num_batches = 0
        self.expert_utilization = torch.zeros(5)  # For 5 experts
    
    def update(
        self,
        loss: float,
        qa_loss: float,
        load_balancing_loss: float,
        expert_weights: torch.Tensor
    ):
        """Update metrics with batch results."""
        self.total_loss += loss
        self.qa_loss += qa_loss
        self.load_balancing_loss += load_balancing_loss
        self.expert_utilization += expert_weights.sum(dim=0).cpu()
        self.num_batches += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """Get average metrics."""
        metrics = {
            'loss': self.total_loss / self.num_batches,
            'qa_loss': self.qa_loss / self.num_batches,
            'load_balancing_loss': self.load_balancing_loss / self.num_batches,
        }
        
        # Add expert utilization percentages
        utilization = (self.expert_utilization / self.expert_utilization.sum()).numpy()
        for i, util in enumerate(utilization):
            metrics[f'expert_{i}_utilization'] = util
        
        return metrics

