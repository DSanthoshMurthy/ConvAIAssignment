"""Data loading utilities for financial QA training."""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from typing import Dict, List, Optional, Tuple
import numpy as np

class FinancialQADataset(Dataset):
    def __init__(
        self,
        questions: List[str],
        answers: List[str],
        expert_categories: List[str],
        tokenizer: DistilBertTokenizer,
        max_length: int = 512
    ):
        """Initialize dataset for financial QA.
        
        Args:
            questions: List of questions
            answers: List of answers
            expert_categories: List of expert categories for each QA pair
            tokenizer: DistilBERT tokenizer
            max_length: Maximum sequence length
        """
        self.questions = questions
        self.answers = answers
        self.expert_categories = expert_categories
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        question = self.questions[idx]
        answer = self.answers[idx]
        category = self.expert_categories[idx]
        
        # Tokenize question and answer
        encoding = self.tokenizer(
            question,
            answer,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get answer start and end positions
        answer_start = encoding.char_to_token(len(question) + 1)  # +1 for [SEP] token
        answer_end = answer_start + len(self.tokenizer.encode(answer, add_special_tokens=False))
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'start_positions': torch.tensor(answer_start, dtype=torch.long),
            'end_positions': torch.tensor(answer_end, dtype=torch.long),
            'expert_category': category
        }

def create_data_loaders(
    train_data: Dict,
    val_data: Dict,
    test_data: Dict,
    tokenizer: DistilBertTokenizer,
    batch_size: int = 8,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation and testing.
    
    Args:
        train_data: Training data dictionary
        val_data: Validation data dictionary
        test_data: Test data dictionary
        tokenizer: DistilBERT tokenizer
        batch_size: Batch size
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = FinancialQADataset(
        questions=train_data['questions'],
        answers=train_data['answers'],
        expert_categories=train_data['expert_categories'],
        tokenizer=tokenizer
    )
    
    val_dataset = FinancialQADataset(
        questions=val_data['questions'],
        answers=val_data['answers'],
        expert_categories=val_data['expert_categories'],
        tokenizer=tokenizer
    )
    
    test_dataset = FinancialQADataset(
        questions=test_data['questions'],
        answers=test_data['answers'],
        expert_categories=test_data['expert_categories'],
        tokenizer=tokenizer
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

