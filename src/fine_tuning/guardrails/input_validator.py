"""Input validation guardrails for financial QA."""

import re
from typing import Dict, List, Optional, Tuple
import spacy
from transformers import DistilBertTokenizer

class InputValidator:
    def __init__(self, tokenizer: DistilBertTokenizer):
        """Initialize input validator.
        
        Args:
            tokenizer: DistilBERT tokenizer for length validation
        """
        self.tokenizer = tokenizer
        
        # Load spaCy for financial term detection
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            # If model not found, download it
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Financial terms and patterns
        self.financial_terms = {
            'general': ['revenue', 'profit', 'loss', 'income', 'expense', 'cost',
                       'asset', 'liability', 'equity', 'cash', 'tax', 'debt',
                       'earnings', 'eps', 'margin', 'ratio', 'balance', 'capital'],
            'temporal': ['year', 'quarter', 'period', 'annual', 'quarterly',
                        'fiscal', 'fy', 'q1', 'q2', 'q3', 'q4'],
            'metrics': ['billion', 'million', 'thousand', 'crore', 'lakh',
                       'percentage', '%', '₹', '$', '€']
        }
        
        # Question patterns
        self.question_patterns = [
            r'^what\s+(?:is|was|were)',
            r'^how\s+(?:much|many)',
            r'^when\s+(?:is|was|were)',
            r'^which\s+(?:is|was|were)',
            r'^where\s+(?:is|was|were)',
            r'^why\s+(?:is|was|were)',
        ]
    
    def validate_question_format(self, question: str) -> Tuple[bool, str]:
        """Validate if the question follows expected format.
        
        Args:
            question: Input question
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if empty or too short
        if not question or len(question.strip()) < 3:  # Reduced minimum length
            return False, "Question is too short"
        
        return True, ""
    
    def validate_financial_terms(self, question: str) -> Tuple[bool, str]:
        """Validate if the question contains relevant financial terms.
        
        Args:
            question: Input question
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        question_lower = question.lower()
        
        # Check for financial terms
        found_terms = []
        for category, terms in self.financial_terms.items():
            for term in terms:
                if term in question_lower:
                    found_terms.append(term)
        
        if not found_terms:
            return False, "Question should contain at least one financial term"
        
        return True, ""
    
    def validate_length(self, question: str) -> Tuple[bool, str]:
        """Validate if question length is within model limits.
        
        Args:
            question: Input question
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        tokens = self.tokenizer.encode(question)
        if len(tokens) > 512:  # DistilBERT max length
            return False, "Question is too long"
        
        return True, ""
    
    def validate_question(self, question: str) -> Tuple[bool, List[str]]:
        """Run all validation checks on input question.
        
        Args:
            question: Input question
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []
        
        # Check format
        format_valid, format_error = self.validate_question_format(question)
        if not format_valid:
            errors.append(format_error)
        
        # Check financial terms
        terms_valid, terms_error = self.validate_financial_terms(question)
        if not terms_valid:
            errors.append(terms_error)
        
        # Check length
        length_valid, length_error = self.validate_length(question)
        if not length_valid:
            errors.append(length_error)
        
        return len(errors) == 0, errors

