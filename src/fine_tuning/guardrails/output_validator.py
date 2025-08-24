"""Output validation guardrails for financial QA."""

import re
from typing import Dict, List, Optional, Tuple
import numpy as np

class OutputValidator:
    def __init__(
        self,
        confidence_threshold: float = 0.4,  # Reduced threshold since we have better validation
        max_number_threshold: float = 1e12  # 1 trillion
    ):
        """Initialize output validator.
        
        Args:
            confidence_threshold: Minimum confidence score for valid answers
            max_number_threshold: Maximum reasonable number in answers
        """
        self.confidence_threshold = confidence_threshold
        self.max_number_threshold = max_number_threshold
        
        # Currency symbols and their expected formats
        self.currency_patterns = {
            '₹': r'₹\s*[\d,]+\.?\d*\s*(?:billion|million|thousand|crore|lakh)?',
            '$': r'\$\s*[\d,]+\.?\d*\s*(?:billion|million|thousand)?',
            '€': r'€\s*[\d,]+\.?\d*\s*(?:billion|million|thousand)?'
        }
        
        # Number multiplier mapping
        self.multipliers = {
            'billion': 1e9,
            'million': 1e6,
            'thousand': 1e3,
            'crore': 1e7,
            'lakh': 1e5
        }
    
    def validate_confidence(
        self,
        confidence_score: float
    ) -> Tuple[bool, str]:
        """Validate if the confidence score is above threshold.
        
        Args:
            confidence_score: Model's confidence score
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if confidence_score < self.confidence_threshold:
            return False, f"Confidence score {confidence_score:.2f} below threshold {self.confidence_threshold}"
        
        return True, ""
    
    def extract_number(self, text: str) -> Optional[float]:
        """Extract numerical value from text with unit conversion.
        
        Args:
            text: Text containing number
            
        Returns:
            Extracted number or None if not found
        """
        # Remove commas and spaces
        text = text.replace(',', '').lower()
        
        # Find number and multiplier
        number_match = re.search(r'[\d.]+', text)
        if not number_match:
            return None
        
        number = float(number_match.group())
        
        # Apply multiplier if present
        for unit, multiplier in self.multipliers.items():
            if unit in text:
                number *= multiplier
                break
        
        return number
    
    def validate_numerical_consistency(
        self,
        answer: str,
        question: str
    ) -> Tuple[bool, str]:
        """Validate numerical values in the answer.
        
        Args:
            answer: Generated answer
            question: Original question
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Extract numbers from answer
        number = self.extract_number(answer)
        if number is None:
            # If question asks for number but none found
            if any(term in question.lower() for term in ['how much', 'how many', 'amount', 'value']):
                return False, "No numerical value found in answer to quantitative question"
            return True, ""
        
        # Check if number is reasonable
        if number > self.max_number_threshold:
            return False, f"Numerical value {number} exceeds maximum threshold"
        
        if number < 0 and 'loss' not in answer.lower():
            return False, "Negative value without loss context"
        
        return True, ""
    
    def validate_currency_format(
        self,
        answer: str
    ) -> Tuple[bool, str]:
        """Validate currency formats in the answer.
        
        Args:
            answer: Generated answer
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if answer contains currency symbols
        has_currency = any(symbol in answer for symbol in self.currency_patterns.keys())
        if not has_currency:
            return True, ""  # No currency to validate
        
        # Validate currency format
        for symbol, pattern in self.currency_patterns.items():
            if symbol in answer:
                if not re.search(pattern, answer):
                    return False, f"Invalid currency format for {symbol}"
        
        return True, ""
    
    def validate_answer_format(
        self,
        answer: str,
        question: str
    ) -> Tuple[bool, str]:
        """Validate general answer format.
        
        Args:
            answer: Generated answer
            question: Original question
            
        Returns:
            Tuple of (is_valid, error_message)
        """
                # No longer checking for question words as they might be part of valid answers
        
        return True, ""
    
    def validate_answer(
        self,
        answer: str,
        question: str,
        confidence_score: float
    ) -> Tuple[bool, List[str]]:
        """Run all validation checks on generated answer.
        
        Args:
            answer: Generated answer
            question: Original question
            confidence_score: Model's confidence score
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []
        
        # Check confidence
        conf_valid, conf_error = self.validate_confidence(confidence_score)
        if not conf_valid:
            errors.append(conf_error)
        
        # Check numerical consistency
        num_valid, num_error = self.validate_numerical_consistency(answer, question)
        if not num_valid:
            errors.append(num_error)
        
        # Check currency format
        curr_valid, curr_error = self.validate_currency_format(answer)
        if not curr_valid:
            errors.append(curr_error)
        
        # Check answer format
        format_valid, format_error = self.validate_answer_format(answer, question)
        if not format_valid:
            errors.append(format_error)
        
        return len(errors) == 0, errors

