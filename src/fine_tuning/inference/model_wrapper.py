"""Wrapper for fine-tuned financial QA model."""

import torch
from transformers import DistilBertTokenizer
from typing import Dict, Optional, Tuple
import time

from ..models.moe import FinancialMoE
from ..guardrails.input_validator import InputValidator
from ..guardrails.output_validator import OutputValidator

class FineTunedFinancialQA:
    def __init__(
        self,
        model_path: str = "checkpoints/best_checkpoint.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the fine-tuned model wrapper.
        
        Args:
            model_path: Path to the fine-tuned model checkpoint
            device: Device to run inference on
        """
        self.device = device
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Initialize model configuration
        try:
            # First try loading with weights_only=True (safer)
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except Exception as e:
            try:
                # If that fails, try loading with weights_only=False
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            except Exception as e:
                # If both loading attempts fail, raise an error
                raise RuntimeError(f"Failed to load model checkpoint from {model_path}. Error: {str(e)}")
        
        config = checkpoint.get('config', {
            'expert_types': [
                'company_info',
                'revenue_metrics',
                'expense_metrics',
                'profitability',
                'segment_analysis'
            ],
            'hidden_size': 384,
            'dropout': 0.1
        })
        
        # Initialize model with proper configuration
        print(f"Debug: Initializing model with config: {config}")
        try:
            self.model = FinancialMoE(
                expert_types=config['expert_types'],
                hidden_size=config.get('hidden_size', 768),  # Use default if not specified
                dropout=config.get('dropout', 0.1)
            )
            print("Debug: Model initialized successfully")
        except Exception as e:
            print(f"Debug: Error initializing model: {str(e)}")
            raise
        
        # Load trained weights if available
        if checkpoint.get('model_state_dict') is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded trained model weights")
        else:
            print("Using model with default weights (not trained)")
        
        self.model.to(device)
        self.model.eval()
        
        # Initialize guardrails
        self.input_validator = InputValidator(self.tokenizer)
        self.output_validator = OutputValidator()
        
        # Statistics
        self.stats = {
            'queries_processed': 0,
            'avg_response_time': 0.0,
            'high_confidence_responses': 0,
            'low_confidence_responses': 0
        }
    
    def clean_answer(self, answer: str) -> str:
        """Clean up model-generated answer.
        
        Args:
            answer: Raw answer from model
            
        Returns:
            Cleaned answer
        """
        # Remove special tokens
        answer = answer.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "")
        
        # Remove extra whitespace
        answer = " ".join(answer.split())
        
        # Remove leading/trailing whitespace
        answer = answer.strip()
        
        return answer
    
    @torch.no_grad()
    def answer_question(
        self,
        question: str,
        user_id: Optional[str] = None
    ) -> Dict:
        """Answer a financial question using the fine-tuned model.
        
        Args:
            question: Input question
            user_id: Optional user identifier for tracking
            
        Returns:
            Dictionary containing:
            - status: 'approved' or 'blocked'
            - answer: Generated answer (if approved)
            - confidence: Model's confidence score
            - expert_weights: Expert utilization weights
            - processing_time: Time taken to process
        """
        start_time = time.time()
        
        # Input validation
        is_valid, errors = self.input_validator.validate_question(question)
        if not is_valid:
            return {
                'status': 'blocked',
                'reason': ' | '.join(errors),
                'processing_time': time.time() - start_time
            }
        
        try:
            print(f"Debug: Input question: '{question}'")
            print(f"Debug: Tokenizing question...")
            
            # Tokenize input
            inputs = self.tokenizer(
                question,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Debug tokenization
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            print(f"Debug: Tokenized sequence: {tokens}")
            print(f"Debug: Token IDs: {inputs['input_ids'][0].tolist()}")
            
            print(f"Debug: Processing input with shape {inputs['input_ids'].shape}")
            
            # Get model outputs
            try:
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                print(f"Debug: Model output obtained successfully")
                
                # Extract answer span and expert weights
                start_logits, end_logits, expert_weights = outputs
                print(f"Debug: Logits shapes - Start: {start_logits.shape}, End: {end_logits.shape}")
                print(f"Debug: Expert weights shape: {expert_weights.shape}")
            except Exception as e:
                print(f"Debug: Error during model inference: {str(e)}")
                raise
            
            # Get answer text with more robust span extraction
            start_probs = torch.softmax(start_logits, dim=-1)
            end_probs = torch.softmax(end_logits, dim=-1)
            
            # Mask out special tokens ([CLS], [SEP]) from probabilities
            special_tokens = [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]
            for special_token in special_tokens:
                special_positions = (inputs['input_ids'][0] == special_token).nonzero().squeeze(-1)
                start_probs[0, special_positions] = 0.0
                end_probs[0, special_positions] = 0.0
            
            # Renormalize probabilities
            start_probs = start_probs / start_probs.sum(dim=-1, keepdim=True)
            end_probs = end_probs / end_probs.sum(dim=-1, keepdim=True)
            
            # Get top-k start and end positions
            k = 5
            top_start = torch.topk(start_probs, k)
            top_end = torch.topk(end_probs, k)
            
            print(f"Debug: Top {k} start positions: {top_start.indices[0].tolist()}")
            print(f"Debug: Top {k} start probabilities: {top_start.values[0].tolist()}")
            print(f"Debug: Top {k} end positions: {top_end.indices[0].tolist()}")
            print(f"Debug: Top {k} end probabilities: {top_end.values[0].tolist()}")
            
            # Try different combinations of start/end to find valid span
            best_score = -float('inf')
            best_answer = None
            
            for i in range(k):
                start_idx = top_start.indices[0][i]
                for j in range(k):
                    end_idx = top_end.indices[0][j]
                    if start_idx <= end_idx and end_idx < inputs['input_ids'].shape[1]:
                        # Calculate score with length penalty to favor longer answers
                        span_length = end_idx - start_idx + 1
                        base_score = top_start.values[0][i] * top_end.values[0][j]
                        length_bonus = min(span_length / 4, 1.0)  # Bonus for answers up to 4 tokens
                        score = base_score * length_bonus
                        
                        answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
                        answer = self.tokenizer.decode(answer_tokens)
                        print(f"Debug: Trying span [{start_idx}, {end_idx}] with score {score:.4f}: '{answer}'")
                        
                        # Only consider non-empty answers that don't consist solely of special tokens
                        answer_clean = answer.replace('[CLS]', '').replace('[SEP]', '').strip()
                        if len(answer_clean) > 0:
                            if score > best_score:
                                best_score = score
                                best_answer = answer_clean
                                print(f"Debug: New best answer found: '{best_answer}' with score {best_score:.4f}")
            
            if best_answer is None:
                # Fallback: use original method
                start_idx = torch.argmax(start_logits)
                end_idx = torch.argmax(end_logits)
                answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
                raw_answer = self.tokenizer.decode(answer_tokens)
            else:
                raw_answer = best_answer
            
            # Clean up the answer
            answer = self.clean_answer(raw_answer)
            
            # Calculate confidence
            confidence = torch.softmax(torch.cat([start_logits, end_logits]), dim=-1).max().item()
            
            # Output validation
            is_valid, errors = self.output_validator.validate_answer(
                answer=answer,
                question=question,
                confidence_score=confidence
            )
            
            if not is_valid:
                return {
                    'status': 'blocked',
                    'reason': ' | '.join(errors),
                    'processing_time': time.time() - start_time
                }
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats['queries_processed'] += 1
            self.stats['avg_response_time'] = (
                (self.stats['avg_response_time'] * (self.stats['queries_processed'] - 1) + processing_time)
                / self.stats['queries_processed']
            )
            
            if confidence >= 0.8:
                self.stats['high_confidence_responses'] += 1
            elif confidence < 0.6:
                self.stats['low_confidence_responses'] += 1
            
            return {
                'status': 'approved',
                'answer': answer,
                'confidence': confidence,
                'expert_weights': expert_weights.cpu().numpy().tolist(),
                'processing_time': processing_time,
                'system_mode': 'Fine-Tuned MoE'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'reason': f"Error processing question: {str(e)}",
                'processing_time': time.time() - start_time
            }
    
    def get_system_stats(self) -> Dict:
        """Get system statistics and metrics."""
        return {
            'system_status': {
                'model_loaded': True,
                'device': self.device,
                'guardrails_active': True
            },
            'security_metrics': {
                'total_requests_processed': self.stats['queries_processed'],
                'average_processing_time': self.stats['avg_response_time']
            },
            'response_quality': {
                'high_confidence_responses': self.stats['high_confidence_responses'],
                'low_confidence_responses': self.stats['low_confidence_responses'],
                'average_confidence': (
                    self.stats['high_confidence_responses'] * 0.9 +
                    self.stats['low_confidence_responses'] * 0.5
                ) / max(1, self.stats['queries_processed'])
            }
        }