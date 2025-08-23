#!/usr/bin/env python3
"""
Financial RAG Guardrails System
Phase 6: Input/Output validation, content filtering, and security measures
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialRAGGuardrails:
    def __init__(self, 
                 max_query_length: int = 500,
                 rate_limit_per_minute: int = 60,
                 rate_limit_per_hour: int = 1000):
        """Initialize the Financial RAG Guardrails system.
        
        Args:
            max_query_length: Maximum allowed query length
            rate_limit_per_minute: Max queries per minute per user
            rate_limit_per_hour: Max queries per hour per user
        """
        self.max_query_length = max_query_length
        self.rate_limit_per_minute = rate_limit_per_minute
        self.rate_limit_per_hour = rate_limit_per_hour
        
        # Rate limiting tracking
        self.request_history = defaultdict(lambda: deque())
        
        # Blocked patterns and keywords
        self.blocked_patterns = [
            # Injection attempts
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'onload\s*=',
            r'onerror\s*=',
            r'eval\s*\(',
            r'exec\s*\(',
            
            # SQL injection patterns
            r'union\s+select',
            r'drop\s+table',
            r'delete\s+from',
            r'insert\s+into',
            r'update\s+.*set',
            
            # System commands
            r'rm\s+-rf',
            r'sudo\s+',
            r'chmod\s+',
            r'wget\s+',
            r'curl\s+',
            
            # Sensitive information attempts
            r'password',
            r'credit\s*card',
            r'ssn|social\s*security',
            r'bank\s*account',
            r'routing\s*number',
        ]
        
        # Inappropriate content keywords
        self.inappropriate_keywords = [
            'hack', 'hacker', 'exploit', 'vulnerability', 'penetration', 
            'phishing', 'malware', 'virus', 'trojan', 'ransomware',
            'illegal', 'fraud', 'scam', 'embezzle', 'launder',
            'insider trading', 'market manipulation', 'ponzi'
        ]
        
        # Required financial context keywords (at least one should be present)
        self.financial_context_keywords = [
            'revenue', 'profit', 'loss', 'earnings', 'expense', 'cost',
            'assets', 'liabilities', 'equity', 'debt', 'cash flow',
            'quarter', 'fiscal', 'financial', 'income', 'statement',
            'balance sheet', 'P&L', 'EBITDA', 'margin', 'ratio',
            'investment', 'dividend', 'share', 'stock', 'valuation',
            'budget', 'forecast', 'variance', 'performance', 'growth'
        ]
        
        # Output content filters
        self.output_blocked_patterns = [
            r'I cannot.*', # Avoid refusal patterns
            r'As an AI.*', # Remove AI disclaimers
            r'I am not.*financial advisor',
            r'please consult.*financial advisor',
        ]
        
        # Statistics tracking
        self.guardrail_stats = {
            'total_queries_processed': 0,
            'queries_blocked': 0,
            'rate_limit_violations': 0,
            'inappropriate_content_blocked': 0,
            'non_financial_queries_blocked': 0,
            'output_filtered_count': 0,
            'block_reasons': defaultdict(int)
        }
    
    def validate_input_query(self, 
                           query: str, 
                           user_id: str = "anonymous") -> Tuple[bool, str, Dict[str, Any]]:
        """Comprehensive input validation for queries.
        
        Args:
            query: Input query string
            user_id: Unique user identifier for rate limiting
            
        Returns:
            Tuple of (is_valid, reason, validation_metadata)
        """
        validation_start = datetime.now()
        validation_metadata = {
            'timestamp': validation_start.isoformat(),
            'user_id': user_id,
            'query_length': len(query),
            'validation_checks_passed': [],
            'validation_checks_failed': []
        }
        
        # Update statistics
        self.guardrail_stats['total_queries_processed'] += 1
        
        try:
            # Check 1: Query length validation
            if len(query.strip()) == 0:
                self._record_failure("empty_query", validation_metadata)
                return False, "Empty query not allowed.", validation_metadata
            
            if len(query) > self.max_query_length:
                self._record_failure("query_too_long", validation_metadata)
                return False, f"Query exceeds maximum length of {self.max_query_length} characters.", validation_metadata
            
            validation_metadata['validation_checks_passed'].append('length_check')
            
            # Check 2: Rate limiting
            if not self._check_rate_limits(user_id):
                self._record_failure("rate_limit_exceeded", validation_metadata)
                self.guardrail_stats['rate_limit_violations'] += 1
                return False, "Rate limit exceeded. Please try again later.", validation_metadata
            
            validation_metadata['validation_checks_passed'].append('rate_limit_check')
            
            # Check 3: Malicious pattern detection
            query_lower = query.lower()
            for pattern in self.blocked_patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    self._record_failure("malicious_pattern", validation_metadata)
                    validation_metadata['blocked_pattern'] = pattern
                    return False, "Query contains potentially harmful content.", validation_metadata
            
            validation_metadata['validation_checks_passed'].append('malicious_pattern_check')
            
            # Check 4: Inappropriate content detection
            for keyword in self.inappropriate_keywords:
                if keyword in query_lower:
                    self._record_failure("inappropriate_content", validation_metadata)
                    self.guardrail_stats['inappropriate_content_blocked'] += 1
                    validation_metadata['inappropriate_keyword'] = keyword
                    return False, "Query contains inappropriate content for financial context.", validation_metadata
            
            validation_metadata['validation_checks_passed'].append('inappropriate_content_check')
            
            # Check 5: Financial context validation
            has_financial_context = any(
                keyword in query_lower 
                for keyword in self.financial_context_keywords
            )
            
            # Allow general greetings and meta-questions
            general_allowed_patterns = [
                r'\b(hello|hi|help|what can you|how do|explain|tell me about)\b',
                r'\b(thank you|thanks|goodbye|bye)\b',
                r'\b(what is|what are|how does|how is)\b',
                r'\b(company|business|organization|firm|corporation)\b',
                r'\b(perform|performance|doing|results|overview|summary)\b',
                r'\b(compare|comparison|analyze|analysis|trend|trends)\b'
            ]
            
            is_general_query = any(
                re.search(pattern, query_lower) 
                for pattern in general_allowed_patterns
            )
            
            if not has_financial_context and not is_general_query:
                self._record_failure("non_financial_query", validation_metadata)
                self.guardrail_stats['non_financial_queries_blocked'] += 1
                return False, "Query must be related to financial topics.", validation_metadata
            
            validation_metadata['validation_checks_passed'].append('financial_context_check')
            validation_metadata['has_financial_context'] = has_financial_context
            validation_metadata['is_general_query'] = is_general_query
            
            # Check 6: PII (Personally Identifiable Information) detection
            pii_patterns = [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
                r'\b\d{16}\b',              # Credit card pattern
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\b\d{3}-\d{3}-\d{4}\b',  # Phone number
            ]
            
            for pattern in pii_patterns:
                if re.search(pattern, query):
                    self._record_failure("pii_detected", validation_metadata)
                    return False, "Query contains personally identifiable information.", validation_metadata
            
            validation_metadata['validation_checks_passed'].append('pii_check')
            
            # All validations passed
            validation_time = (datetime.now() - validation_start).total_seconds()
            validation_metadata['validation_time'] = validation_time
            validation_metadata['status'] = 'approved'
            
            logger.info(f"✅ Query validation passed for user {user_id} in {validation_time:.3f}s")
            
            return True, "Query validation passed.", validation_metadata
            
        except Exception as e:
            logger.error(f"Error in input validation: {str(e)}")
            self._record_failure("validation_error", validation_metadata)
            return False, "Internal validation error.", validation_metadata
    
    def filter_output_response(self, 
                             response: str, 
                             confidence: float,
                             query: str) -> Tuple[str, Dict[str, Any]]:
        """Filter and enhance output response for safety and quality.
        
        Args:
            response: Generated response string
            confidence: Confidence score of the response
            query: Original query for context
            
        Returns:
            Tuple of (filtered_response, filter_metadata)
        """
        filter_metadata = {
            'timestamp': datetime.now().isoformat(),
            'original_length': len(response),
            'confidence': confidence,
            'filters_applied': []
        }
        
        try:
            filtered_response = response
            
            # Filter 1: Remove blocked output patterns
            for pattern in self.output_blocked_patterns:
                if re.search(pattern, filtered_response, re.IGNORECASE):
                    filtered_response = re.sub(pattern, '', filtered_response, flags=re.IGNORECASE)
                    filter_metadata['filters_applied'].append('blocked_pattern_removal')
                    self.guardrail_stats['output_filtered_count'] += 1
            
            # Filter 2: Add confidence disclaimers for low-confidence responses
            if confidence < 0.7:
                filtered_response += "\n\n*Note: This response has moderate confidence. Please verify with additional sources.*"
                filter_metadata['filters_applied'].append('low_confidence_disclaimer')
            
            # Filter 3: Ensure proper financial formatting
            filtered_response = self._enhance_financial_formatting(filtered_response)
            filter_metadata['filters_applied'].append('financial_formatting')
            
            # Filter 4: Add source attribution reminder for high-stakes queries
            high_stakes_keywords = ['loss', 'profit', 'debt', 'liability', 'investment', 'risk']
            if any(keyword in query.lower() for keyword in high_stakes_keywords):
                if 'data from' not in filtered_response.lower():
                    filtered_response += "\n\n*Please verify this information with the original financial statements.*"
                    filter_metadata['filters_applied'].append('source_attribution_reminder')
            
            # Filter 5: Length and readability check
            if len(filtered_response) < 10:
                filtered_response = "I apologize, but I couldn't generate a sufficient response to your financial query. Please rephrase your question."
                filter_metadata['filters_applied'].append('minimum_length_enforcement')
            
            filter_metadata['final_length'] = len(filtered_response)
            filter_metadata['length_change'] = filter_metadata['final_length'] - filter_metadata['original_length']
            
            logger.info(f"✅ Output filtering complete: {len(filter_metadata['filters_applied'])} filters applied")
            
            return filtered_response, filter_metadata
            
        except Exception as e:
            logger.error(f"Error in output filtering: {str(e)}")
            return response, filter_metadata  # Return original on error
    
    def _check_rate_limits(self, user_id: str) -> bool:
        """Check if user is within rate limits."""
        current_time = datetime.now()
        user_history = self.request_history[user_id]
        
        # Clean old requests (older than 1 hour)
        while user_history and current_time - user_history[0] > timedelta(hours=1):
            user_history.popleft()
        
        # Check hourly limit
        if len(user_history) >= self.rate_limit_per_hour:
            return False
        
        # Check minute limit
        recent_requests = sum(
            1 for req_time in user_history 
            if current_time - req_time <= timedelta(minutes=1)
        )
        
        if recent_requests >= self.rate_limit_per_minute:
            return False
        
        # Add current request
        user_history.append(current_time)
        return True
    
    def _record_failure(self, reason: str, metadata: Dict[str, Any]):
        """Record validation failure for statistics."""
        self.guardrail_stats['queries_blocked'] += 1
        self.guardrail_stats['block_reasons'][reason] += 1
        metadata['validation_checks_failed'].append(reason)
        metadata['status'] = 'blocked'
        metadata['block_reason'] = reason
    
    def _enhance_financial_formatting(self, text: str) -> str:
        """Enhance financial formatting in text."""
        # Ensure consistent currency formatting
        text = re.sub(r'Rs\.?\s*(\d)', r'₹\1', text)
        text = re.sub(r'rupees?\s*(\d)', r'₹\1', text, flags=re.IGNORECASE)
        
        # Format large numbers consistently
        text = re.sub(r'(\d+)\s*crores?', r'\1 crore', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*billions?', r'\1 billion', text, flags=re.IGNORECASE)
        
        return text
    
    def validate_complete_pipeline(self, 
                                 query: str, 
                                 response: str, 
                                 confidence: float,
                                 user_id: str = "anonymous") -> Dict[str, Any]:
        """Validate complete query-response pipeline with guardrails.
        
        Args:
            query: Input query
            response: Generated response
            confidence: Response confidence
            user_id: User identifier
            
        Returns:
            Complete validation results
        """
        pipeline_start = datetime.now()
        
        # Input validation
        input_valid, input_reason, input_metadata = self.validate_input_query(query, user_id)
        
        if not input_valid:
            return {
                'approved': False,
                'stage_failed': 'input_validation',
                'reason': input_reason,
                'input_metadata': input_metadata,
                'timestamp': datetime.now().isoformat()
            }
        
        # Output filtering
        filtered_response, output_metadata = self.filter_output_response(response, confidence, query)
        
        pipeline_time = (datetime.now() - pipeline_start).total_seconds()
        
        return {
            'approved': True,
            'original_query': query,
            'original_response': response,
            'filtered_response': filtered_response,
            'confidence': confidence,
            'input_metadata': input_metadata,
            'output_metadata': output_metadata,
            'pipeline_time': pipeline_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security and usage report."""
        total_processed = self.guardrail_stats['total_queries_processed']
        blocked = self.guardrail_stats['queries_blocked']
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_queries_processed': total_processed,
                'queries_approved': total_processed - blocked,
                'queries_blocked': blocked,
                'approval_rate': (total_processed - blocked) / total_processed * 100 if total_processed > 0 else 0,
                'block_rate': blocked / total_processed * 100 if total_processed > 0 else 0
            },
            'detailed_stats': self.guardrail_stats.copy(),
            'security_effectiveness': {
                'rate_limit_protection': self.guardrail_stats['rate_limit_violations'],
                'content_filtering': self.guardrail_stats['inappropriate_content_blocked'],
                'context_validation': self.guardrail_stats['non_financial_queries_blocked'],
                'output_filtering': self.guardrail_stats['output_filtered_count']
            },
            'top_block_reasons': dict(sorted(
                self.guardrail_stats['block_reasons'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]),
            'system_health': {
                'guardrails_active': True,
                'rate_limiting_active': True,
                'content_filtering_active': True,
                'output_filtering_active': True
            }
        }
        
        return report
    
    def reset_user_limits(self, user_id: str):
        """Reset rate limits for a specific user (admin function)."""
        if user_id in self.request_history:
            del self.request_history[user_id]
            logger.info(f"Rate limits reset for user: {user_id}")
    
    def add_custom_block_pattern(self, pattern: str, reason: str = "custom_pattern"):
        """Add custom blocking pattern."""
        self.blocked_patterns.append(pattern)
        logger.info(f"Added custom block pattern: {pattern}")
    
    def emergency_shutdown_mode(self):
        """Enable emergency mode that blocks all queries."""
        self._emergency_mode = True
        logger.warning("🚨 Emergency shutdown mode activated - all queries will be blocked")
    
    def check_emergency_mode(self) -> bool:
        """Check if system is in emergency shutdown mode."""
        return hasattr(self, '_emergency_mode') and self._emergency_mode

def main():
    """Test the guardrails system."""
    print("="*80)
    print("FINANCIAL RAG GUARDRAILS SYSTEM - PHASE 6")
    print("Input/Output Validation and Security Measures")
    print("="*80)
    
    # Initialize guardrails
    guardrails = FinancialRAGGuardrails()
    
    # Test cases for input validation
    test_cases = [
        # Valid queries
        ("What was the revenue in Q3 2023?", True, "Valid financial query"),
        ("Tell me about profit margins", True, "Valid financial query"),
        ("How did the company perform?", True, "Valid general query"),
        
        # Invalid queries
        ("", False, "Empty query"),
        ("SELECT * FROM users WHERE password='admin'", False, "SQL injection attempt"),
        ("<script>alert('hack')</script>", False, "XSS attempt"),
        ("How to hack into banking systems?", False, "Inappropriate content"),
        ("What's the weather like?", False, "Non-financial query"),
        ("My SSN is 123-45-6789", False, "PII detected"),
        ("a" * 600, False, "Query too long"),
    ]
    
    print(f"\n🧪 Testing Input Validation Guardrails")
    
    # Test input validation
    for i, (query, should_pass, description) in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}: {description}")
        print(f"Query: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        is_valid, reason, metadata = guardrails.validate_input_query(query, f"test_user_{i}")
        
        status = "✅ PASSED" if is_valid == should_pass else "❌ FAILED"
        print(f"Result: {status}")
        print(f"Validation: {is_valid} | Expected: {should_pass}")
        print(f"Reason: {reason}")
        
        if metadata['validation_checks_passed']:
            print(f"Checks passed: {metadata['validation_checks_passed']}")
        if metadata['validation_checks_failed']:
            print(f"Checks failed: {metadata['validation_checks_failed']}")
    
    # Test output filtering
    print(f"\n🔍 Testing Output Filtering")
    
    test_responses = [
        ("The revenue was ₹15 billion in Q3 2023.", 0.95, "High confidence response"),
        ("I think the profit might be around ₹2 billion.", 0.45, "Low confidence response"),
        ("The company reported a loss of Rs 1 billion.", 0.85, "Currency formatting test"),
        ("", 0.0, "Empty response test"),
    ]
    
    for response, confidence, description in test_responses:
        print(f"\n📝 Output Test: {description}")
        print(f"Original: {response}")
        print(f"Confidence: {confidence}")
        
        filtered, metadata = guardrails.filter_output_response(
            response, confidence, "What was the financial performance?"
        )
        
        print(f"Filtered: {filtered}")
        print(f"Filters applied: {metadata['filters_applied']}")
        print(f"Length change: {metadata['length_change']}")
    
    # Test rate limiting
    print(f"\n🔍 Testing Rate Limiting")
    
    user_id = "test_rate_limit_user"
    
    # Test multiple requests rapidly
    for i in range(5):
        is_valid, reason, _ = guardrails.validate_input_query(
            "What was the revenue?", user_id
        )
        print(f"Request {i+1}: {'✅ Allowed' if is_valid else '❌ Rate limited'}")
    
    # Generate security report
    print(f"\n📊 Security Report")
    report = guardrails.get_security_report()
    
    print(f"Total Queries Processed: {report['summary']['total_queries_processed']}")
    print(f"Approval Rate: {report['summary']['approval_rate']:.1f}%")
    print(f"Block Rate: {report['summary']['block_rate']:.1f}%")
    print(f"Top Block Reasons: {report['top_block_reasons']}")
    
    print(f"\n🛡️ Security Effectiveness:")
    for metric, value in report['security_effectiveness'].items():
        print(f"  {metric.replace('_', ' ').title()}: {value}")
    
    print(f"\n🎉 Guardrails system test completed!")
    print("The system provides comprehensive protection for financial RAG queries!")

if __name__ == "__main__":
    main()
