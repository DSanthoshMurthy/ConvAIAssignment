#!/usr/bin/env python3
"""
Secured Financial RAG Pipeline
Integration of Complete RAG Pipeline with Comprehensive Guardrails
"""

import logging
from typing import List, Dict, Any, Optional
# Fix imports for direct execution
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from complete_rag_pipeline import CompleteFinancialRAG
from guardrails import FinancialRAGGuardrails
import time
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecuredFinancialRAG:
    def __init__(self, 
                 indexes_dir: str = "data/indexes",
                 chroma_db_dir: str = "data/indexes/chroma_db",
                 enable_cross_encoder: bool = True,
                 enable_strict_guardrails: bool = True):
        """Initialize the Secured Financial RAG system.
        
        Args:
            indexes_dir: Directory containing all indexes
            chroma_db_dir: ChromaDB persistence directory
            enable_cross_encoder: Whether to enable cross-encoder re-ranking
            enable_strict_guardrails: Whether to enable strict security measures
        """
        # Initialize core RAG pipeline
        self.rag_pipeline = CompleteFinancialRAG(
            indexes_dir=indexes_dir,
            chroma_db_dir=chroma_db_dir,
            enable_cross_encoder=enable_cross_encoder
        )
        
        # Initialize guardrails system
        self.guardrails = FinancialRAGGuardrails(
            max_query_length=500,
            rate_limit_per_minute=30,  # More conservative for production
            rate_limit_per_hour=500
        )
        
        self.enable_strict_guardrails = enable_strict_guardrails
        self.loaded = False
        
        # Security and performance statistics
        self.secured_stats = {
            'total_requests': 0,
            'approved_requests': 0,
            'blocked_requests': 0,
            'average_processing_time': 0.0,
            'security_incidents': 0,
            'high_confidence_responses': 0,
            'low_confidence_responses': 0
        }
    
    def load_system(self) -> bool:
        """Load all components of the secured RAG system."""
        try:
            logger.info("🚀 Loading Secured Financial RAG System...")
            
            # Load core RAG pipeline
            if not self.rag_pipeline.load_system():
                logger.error("Failed to load core RAG pipeline")
                return False
            
            logger.info("✅ Core RAG pipeline loaded")
            logger.info("✅ Security guardrails active")
            
            self.loaded = True
            logger.info("🎉 Secured Financial RAG System ready for production!")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading secured RAG system: {str(e)}")
            return False
    
    def secure_query_processing(self, 
                              query: str, 
                              user_id: str = "anonymous",
                              top_k: int = 5,
                              fusion_method: str = 'weighted',
                              include_explanation: bool = False) -> Dict[str, Any]:
        """Process a query through the complete secured RAG pipeline.
        
        Args:
            query: User query string
            user_id: Unique user identifier for security tracking
            top_k: Number of relevant chunks to retrieve
            fusion_method: 'weighted' or 'rrf' for hybrid fusion
            include_explanation: Whether to include detailed explanation
            
        Returns:
            Secured response with validation metadata
        """
        if not self.loaded:
            raise RuntimeError("Secured RAG system not loaded. Call load_system() first.")
        
        request_start = datetime.now()
        self.secured_stats['total_requests'] += 1
        
        logger.info(f"🔒 Processing secured query from user {user_id}: '{query[:50]}...'")
        
        try:
            # Stage 1: Input Validation with Guardrails
            validation_start = time.time()
            
            if self.enable_strict_guardrails:
                is_valid, validation_reason, validation_metadata = self.guardrails.validate_input_query(query, user_id)
                
                if not is_valid:
                    self.secured_stats['blocked_requests'] += 1
                    self.secured_stats['security_incidents'] += 1
                    
                    logger.warning(f"🚫 Query blocked for user {user_id}: {validation_reason}")
                    
                    return {
                        'status': 'blocked',
                        'reason': validation_reason,
                        'user_id': user_id,
                        'validation_metadata': validation_metadata,
                        'timestamp': datetime.now().isoformat(),
                        'processing_stage': 'input_validation'
                    }
                
                logger.info(f"✅ Input validation passed in {time.time() - validation_start:.3f}s")
            
            # Stage 2: RAG Pipeline Processing
            rag_start = time.time()
            
            rag_result = self.rag_pipeline.process_query(
                query=query,
                top_k=top_k,
                fusion_method=fusion_method,
                include_explanation=include_explanation
            )
            
            rag_time = time.time() - rag_start
            logger.info(f"✅ RAG processing completed in {rag_time:.3f}s")
            
            # Stage 3: Output Filtering and Enhancement
            filter_start = time.time()
            
            original_answer = rag_result.get('answer', '')
            confidence = rag_result.get('confidence', 0.0)
            
            if self.enable_strict_guardrails:
                filtered_answer, filter_metadata = self.guardrails.filter_output_response(
                    original_answer, confidence, query
                )
            else:
                filtered_answer = original_answer
                filter_metadata = {'filters_applied': []}
            
            filter_time = time.time() - filter_start
            logger.info(f"✅ Output filtering completed in {filter_time:.3f}s")
            
            # Update statistics
            self.secured_stats['approved_requests'] += 1
            
            if confidence >= 0.8:
                self.secured_stats['high_confidence_responses'] += 1
            elif confidence < 0.6:
                self.secured_stats['low_confidence_responses'] += 1
            
            # Calculate total processing time
            total_time = (datetime.now() - request_start).total_seconds()
            
            # Update average processing time
            n = self.secured_stats['approved_requests']
            self.secured_stats['average_processing_time'] = (
                (self.secured_stats['average_processing_time'] * (n-1) + total_time) / n
            )
            
            # Build secured response
            secured_response = {
                'status': 'approved',
                'query': query,
                'answer': filtered_answer,
                'confidence': confidence,
                'user_id': user_id,
                'sources': rag_result.get('sources', []),
                'processing_metadata': {
                    'total_time': total_time,
                    'validation_time': validation_start if self.enable_strict_guardrails else 0,
                    'rag_processing_time': rag_time,
                    'output_filtering_time': filter_time,
                    'guardrails_enabled': self.enable_strict_guardrails
                },
                'security_metadata': {
                    'input_validated': self.enable_strict_guardrails,
                    'output_filtered': self.enable_strict_guardrails,
                    'validation_checks_passed': validation_metadata.get('validation_checks_passed', []) if self.enable_strict_guardrails else [],
                    'filters_applied': filter_metadata.get('filters_applied', [])
                },
                'rag_metadata': rag_result.get('retrieval_metadata', {}),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add explanation if requested
            if include_explanation:
                secured_response['explanation'] = rag_result.get('explanation', {})
                secured_response['explanation']['security_explanation'] = {
                    'guardrails_applied': self.enable_strict_guardrails,
                    'validation_summary': validation_metadata if self.enable_strict_guardrails else {},
                    'output_enhancement_summary': filter_metadata
                }
            
            logger.info(f"✅ Secured query processing complete: confidence {confidence:.2f} in {total_time:.3f}s")
            
            return secured_response
            
        except Exception as e:
            logger.error(f"Error in secured query processing: {str(e)}")
            self.secured_stats['security_incidents'] += 1
            
            return {
                'status': 'error',
                'reason': f"Internal processing error: {str(e)}",
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'processing_stage': 'internal_error'
            }
    
    def batch_secure_processing(self, 
                              queries: List[str], 
                              user_id: str = "batch_user",
                              top_k: int = 5) -> List[Dict[str, Any]]:
        """Process multiple queries with security measures."""
        logger.info(f"🔄 Processing {len(queries)} queries in secured batch mode")
        
        results = []
        batch_start = time.time()
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing secured batch query {i}/{len(queries)}")
            
            try:
                result = self.secure_query_processing(
                    query=query,
                    user_id=f"{user_id}_batch_{i}",
                    top_k=top_k
                )
                results.append(result)
                
                # Log result
                status = result.get('status', 'unknown')
                confidence = result.get('confidence', 0)
                logger.info(f"✓ Batch query {i} {status} (confidence: {confidence:.2f})")
                
            except Exception as e:
                logger.error(f"Error processing batch query {i}: {str(e)}")
                results.append({
                    'status': 'error',
                    'query': query,
                    'reason': str(e),
                    'batch_index': i
                })
        
        batch_time = time.time() - batch_start
        
        # Batch statistics
        approved = sum(1 for r in results if r.get('status') == 'approved')
        blocked = sum(1 for r in results if r.get('status') == 'blocked')
        errors = sum(1 for r in results if r.get('status') == 'error')
        
        logger.info(f"✅ Secured batch processing complete: {approved} approved, {blocked} blocked, {errors} errors in {batch_time:.2f}s")
        
        return results
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive security and performance dashboard."""
        # Get guardrails report
        guardrails_report = self.guardrails.get_security_report()
        
        # Get RAG system stats
        rag_stats = self.rag_pipeline.get_system_stats()
        
        # Calculate security metrics
        total_requests = self.secured_stats['total_requests']
        approval_rate = (self.secured_stats['approved_requests'] / total_requests * 100) if total_requests > 0 else 0
        
        dashboard = {
            'dashboard_timestamp': datetime.now().isoformat(),
            'system_status': {
                'rag_pipeline_loaded': self.rag_pipeline.loaded,
                'guardrails_active': self.enable_strict_guardrails,
                'cross_encoder_enabled': self.rag_pipeline.retriever.enable_reranking,
                'emergency_mode': self.guardrails.check_emergency_mode()
            },
            'security_metrics': {
                'total_requests_processed': total_requests,
                'approval_rate_percentage': approval_rate,
                'blocked_requests': self.secured_stats['blocked_requests'],
                'security_incidents': self.secured_stats['security_incidents'],
                'average_processing_time': self.secured_stats['average_processing_time']
            },
            'response_quality': {
                'high_confidence_responses': self.secured_stats['high_confidence_responses'],
                'low_confidence_responses': self.secured_stats['low_confidence_responses'],
                'average_rag_confidence': rag_stats.get('avg_confidence', 0),
                'response_time_average': rag_stats.get('total_time', 0)
            },
            'guardrails_effectiveness': guardrails_report['security_effectiveness'],
            'system_configuration': {
                'max_query_length': self.guardrails.max_query_length,
                'rate_limit_per_minute': self.guardrails.rate_limit_per_minute,
                'rate_limit_per_hour': self.guardrails.rate_limit_per_hour,
                'strict_guardrails': self.enable_strict_guardrails
            },
            'detailed_stats': {
                'secured_rag_stats': self.secured_stats,
                'guardrails_stats': guardrails_report['detailed_stats'],
                'rag_pipeline_stats': rag_stats
            }
        }
        
        return dashboard
    
    def emergency_lockdown(self):
        """Emergency lockdown that blocks all requests."""
        self.guardrails.emergency_shutdown_mode()
        logger.critical("🚨 EMERGENCY LOCKDOWN ACTIVATED - All queries will be blocked")
    
    def reset_user_access(self, user_id: str):
        """Reset access controls for a specific user."""
        self.guardrails.reset_user_limits(user_id)
        logger.info(f"Access controls reset for user: {user_id}")

def main():
    """Test the secured RAG pipeline."""
    print("="*80)
    print("SECURED FINANCIAL RAG PIPELINE - FINAL PRODUCTION SYSTEM")
    print("Complete RAG + Advanced Security Guardrails")
    print("="*80)
    
    # Initialize secured RAG system
    secured_rag = SecuredFinancialRAG(enable_strict_guardrails=True)
    
    # Load system
    if not secured_rag.load_system():
        print("❌ Failed to load secured RAG system")
        return
    
    # Test queries including security challenges
    test_queries = [
        # Valid financial queries
        "What was the revenue from operations in Q3 2023?",
        "How did the company perform financially?",
        "Tell me about employee benefit expenses",
        
        # Security test cases
        "<script>alert('xss')</script>",
        "DROP TABLE users;",
        "How to hack banking systems?",
        "What's the weather like today?",  # Non-financial
    ]
    
    print(f"\n🧪 Testing Secured RAG Pipeline")
    print(f"Guardrails enabled: {secured_rag.enable_strict_guardrails}")
    print(f"Cross-encoder re-ranking: {secured_rag.rag_pipeline.retriever.enable_reranking}")
    
    test_results = []
    
    # Process each query
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Secured Query {i}: {query}")
        print("="*70)
        
        # Process with security
        result = secured_rag.secure_query_processing(
            query=query,
            user_id=f"test_user_{i}",
            top_k=3,
            include_explanation=False
        )
        
        test_results.append(result)
        
        # Display results
        status = result.get('status', 'unknown')
        print(f"🔒 Security Status: {status.upper()}")
        
        if status == 'approved':
            print(f"🤖 Answer: {result['answer']}")
            print(f"📊 Confidence: {result['confidence']:.2f}")
            print(f"⚡ Processing Time: {result['processing_metadata']['total_time']:.3f}s")
            print(f"🛡️ Security Checks: {len(result['security_metadata']['validation_checks_passed'])} passed")
            print(f"🔧 Filters Applied: {result['security_metadata']['filters_applied']}")
        elif status == 'blocked':
            print(f"🚫 Blocked Reason: {result['reason']}")
            print(f"🛡️ Security Stage: {result['processing_stage']}")
        elif status == 'error':
            print(f"❌ Error: {result['reason']}")
    
    # Generate security dashboard
    print(f"\n📊 Security Dashboard")
    print("="*50)
    
    dashboard = secured_rag.get_security_dashboard()
    
    print(f"System Status:")
    for key, value in dashboard['system_status'].items():
        print(f"  {key.replace('_', ' ').title()}: {'✅' if value else '❌'} {value}")
    
    print(f"\nSecurity Metrics:")
    metrics = dashboard['security_metrics']
    print(f"  Total Requests: {metrics['total_requests_processed']}")
    print(f"  Approval Rate: {metrics['approval_rate_percentage']:.1f}%")
    print(f"  Blocked Requests: {metrics['blocked_requests']}")
    print(f"  Security Incidents: {metrics['security_incidents']}")
    print(f"  Avg Processing Time: {metrics['average_processing_time']:.3f}s")
    
    print(f"\nResponse Quality:")
    quality = dashboard['response_quality']
    print(f"  High Confidence: {quality['high_confidence_responses']}")
    print(f"  Low Confidence: {quality['low_confidence_responses']}")
    
    print(f"\nGuardrails Effectiveness:")
    effectiveness = dashboard['guardrails_effectiveness']
    for metric, value in effectiveness.items():
        print(f"  {metric.replace('_', ' ').title()}: {value}")
    
    print(f"\n🎯 System Configuration:")
    config = dashboard['system_configuration']
    print(f"  Max Query Length: {config['max_query_length']} chars")
    print(f"  Rate Limit/Minute: {config['rate_limit_per_minute']}")
    print(f"  Rate Limit/Hour: {config['rate_limit_per_hour']}")
    print(f"  Strict Guardrails: {'✅' if config['strict_guardrails'] else '❌'}")
    
    print(f"\n🎉 Secured RAG pipeline test completed!")
    print("The system provides enterprise-grade security for financial queries!")
    print(f"Approval rate: {dashboard['security_metrics']['approval_rate_percentage']:.1f}% with comprehensive protection!")

if __name__ == "__main__":
    main()
