#!/usr/bin/env python3
"""
Data Loading Diagnostic System
Comprehensive diagnosis and recovery for cloud data loading failures
"""

import logging
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple
import os
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoadingDiagnostic:
    def __init__(self, base_dir: str = None):
        """Initialize data loading diagnostic system."""
        if base_dir is None:
            # Auto-detect base directory
            current_file = Path(__file__).resolve()
            self.base_dir = current_file.parent.parent.parent  # Go up to project root
        else:
            self.base_dir = Path(base_dir)
        
        self.indexes_dir = self.base_dir / "data" / "indexes"
        self.chunks_dir = self.base_dir / "data" / "chunks" 
        self.processed_dir = self.base_dir / "data" / "processed"
        
        self.diagnostic_results = {}
        
    def run_complete_diagnostic(self) -> Dict[str, Any]:
        """Run complete data loading diagnostic."""
        logger.info("🔍 Starting comprehensive data loading diagnostic...")
        
        results = {
            'base_directory': str(self.base_dir),
            'deployment_mode': self._detect_deployment_mode(),
            'file_system_check': self._check_file_system(),
            'critical_files_check': self._check_critical_files(),
            'data_integrity_check': self._check_data_integrity(),
            'permissions_check': self._check_file_permissions(),
            'size_analysis': self._analyze_file_sizes(),
            'recovery_suggestions': self._generate_recovery_suggestions()
        }
        
        self.diagnostic_results = results
        self._log_diagnostic_summary(results)
        
        return results
    
    def _detect_deployment_mode(self) -> str:
        """Detect if running locally or in cloud."""
        base_path = str(self.base_dir)
        
        if '/Users/' in base_path or '/home/' in base_path:
            return "local"
        elif '/mount/' in base_path or '/app/' in base_path or '/workspace/' in base_path:
            return "cloud"
        else:
            return "unknown"
    
    def _check_file_system(self) -> Dict[str, Any]:
        """Check basic file system structure."""
        fs_check = {}
        
        # Check main directories
        dirs_to_check = [
            self.base_dir / "data",
            self.indexes_dir,
            self.chunks_dir,
            self.processed_dir,
            self.indexes_dir / "chroma_db"
        ]
        
        for directory in dirs_to_check:
            rel_path = directory.relative_to(self.base_dir)
            fs_check[str(rel_path)] = {
                'exists': directory.exists(),
                'is_dir': directory.is_dir() if directory.exists() else False,
                'readable': os.access(directory, os.R_OK) if directory.exists() else False,
                'file_count': len(list(directory.iterdir())) if directory.exists() and directory.is_dir() else 0
            }
        
        return fs_check
    
    def _check_critical_files(self) -> Dict[str, Any]:
        """Check critical data files required for RAG system."""
        critical_files = {
            # Index files
            'bm25_index': self.indexes_dir / "bm25_index.pkl",
            'bm25_chunk_mapping': self.indexes_dir / "bm25_chunk_mapping.json", 
            'index_metadata': self.indexes_dir / "index_metadata.json",
            
            # ChromaDB files
            'chroma_sqlite': self.indexes_dir / "chroma_db" / "chroma.sqlite3",
            
            # Chunk data
            'financial_chunks': self.chunks_dir / "financial_chunks_100_tokens.json",
            'chunk_statistics': self.chunks_dir / "chunk_statistics.json",
            
            # Q&A data
            'qa_pairs': self.processed_dir / "xbrl_qa_pairs.json",
        }
        
        file_check = {}
        
        for name, filepath in critical_files.items():
            if filepath.exists():
                try:
                    size_mb = filepath.stat().st_size / (1024 * 1024)
                    file_check[name] = {
                        'status': 'available',
                        'size_mb': round(size_mb, 2),
                        'readable': os.access(filepath, os.R_OK),
                        'path': str(filepath)
                    }
                    
                    # Additional checks for specific file types
                    if name == 'bm25_index':
                        file_check[name]['loadable'] = self._test_pickle_load(filepath)
                    elif name.endswith('json'):
                        file_check[name]['valid_json'] = self._test_json_load(filepath)
                        
                except Exception as e:
                    file_check[name] = {
                        'status': 'error',
                        'error': str(e),
                        'path': str(filepath)
                    }
            else:
                file_check[name] = {
                    'status': 'missing',
                    'path': str(filepath)
                }
        
        return file_check
    
    def _test_pickle_load(self, filepath: Path) -> bool:
        """Test if pickle file can be loaded."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            return True
        except Exception:
            return False
    
    def _test_json_load(self, filepath: Path) -> bool:
        """Test if JSON file can be loaded."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return True
        except Exception:
            return False
    
    def _check_data_integrity(self) -> Dict[str, Any]:
        """Check data integrity and consistency."""
        integrity = {}
        
        # Check chunk consistency
        try:
            chunks_file = self.chunks_dir / "financial_chunks_100_tokens.json"
            mapping_file = self.indexes_dir / "bm25_chunk_mapping.json"
            
            if chunks_file.exists() and mapping_file.exists():
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                    
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                
                integrity['chunk_consistency'] = {
                    'chunks_count': len(chunks),
                    'mapping_count': len(mapping),
                    'consistent': len(chunks) == len(mapping)
                }
            else:
                integrity['chunk_consistency'] = {'error': 'Required files missing'}
                
        except Exception as e:
            integrity['chunk_consistency'] = {'error': str(e)}
        
        # Check Q&A data
        try:
            qa_file = self.processed_dir / "xbrl_qa_pairs.json"
            if qa_file.exists():
                with open(qa_file, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)
                    
                integrity['qa_data'] = {
                    'total_pairs': len(qa_data),
                    'has_revenue_data': any('revenue' in str(item).lower() for item in qa_data[:10])
                }
        except Exception as e:
            integrity['qa_data'] = {'error': str(e)}
        
        return integrity
    
    def _check_file_permissions(self) -> Dict[str, str]:
        """Check file permissions for critical files."""
        permissions = {}
        
        # Check if we can read data directory
        for dir_name in ['data', 'data/indexes', 'data/chunks', 'data/processed']:
            dir_path = self.base_dir / dir_name
            if dir_path.exists():
                permissions[dir_name] = {
                    'readable': os.access(dir_path, os.R_OK),
                    'writable': os.access(dir_path, os.W_OK),
                    'executable': os.access(dir_path, os.X_OK)
                }
        
        return permissions
    
    def _analyze_file_sizes(self) -> Dict[str, Any]:
        """Analyze file sizes and detect potential issues."""
        analysis = {}
        
        # Expected file sizes (in MB)
        expected_sizes = {
            'bm25_index.pkl': (5, 50),  # 5-50 MB expected
            'financial_chunks_100_tokens.json': (2, 20),  # 2-20 MB
            'xbrl_qa_pairs.json': (0.1, 10),  # 0.1-10 MB
            'chroma.sqlite3': (10, 100)  # 10-100 MB
        }
        
        for filename, (min_size, max_size) in expected_sizes.items():
            file_found = False
            for root, dirs, files in os.walk(self.base_dir / "data"):
                if filename in files:
                    filepath = Path(root) / filename
                    size_mb = filepath.stat().st_size / (1024 * 1024)
                    
                    status = "normal"
                    if size_mb < min_size:
                        status = "too_small"
                    elif size_mb > max_size:
                        status = "too_large"
                    
                    analysis[filename] = {
                        'size_mb': round(size_mb, 2),
                        'status': status,
                        'path': str(filepath)
                    }
                    file_found = True
                    break
            
            if not file_found:
                analysis[filename] = {'status': 'missing'}
        
        return analysis
    
    def _generate_recovery_suggestions(self) -> List[str]:
        """Generate recovery suggestions based on diagnostic results."""
        suggestions = []
        
        # Check what we've found so far
        if not hasattr(self, 'diagnostic_results') or not self.diagnostic_results:
            return ["Run diagnostic first"]
        
        # Generic suggestions based on common issues
        suggestions.extend([
            "Verify all data files are properly committed to git repository",
            "Check .gitignore to ensure data files are not excluded",
            "Ensure cloud deployment includes data directory",
            "Verify file permissions in cloud environment",
            "Check for SQLite version compatibility issues",
            "Consider re-generating indexes if corrupted"
        ])
        
        return suggestions
    
    def _log_diagnostic_summary(self, results: Dict[str, Any]) -> None:
        """Log diagnostic summary."""
        logger.info("=" * 80)
        logger.info("🔍 DATA LOADING DIAGNOSTIC SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"📁 Base Directory: {results['base_directory']}")
        logger.info(f"🌍 Deployment Mode: {results['deployment_mode']}")
        
        # Log critical issues
        critical_issues = []
        
        file_check = results.get('critical_files_check', {})
        for filename, info in file_check.items():
            if info.get('status') == 'missing':
                critical_issues.append(f"Missing: {filename}")
            elif info.get('status') == 'error':
                critical_issues.append(f"Error loading: {filename}")
        
        if critical_issues:
            logger.error("🚨 CRITICAL ISSUES FOUND:")
            for issue in critical_issues:
                logger.error(f"  - {issue}")
        else:
            logger.info("✅ No critical issues found")
        
        logger.info("=" * 80)

def run_diagnostic():
    """Run diagnostic as standalone script."""
    diagnostic = DataLoadingDiagnostic()
    results = diagnostic.run_complete_diagnostic()
    
    # Print JSON results for programmatic access
    print("\n" + "="*80)
    print("DIAGNOSTIC RESULTS (JSON)")
    print("="*80)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_diagnostic()
