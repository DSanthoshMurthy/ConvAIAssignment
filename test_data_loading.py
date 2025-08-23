#!/usr/bin/env python3
"""
Quick Data Loading Test
Run this to quickly test if all data files are loading correctly
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.rag.data_loader_diagnostic import DataLoadingDiagnostic

def main():
    print("🔍 Running quick data loading test...")
    print("="*60)
    
    # Run diagnostic
    diagnostic = DataLoadingDiagnostic()
    results = diagnostic.run_complete_diagnostic()
    
    # Quick summary
    print("\n📊 QUICK SUMMARY:")
    print("-" * 40)
    
    deployment_mode = results.get('deployment_mode', 'unknown')
    print(f"🌍 Mode: {deployment_mode.upper()}")
    
    # Check critical files
    critical_files = results.get('critical_files_check', {})
    missing_files = []
    available_files = []
    error_files = []
    
    for filename, info in critical_files.items():
        status = info.get('status', 'unknown')
        if status == 'available':
            available_files.append(filename)
        elif status == 'missing':
            missing_files.append(filename)
        elif status == 'error':
            error_files.append(filename)
    
    print(f"✅ Available: {len(available_files)} files")
    print(f"❌ Missing: {len(missing_files)} files")
    print(f"⚠️  Errors: {len(error_files)} files")
    
    if missing_files:
        print(f"\n🚨 MISSING FILES:")
        for filename in missing_files:
            print(f"   - {filename}")
    
    if error_files:
        print(f"\n⚠️  ERROR FILES:")
        for filename in error_files:
            error_msg = critical_files[filename].get('error', 'Unknown error')
            print(f"   - {filename}: {error_msg}")
    
    # Check data integrity
    integrity = results.get('data_integrity_check', {})
    if integrity:
        chunk_consistency = integrity.get('chunk_consistency', {})
        if 'error' not in chunk_consistency:
            chunks_count = chunk_consistency.get('chunks_count', 0)
            consistent = chunk_consistency.get('consistent', False)
            print(f"\n📋 Chunks: {chunks_count} {'✅' if consistent else '❌'}")
        
        qa_data = integrity.get('qa_data', {})
        if 'error' not in qa_data:
            qa_pairs = qa_data.get('total_pairs', 0)
            has_revenue = qa_data.get('has_revenue_data', False)
            print(f"📋 Q&A Pairs: {qa_pairs} {'✅' if has_revenue else '❌'} (Revenue: {has_revenue})")
    
    # Overall assessment
    print("\n" + "="*60)
    if not missing_files and not error_files:
        print("🎉 ALL SYSTEMS GO! Data loading should work correctly.")
    elif missing_files:
        print("🚨 CRITICAL ISSUE: Missing data files will cause system failure.")
    else:
        print("⚠️  PARTIAL ISSUE: Some data files have errors but system might work.")
    
    print("="*60)
    
    return len(missing_files) == 0 and len(error_files) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
