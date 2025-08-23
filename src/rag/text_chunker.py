#!/usr/bin/env python3
"""
Text Chunker for Financial RAG System
Creates 100-token chunks with metadata from financial data.
"""

import pandas as pd
import json
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
from transformers import AutoTokenizer
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialTextChunker:
    def __init__(self, 
                 processed_data_dir: str = "data/processed",
                 chunks_output_dir: str = "data/chunks",
                 chunk_size: int = 100,
                 overlap_ratio: float = 0.2):
        """Initialize the Financial Text Chunker.
        
        Args:
            processed_data_dir: Directory with processed CSV files
            chunks_output_dir: Directory to save chunks
            chunk_size: Target tokens per chunk
            overlap_ratio: Overlap between chunks (0.2 = 20%)
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.chunks_output_dir = Path(chunks_output_dir)
        self.chunk_size = chunk_size
        self.overlap_tokens = int(chunk_size * overlap_ratio)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        # Create output directory
        self.chunks_output_dir.mkdir(exist_ok=True)
        
        # Storage for all chunks
        self.all_chunks = []
        self.chunk_counter = 0
    
    def extract_quarter_info(self, filename: str) -> Dict[str, str]:
        """Extract quarter and year information from filename."""
        # Map filenames to readable quarters
        quarter_mapping = {
            'jun_2022': 'Q1 FY2022-23',
            'sep_2022': 'Q2 FY2022-23', 
            'dec_2022': 'Q3 FY2022-23',
            'march_2023': 'Q4 FY2022-23',
            'jun_2023': 'Q1 FY2023-24',
            'sep_2023': 'Q2 FY2023-24',
            'dec_2023': 'Q3 FY2023-24',
            'marc_2024': 'Q4 FY2023-24'
        }
        
        filename_base = filename.replace('.csv', '')
        return {
            'quarter': quarter_mapping.get(filename_base, filename_base.title()),
            'fiscal_year': 'FY2023-24' if '2023' in filename or '2024' in filename else 'FY2022-23',
            'calendar_year': '2024' if '2024' in filename else '2023' if '2023' in filename else '2022'
        }
    
    def categorize_financial_element(self, element_name: str) -> Dict[str, Any]:
        """Categorize financial elements into sections and types."""
        element_lower = element_name.lower()
        
        # Define category mappings
        categories = {
            'revenue': ['revenue', 'sales', 'income'],
            'expenses': ['expense', 'cost', 'depreciation', 'finance'],
            'profit_loss': ['profit', 'loss', 'earnings', 'ebitda'],
            'assets': ['asset', 'investment', 'property', 'cash'],
            'liabilities': ['liability', 'borrowing', 'debt', 'payable'],
            'equity': ['equity', 'capital', 'share'],
            'ratios': ['ratio', 'per', 'percentage'],
            'cash_flow': ['cash flow', 'operating activities', 'investing activities', 'financing activities']
        }
        
        section_type = 'other'
        for category, keywords in categories.items():
            if any(keyword in element_lower for keyword in keywords):
                section_type = category
                break
        
        # Extract financial terms
        financial_terms = []
        for keyword in ['revenue', 'profit', 'loss', 'asset', 'liability', 'cash', 'expense', 'income']:
            if keyword in element_lower:
                financial_terms.append(keyword)
        
        return {
            'section_type': section_type,
            'financial_terms': financial_terms,
            'is_financial_metric': len(financial_terms) > 0
        }
    
    def create_text_content(self, row: pd.Series, quarter_info: Dict[str, str]) -> str:
        """Create meaningful text content from a financial data row."""
        element_name = row['Element Name']
        fact_value = row['Fact Value']
        period = row['Period']
        unit = row.get('Unit', '')
        
        # Skip basic company info
        skip_elements = ['ScripCode', 'Symbol', 'MSEISymbol', 'NameOfTheCompany', 'ClassOfSecurity']
        if element_name in skip_elements:
            return ""
        
        # Create readable text
        if pd.isna(fact_value) or fact_value == "":
            return f"In {quarter_info['quarter']} ({period}), {element_name} data was not available."
        
        # Format numeric values
        try:
            numeric_value = float(fact_value)
            if abs(numeric_value) >= 1e9:
                formatted_value = f"₹{numeric_value/1e9:.2f} billion"
            elif abs(numeric_value) >= 1e7:
                formatted_value = f"₹{numeric_value/1e7:.2f} crores"
            elif abs(numeric_value) >= 1e5:
                formatted_value = f"₹{numeric_value/1e5:.2f} lakhs"
            else:
                formatted_value = f"₹{numeric_value:.2f}"
                
            return f"In {quarter_info['quarter']} ({period}), {element_name} was {formatted_value}. "
            
        except (ValueError, TypeError):
            # Non-numeric values
            return f"In {quarter_info['quarter']} ({period}), {element_name} was {fact_value}. "
    
    def tokenize_and_chunk(self, text: str) -> List[Dict[str, Any]]:
        """Split text into chunks based on token count."""
        if not text.strip():
            return []
        
        # Tokenize the text
        tokens = self.tokenizer.encode(text)
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            # Define chunk boundaries
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            
            # Extract chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            # Only keep chunks with meaningful content
            if len(chunk_text.strip()) > 20:  # Minimum meaningful length
                chunks.append({
                    'text': chunk_text.strip(),
                    'tokens': len(chunk_tokens),
                    'start_token': start_idx,
                    'end_token': end_idx
                })
            
            # Move to next chunk with overlap
            start_idx = end_idx - self.overlap_tokens
            
            # Break if we're not making progress
            if start_idx >= end_idx - self.overlap_tokens:
                break
        
        return chunks
    
    def process_csv_file(self, csv_path: Path) -> List[Dict[str, Any]]:
        """Process a single CSV file into chunks."""
        logger.info(f"Processing {csv_path.name}...")
        
        try:
            df = pd.read_csv(csv_path)
            quarter_info = self.extract_quarter_info(csv_path.name)
            
            # Group related financial elements
            financial_texts = []
            current_section = ""
            current_text = ""
            
            for _, row in df.iterrows():
                element_categorization = self.categorize_financial_element(row['Element Name'])
                text_content = self.create_text_content(row, quarter_info)
                
                if text_content:
                    # Group by section type
                    if element_categorization['section_type'] != current_section:
                        if current_text:
                            financial_texts.append({
                                'text': current_text.strip(),
                                'section': current_section,
                                'quarter_info': quarter_info
                            })
                        current_section = element_categorization['section_type']
                        current_text = text_content
                    else:
                        current_text += text_content
            
            # Add the last section
            if current_text:
                financial_texts.append({
                    'text': current_text.strip(),
                    'section': current_section,
                    'quarter_info': quarter_info
                })
            
            # Create chunks from grouped texts
            file_chunks = []
            for text_group in financial_texts:
                text_chunks = self.tokenize_and_chunk(text_group['text'])
                
                for i, chunk in enumerate(text_chunks):
                    self.chunk_counter += 1
                    
                    chunk_data = {
                        'chunk_id': f"{quarter_info['quarter'].replace(' ', '_')}_{text_group['section']}_{self.chunk_counter:04d}",
                        'text': chunk['text'],
                        'tokens': chunk['tokens'],
                        'quarter': quarter_info['quarter'],
                        'fiscal_year': quarter_info['fiscal_year'],
                        'calendar_year': quarter_info['calendar_year'],
                        'section': text_group['section'],
                        'source_file': csv_path.name,
                        'chunk_index': i,
                        'metadata': {
                            'company': 'Jaiprakash Associates Limited',
                            'total_chunks_in_section': len(text_chunks),
                            'processing_timestamp': datetime.now().isoformat()
                        }
                    }
                    
                    file_chunks.append(chunk_data)
            
            logger.info(f"✓ Created {len(file_chunks)} chunks from {csv_path.name}")
            return file_chunks
            
        except Exception as e:
            logger.error(f"Error processing {csv_path.name}: {str(e)}")
            return []
    
    def process_qa_pairs(self, qa_path: Path) -> List[Dict[str, Any]]:
        """Process Q&A pairs into chunks for memory-augmented retrieval."""
        logger.info("Processing Q&A pairs for memory augmentation...")
        
        try:
            with open(qa_path, 'r', encoding='utf-8') as f:
                qa_pairs = json.load(f)
            
            qa_chunks = []
            for i, qa_pair in enumerate(qa_pairs):
                self.chunk_counter += 1
                
                # Create a chunk combining question and answer
                combined_text = f"Question: {qa_pair['question']} Answer: {qa_pair['answer']}"
                tokens = len(self.tokenizer.encode(combined_text))
                
                chunk_data = {
                    'chunk_id': f"QA_PAIR_{self.chunk_counter:04d}",
                    'text': combined_text,
                    'tokens': tokens,
                    'quarter': qa_pair.get('quarter', 'All Quarters'),
                    'fiscal_year': 'FY2022-24',
                    'calendar_year': 'Multi-Year',
                    'section': 'qa_memory',
                    'source_file': qa_path.name,
                    'chunk_index': i,
                    'metadata': {
                        'company': 'Jaiprakash Associates Limited',
                        'is_qa_pair': True,
                        'original_question': qa_pair['question'],
                        'original_answer': qa_pair['answer'],
                        'processing_timestamp': datetime.now().isoformat()
                    }
                }
                
                qa_chunks.append(chunk_data)
            
            logger.info(f"✓ Created {len(qa_chunks)} Q&A memory chunks")
            return qa_chunks
            
        except Exception as e:
            logger.error(f"Error processing Q&A pairs: {str(e)}")
            return []
    
    def process_all_data(self) -> Dict[str, int]:
        """Process all available data sources into chunks."""
        logger.info("Starting comprehensive data processing...")
        
        # Process CSV files
        csv_files = list(self.processed_data_dir.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        for csv_file in csv_files:
            if csv_file.name != 'stament2.csv':  # Skip potentially different format
                chunks = self.process_csv_file(csv_file)
                self.all_chunks.extend(chunks)
        
        # Process Q&A pairs
        qa_file = self.processed_data_dir / "xbrl_qa_pairs.json"
        if qa_file.exists():
            qa_chunks = self.process_qa_pairs(qa_file)
            self.all_chunks.extend(qa_chunks)
        
        # Generate statistics
        stats = self.generate_chunk_statistics()
        
        # Save all chunks
        self.save_chunks()
        
        logger.info(f"✅ Processing complete: {len(self.all_chunks)} total chunks created")
        return stats
    
    def generate_chunk_statistics(self) -> Dict[str, int]:
        """Generate comprehensive statistics about the created chunks."""
        if not self.all_chunks:
            return {}
        
        stats = {
            'total_chunks': len(self.all_chunks),
            'avg_tokens_per_chunk': sum(chunk['tokens'] for chunk in self.all_chunks) / len(self.all_chunks),
            'chunks_by_section': {},
            'chunks_by_quarter': {},
            'chunks_by_fiscal_year': {}
        }
        
        for chunk in self.all_chunks:
            # Count by section
            section = chunk['section']
            stats['chunks_by_section'][section] = stats['chunks_by_section'].get(section, 0) + 1
            
            # Count by quarter
            quarter = chunk['quarter']
            stats['chunks_by_quarter'][quarter] = stats['chunks_by_quarter'].get(quarter, 0) + 1
            
            # Count by fiscal year
            fy = chunk['fiscal_year']
            stats['chunks_by_fiscal_year'][fy] = stats['chunks_by_fiscal_year'].get(fy, 0) + 1
        
        return stats
    
    def save_chunks(self):
        """Save all chunks to JSON file."""
        output_file = self.chunks_output_dir / "financial_chunks_100_tokens.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_chunks, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Saved {len(self.all_chunks)} chunks to {output_file}")
        
        # Save statistics
        stats = self.generate_chunk_statistics()
        stats_file = self.chunks_output_dir / "chunk_statistics.json"
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Saved statistics to {stats_file}")
    
    def get_sample_chunks(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get sample chunks for inspection."""
        if not self.all_chunks:
            return []
        
        return self.all_chunks[:n]

def main():
    """Main function to run the chunking process."""
    chunker = FinancialTextChunker()
    
    # Process all data
    stats = chunker.process_all_data()
    
    # Display results
    print("\n" + "="*60)
    print("FINANCIAL TEXT CHUNKING RESULTS")
    print("="*60)
    print(f"Total chunks created: {stats.get('total_chunks', 0)}")
    print(f"Average tokens per chunk: {stats.get('avg_tokens_per_chunk', 0):.1f}")
    
    if 'chunks_by_section' in stats:
        print(f"\nChunks by section:")
        for section, count in stats['chunks_by_section'].items():
            print(f"  {section}: {count}")
    
    if 'chunks_by_quarter' in stats:
        print(f"\nChunks by quarter:")
        for quarter, count in stats['chunks_by_quarter'].items():
            print(f"  {quarter}: {count}")
    
    # Show sample chunks
    print(f"\n📝 Sample chunks:")
    samples = chunker.get_sample_chunks(3)
    for i, chunk in enumerate(samples, 1):
        print(f"\n{i}. Chunk ID: {chunk['chunk_id']}")
        print(f"   Section: {chunk['section']}")
        print(f"   Quarter: {chunk['quarter']}")
        print(f"   Tokens: {chunk['tokens']}")
        print(f"   Text: {chunk['text'][:100]}...")

if __name__ == "__main__":
    main()
