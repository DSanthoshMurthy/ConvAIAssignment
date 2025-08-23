import os
import pandas as pd
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
import logging
from typing import Dict, List, Union, Tuple
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinancialDataLoader:
    def __init__(self, raw_data_dir: str, processed_data_dir: str):
        """Initialize the Financial Data Loader.
        
        Args:
            raw_data_dir (str): Directory containing raw financial documents
            processed_data_dir (str): Directory to save processed data
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data: Dict[str, Union[str, pd.DataFrame]] = {}
        
    def process_excel(self, file_path: Path) -> pd.DataFrame:
        """Process Excel files and extract financial data.
        
        Args:
            file_path (Path): Path to the Excel file
            
        Returns:
            pd.DataFrame: Extracted data as a DataFrame
        """
        logger.info(f"Processing Excel file: {file_path}")
        
        try:
            # Read Excel file
            df = pd.read_excel(file_path)
            
            # Basic cleaning
            # Remove empty rows and columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing Excel {file_path}: {str(e)}")
            raise
    
    def process_pdf(self, file_path: Path) -> str:
        """Process PDF files and extract text using OCR if needed.
        
        Args:
            file_path (Path): Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        logger.info(f"Processing PDF file: {file_path}")
        
        try:
            # First try direct PDF text extraction
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                
            # If extracted text is too short or empty, use OCR
            if len(text.strip()) < 100:
                logger.info("Direct extraction yielded little text. Using OCR...")
                images = convert_from_path(file_path)
                text = ""
                for image in images:
                    text += pytesseract.image_to_string(image)
            
            return text
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text by removing headers, footers, and page numbers.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        # Remove page numbers
        text = text.replace('\f', '\n')  # Form feed to newline
        text = '\n'.join(line for line in text.split('\n') if not line.strip().isdigit())
        
        # Remove multiple spaces and newlines
        text = ' '.join(text.split())
        
        return text
    
    def identify_sections(self, text: str) -> Dict[str, str]:
        """Identify and extract different sections from the financial text.
        
        Args:
            text (str): Cleaned text to segment
            
        Returns:
            Dict[str, str]: Dictionary mapping section names to their content
        """
        sections = {}
        
        # Common section markers in financial statements
        markers = {
            'income_statement': ['income statement', 'profit and loss', 'statement of operations'],
            'balance_sheet': ['balance sheet', 'statement of financial position'],
            'cash_flow': ['cash flow statement', 'statement of cash flows'],
            'notes': ['notes to', 'notes to the financial statements']
        }
        
        # Convert text to lowercase for easier matching
        text_lower = text.lower()
        
        for section_name, keywords in markers.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Find the start of this section
                    start_idx = text_lower.find(keyword)
                    
                    # Find the start of the next section (if any)
                    next_starts = []
                    for other_keywords in markers.values():
                        for other_keyword in other_keywords:
                            if other_keyword != keyword:
                                idx = text_lower.find(other_keyword, start_idx + len(keyword))
                                if idx != -1:
                                    next_starts.append(idx)
                    
                    # Get the end index
                    end_idx = min(next_starts) if next_starts else len(text)
                    
                    # Extract the section content
                    sections[section_name] = text[start_idx:end_idx].strip()
                    break
        
        return sections
    
    def process_all_documents(self) -> None:
        """Process all financial documents in the raw data directory."""
        # Process Excel files
        excel_files = list(self.raw_data_dir.glob('*.xlsx'))
        for file_path in excel_files:
            try:
                df = self.process_excel(file_path)
                self.processed_data[file_path.name] = df
                
                # Save processed DataFrame
                output_path = self.processed_data_dir / f"{file_path.stem}.csv"
                df.to_csv(output_path, index=False)
                logger.info(f"Saved processed Excel data to {output_path}")
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                continue
        
        # Process PDF files
        pdf_files = list(self.raw_data_dir.glob('*.pdf'))
        for file_path in pdf_files:
            try:
                # Extract text
                text = self.process_pdf(file_path)
                
                # Clean text
                cleaned_text = self.clean_text(text)
                
                # Identify sections
                sections = self.identify_sections(cleaned_text)
                
                self.processed_data[file_path.name] = {
                    'full_text': cleaned_text,
                    'sections': sections
                }
                
                # Save processed text
                output_base = self.processed_data_dir / file_path.stem
                
                # Save full text
                with open(output_base.with_suffix('.txt'), 'w') as f:
                    f.write(cleaned_text)
                
                # Save sections
                for section_name, content in sections.items():
                    section_path = output_base.with_name(f"{file_path.stem}_{section_name}.txt")
                    with open(section_path, 'w') as f:
                        f.write(content)
                
                logger.info(f"Saved processed PDF data from {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                continue

if __name__ == "__main__":
    # Initialize loader
    loader = FinancialDataLoader(
        raw_data_dir="data/raw",
        processed_data_dir="data/processed"
    )
    
    # Process all documents
    loader.process_all_documents()

