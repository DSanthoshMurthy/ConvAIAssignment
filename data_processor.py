import os
import PyPDF2
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
from typing import List, Dict, Tuple
import re
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinancialDataProcessor:
    def __init__(self, data_dir: str):
        """Initialize the Financial Data Processor.
        
        Args:
            data_dir (str): Directory containing financial documents
        """
        self.data_dir = data_dir
        self.processed_text = {}
        self.sections = {}
        
    def process_pdf(self, file_path: str) -> str:
        """Process PDF files and extract text using OCR if needed.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        logger.info(f"Processing PDF file: {file_path}")
        
        try:
            # First try direct PDF text extraction
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
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
    
    def process_excel(self, file_path: str) -> str:
        """Process Excel files and extract relevant financial data.
        
        Args:
            file_path (str): Path to the Excel file
            
        Returns:
            str: Extracted text from the Excel file
        """
        logger.info(f"Processing Excel file: {file_path}")
        
        try:
            # Read Excel file
            df = pd.read_excel(file_path)
            
            # Convert DataFrame to string representation
            text = df.to_string()
            
            return text
            
        except Exception as e:
            logger.error(f"Error processing Excel {file_path}: {str(e)}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text by removing headers, footers, and page numbers.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        # Remove page numbers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove headers/footers (common patterns)
        text = re.sub(r'(?m)^\s*Page \d+ of \d+\s*$', '', text)
        text = re.sub(r'(?m)^\s*Confidential\s*$', '', text)
        
        # Remove multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()
    
    def segment_sections(self, text: str) -> Dict[str, str]:
        """Segment the financial report into logical sections.
        
        Args:
            text (str): Cleaned text to segment
            
        Returns:
            Dict[str, str]: Dictionary of section name to section content
        """
        sections = {}
        
        # Common section markers in financial statements
        section_markers = {
            'income_statement': r'(?i)(consolidated\s+)?income\s+statement',
            'balance_sheet': r'(?i)(consolidated\s+)?balance\s+sheet',
            'cash_flow': r'(?i)(consolidated\s+)?cash\s+flow\s+statement',
            'notes': r'(?i)notes\s+to\s+(the\s+)?financial\s+statements'
        }
        
        # Find sections
        current_section = 'other'
        lines = text.split('\n')
        current_content = []
        
        for line in lines:
            new_section_found = False
            for section_name, pattern in section_markers.items():
                if re.search(pattern, line, re.IGNORECASE):
                    # Save previous section
                    if current_content:
                        sections[current_section] = '\n'.join(current_content)
                    # Start new section
                    current_section = section_name
                    current_content = []
                    new_section_found = True
                    break
            
            if not new_section_found:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def process_all_documents(self) -> None:
        """Process all financial documents in the data directory."""
        for filename in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, filename)
            
            try:
                if filename.lower().endswith('.pdf'):
                    text = self.process_pdf(file_path)
                elif filename.lower().endswith(('.xlsx', '.xls')):
                    text = self.process_excel(file_path)
                else:
                    logger.warning(f"Unsupported file type: {filename}")
                    continue
                
                # Clean the extracted text
                cleaned_text = self.clean_text(text)
                self.processed_text[filename] = cleaned_text
                
                # Segment into sections
                self.sections[filename] = self.segment_sections(cleaned_text)
                
                logger.info(f"Successfully processed {filename}")
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                continue
    
    def generate_qa_pairs(self) -> List[Dict[str, str]]:
        """Generate Q&A pairs from the processed financial data.
        
        Returns:
            List[Dict[str, str]]: List of Q&A pairs
        """
        qa_pairs = []
        
        # Template questions for different sections
        templates = {
            'income_statement': [
                "What was the total revenue in {year}?",
                "What was the operating income in {year}?",
                "What was the net income in {year}?",
                "What were the total expenses in {year}?"
            ],
            'balance_sheet': [
                "What were the total assets in {year}?",
                "What were the total liabilities in {year}?",
                "What was the shareholders' equity in {year}?",
                "What was the current ratio in {year}?"
            ],
            'cash_flow': [
                "What was the operating cash flow in {year}?",
                "What was the investing cash flow in {year}?",
                "What was the financing cash flow in {year}?",
                "What was the net change in cash in {year}?"
            ]
        }
        
        # TODO: Implement logic to extract answers and generate Q&A pairs
        # This will require more sophisticated NLP and pattern matching
        # based on the actual structure of your financial documents
        
        return qa_pairs

if __name__ == "__main__":
    # Initialize processor
    processor = FinancialDataProcessor(".")
    
    # Process all documents
    processor.process_all_documents()
    
    # Generate Q&A pairs
    qa_pairs = processor.generate_qa_pairs()
    
    # Save processed data
    # TODO: Implement data saving logic

