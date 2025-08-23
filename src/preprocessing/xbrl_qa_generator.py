#!/usr/bin/env python3
"""
XBRL QA Generator
Generates Q&A pairs from processed XBRL financial data.
"""

import pandas as pd
import json
from pathlib import Path
import logging
from typing import List, Dict
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class XBRLQAGenerator:
    def __init__(self, processed_data_dir: str = "data/processed", output_dir: str = "data/processed"):
        """Initialize the XBRL QA Generator.
        
        Args:
            processed_data_dir (str): Directory containing processed financial CSV data
            output_dir (str): Directory to save generated Q/A pairs
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def format_currency(self, value: float) -> str:
        """Format currency values in appropriate units with 2 decimal places."""
        if pd.isna(value):
            return "Not Available"
        
        abs_value = abs(value)
        if abs_value >= 1e9:  # Billion
            formatted = f"₹{abs_value/1e9:.2f} billion"
        elif abs_value >= 1e7:  # Crores
            formatted = f"₹{abs_value/1e7:.2f} crores"
        elif abs_value >= 1e5:  # Lakhs
            formatted = f"₹{abs_value/1e5:.2f} lakhs"
        else:
            formatted = f"₹{abs_value:.2f}"
        
        return f"-{formatted}" if value < 0 else formatted
    
    def get_element_value(self, df: pd.DataFrame, element_name: str) -> float:
        """Safely get element value from dataframe."""
        try:
            matching_rows = df[df['Element Name'] == element_name]
            if not matching_rows.empty:
                fact_value = matching_rows['Fact Value'].iloc[0]
                return pd.to_numeric(fact_value, errors='coerce')
            return None
        except:
            return None
    
    def get_element_text(self, df: pd.DataFrame, element_name: str) -> str:
        """Safely get element text value from dataframe."""
        try:
            matching_rows = df[df['Element Name'] == element_name]
            if not matching_rows.empty:
                return str(matching_rows['Fact Value'].iloc[0])
            return None
        except:
            return None
    
    def generate_qa_pairs_for_quarter(self, csv_path: Path) -> List[Dict[str, str]]:
        """Generate Q/A pairs from a single quarter's CSV data.
        
        Args:
            csv_path (Path): Path to the CSV file
            
        Returns:
            List[Dict[str, str]]: List of Q/A pairs
        """
        logger.info(f"Processing {csv_path.name} for Q/A generation")
        
        try:
            df = pd.read_csv(csv_path)
            qa_pairs = []
            
            # Extract period information
            period = df['Period'].iloc[0] if not df.empty else "Unknown Period"
            quarter_name = csv_path.stem.replace('_', ' ').title()
            
            # Basic company information
            company_name = self.get_element_text(df, 'NameOfTheCompany')
            if company_name:
                qa_pairs.append({
                    "question": f"What is the name of the company for the {quarter_name} report?",
                    "answer": f"The company name is {company_name}",
                    "quarter": quarter_name,
                    "period": period
                })
            
            # Financial year information
            fy_start = self.get_element_text(df, 'DateOfStartOfFinancialYear')
            fy_end = self.get_element_text(df, 'DateOfEndOfFinancialYear')
            if fy_start and fy_end:
                qa_pairs.append({
                    "question": f"What is the financial year period for the {quarter_name} report?",
                    "answer": f"The financial year period is from {fy_start} to {fy_end}",
                    "quarter": quarter_name,
                    "period": period
                })
            
            # Revenue metrics
            revenue = self.get_element_value(df, 'RevenueFromOperations')
            if revenue is not None:
                qa_pairs.append({
                    "question": f"What was the revenue from operations in {quarter_name}?",
                    "answer": f"The revenue from operations was {self.format_currency(revenue)}",
                    "quarter": quarter_name,
                    "period": period
                })
            
            other_income = self.get_element_value(df, 'OtherIncome')
            if other_income is not None:
                qa_pairs.append({
                    "question": f"What was the other income in {quarter_name}?",
                    "answer": f"The other income was {self.format_currency(other_income)}",
                    "quarter": quarter_name,
                    "period": period
                })
            
            total_income = self.get_element_value(df, 'Income')
            if total_income is not None:
                qa_pairs.append({
                    "question": f"What was the total income in {quarter_name}?",
                    "answer": f"The total income was {self.format_currency(total_income)}",
                    "quarter": quarter_name,
                    "period": period
                })
            
            # Expense metrics
            finance_costs = self.get_element_value(df, 'FinanceCosts')
            if finance_costs is not None:
                qa_pairs.append({
                    "question": f"What were the finance costs in {quarter_name}?",
                    "answer": f"The finance costs were {self.format_currency(finance_costs)}",
                    "quarter": quarter_name,
                    "period": period
                })
            
            employee_benefits = self.get_element_value(df, 'EmployeeBenefitExpense')
            if employee_benefits is not None:
                qa_pairs.append({
                    "question": f"What was the employee benefit expense in {quarter_name}?",
                    "answer": f"The employee benefit expense was {self.format_currency(employee_benefits)}",
                    "quarter": quarter_name,
                    "period": period
                })
            
            other_expenses = self.get_element_value(df, 'OtherExpenses')
            if other_expenses is not None:
                qa_pairs.append({
                    "question": f"What were the other expenses in {quarter_name}?",
                    "answer": f"The other expenses were {self.format_currency(other_expenses)}",
                    "quarter": quarter_name,
                    "period": period
                })
            
            # Profit/Loss metrics
            profit_before_tax = self.get_element_value(df, 'ProfitBeforeTax')
            if profit_before_tax is not None:
                qa_pairs.append({
                    "question": f"What was the profit before tax in {quarter_name}?",
                    "answer": f"The {'loss' if profit_before_tax < 0 else 'profit'} before tax was {self.format_currency(abs(profit_before_tax))}",
                    "quarter": quarter_name,
                    "period": period
                })
            
            net_profit = self.get_element_value(df, 'ProfitLossForPeriod')
            if net_profit is not None:
                qa_pairs.append({
                    "question": f"What was the net profit/loss for {quarter_name}?",
                    "answer": f"The company reported a {'loss' if net_profit < 0 else 'profit'} of {self.format_currency(abs(net_profit))}",
                    "quarter": quarter_name,
                    "period": period
                })
            
            # Tax metrics
            current_tax = self.get_element_value(df, 'CurrentTax')
            if current_tax is not None:
                qa_pairs.append({
                    "question": f"What was the current tax expense in {quarter_name}?",
                    "answer": f"The current tax expense was {self.format_currency(current_tax)}",
                    "quarter": quarter_name,
                    "period": period
                })
            
            # EPS metrics
            basic_eps = self.get_element_value(df, 'BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations')
            if basic_eps is not None:
                qa_pairs.append({
                    "question": f"What was the basic earnings per share (EPS) in {quarter_name}?",
                    "answer": f"The basic EPS was ₹{basic_eps:.2f} per share",
                    "quarter": quarter_name,
                    "period": period
                })
            
            # Depreciation
            depreciation = self.get_element_value(df, 'DepreciationDepletionAndAmortisationExpense')
            if depreciation is not None:
                qa_pairs.append({
                    "question": f"What was the depreciation expense in {quarter_name}?",
                    "answer": f"The depreciation and amortization expense was {self.format_currency(depreciation)}",
                    "quarter": quarter_name,
                    "period": period
                })
            
            # Comprehensive Income
            comprehensive_income = self.get_element_value(df, 'ComprehensiveIncomeForThePeriod')
            if comprehensive_income is not None:
                qa_pairs.append({
                    "question": f"What was the comprehensive income in {quarter_name}?",
                    "answer": f"The comprehensive income was {self.format_currency(comprehensive_income)}",
                    "quarter": quarter_name,
                    "period": period
                })
            
            # Segment information
            segment_revenue = self.get_element_value(df, 'SegmentRevenue')
            if segment_revenue is not None:
                qa_pairs.append({
                    "question": f"What was the segment revenue in {quarter_name}?",
                    "answer": f"The segment revenue was {self.format_currency(segment_revenue)}",
                    "quarter": quarter_name,
                    "period": period
                })
            
            # Generate analytical questions
            if revenue and finance_costs:
                finance_to_revenue_ratio = (finance_costs / revenue) * 100
                qa_pairs.append({
                    "question": f"What percentage of revenue was spent on finance costs in {quarter_name}?",
                    "answer": f"Finance costs represented {finance_to_revenue_ratio:.2f}% of revenue in {quarter_name}",
                    "quarter": quarter_name,
                    "period": period
                })
            
            # Performance analysis
            if profit_before_tax and profit_before_tax < 0:
                key_factors = []
                if finance_costs and finance_costs > 0:
                    key_factors.append(f"finance costs of {self.format_currency(finance_costs)}")
                if other_expenses and other_expenses > 0:
                    key_factors.append(f"other expenses of {self.format_currency(other_expenses)}")
                
                if key_factors:
                    qa_pairs.append({
                        "question": f"What were the main factors contributing to the loss in {quarter_name}?",
                        "answer": f"The main factors contributing to the loss included {' and '.join(key_factors)}",
                        "quarter": quarter_name,
                        "period": period
                    })
            
            logger.info(f"Generated {len(qa_pairs)} Q/A pairs for {quarter_name}")
            return qa_pairs
            
        except Exception as e:
            logger.error(f"Error generating Q/A pairs for {csv_path}: {str(e)}")
            return []
    
    def process_all_quarters(self) -> List[Dict[str, str]]:
        """Process all CSV files and generate Q/A pairs."""
        all_qa_pairs = []
        
        csv_files = list(self.processed_data_dir.glob('*.csv'))
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        for csv_path in csv_files:
            if csv_path.name != 'stament2.csv':  # Skip this file as it might be different format
                qa_pairs = self.generate_qa_pairs_for_quarter(csv_path)
                all_qa_pairs.extend(qa_pairs)
        
        logger.info(f"Generated total of {len(all_qa_pairs)} Q/A pairs from all quarters")
        return all_qa_pairs
    
    def save_qa_pairs(self, qa_pairs: List[Dict[str, str]], filename: str = "xbrl_qa_pairs.json") -> Path:
        """Save generated Q/A pairs to a JSON file.
        
        Args:
            qa_pairs (List[Dict[str, str]]): List of Q/A pairs to save
            filename (str): Output filename
            
        Returns:
            Path: Path to the saved file
        """
        output_file = self.output_dir / filename
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(qa_pairs)} Q/A pairs to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving Q/A pairs: {str(e)}")
            raise
    
    def generate_summary_report(self, qa_pairs: List[Dict[str, str]]) -> Dict:
        """Generate a summary report of the Q/A pairs."""
        quarters = list(set([pair.get('quarter', 'Unknown') for pair in qa_pairs]))
        
        summary = {
            'total_qa_pairs': len(qa_pairs),
            'quarters_covered': len(quarters),
            'quarter_list': sorted(quarters),
            'qa_by_quarter': {}
        }
        
        for quarter in quarters:
            quarter_pairs = [pair for pair in qa_pairs if pair.get('quarter') == quarter]
            summary['qa_by_quarter'][quarter] = len(quarter_pairs)
        
        return summary
    
    def run_complete_process(self):
        """Run the complete Q/A generation process."""
        logger.info("Starting XBRL Q/A generation process...")
        
        # Generate Q/A pairs
        all_qa_pairs = self.process_all_quarters()
        
        if not all_qa_pairs:
            logger.error("No Q/A pairs generated!")
            return False
        
        # Save Q/A pairs
        output_file = self.save_qa_pairs(all_qa_pairs)
        
        # Generate and save summary
        summary = self.generate_summary_report(all_qa_pairs)
        summary_file = self.output_dir / "qa_generation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("XBRL Q&A GENERATION SUMMARY")
        print("="*60)
        print(f"Total Q&A pairs generated: {summary['total_qa_pairs']}")
        print(f"Quarters covered: {summary['quarters_covered']}")
        print(f"Quarter list: {', '.join(summary['quarter_list'])}")
        
        print(f"\nQ&A pairs by quarter:")
        for quarter, count in summary['qa_by_quarter'].items():
            print(f"  {quarter}: {count} pairs")
        
        print(f"\n✅ Q&A pairs saved to: {output_file}")
        print(f"✅ Summary saved to: {summary_file}")
        
        return True

def main():
    """Main function to run the XBRL Q/A generation."""
    generator = XBRLQAGenerator()
    success = generator.run_complete_process()
    
    if success:
        print("\n🎉 XBRL Q/A generation completed successfully!")
    else:
        print("\n❌ XBRL Q/A generation failed. Please check the logs.")

if __name__ == "__main__":
    main()
