#!/usr/bin/env python3
"""
Financial Report Combiner
Combines quarterly financial reports into a comprehensive 2-year analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialReportCombiner:
    def __init__(self, data_directory='.'):
        """Initialize the Financial Report Combiner."""
        self.data_directory = Path(data_directory)
        self.quarterly_files = [
            'jun_2022.xlsx',    # Q1 FY2022-23
            'sep_2022.xlsx',    # Q2 FY2022-23  
            'dec_2022.xlsx',    # Q3 FY2022-23
            'march_2023.xlsx',  # Q4 FY2022-23
            'jun_2023.xlsx',    # Q1 FY2023-24
            'sep_2023.xlsx',    # Q2 FY2023-24
            'dec_2023.xlsx',    # Q3 FY2023-24
            'marc_2024.xlsx'    # Q4 FY2023-24
        ]
        self.combined_data = None
        self.key_metrics = {}
        
    def load_quarterly_data(self):
        """Load and combine all quarterly financial data."""
        logger.info("Loading quarterly financial data...")
        all_data = []
        
        for file_name in self.quarterly_files:
            file_path = self.data_directory / file_name
            
            if file_path.exists():
                logger.info(f"Processing {file_name}...")
                df = pd.read_excel(file_path)
                
                # Add metadata
                df['Quarter'] = file_name.replace('.xlsx', '')
                df['Source_File'] = file_name
                
                # Extract quarter info from period
                period = df['Period'].iloc[0] if not df.empty else ''
                df['Period_Start'] = period.split(' To ')[0] if ' To ' in period else ''
                df['Period_End'] = period.split(' To ')[1] if ' To ' in period else ''
                
                all_data.append(df)
                logger.info(f"✓ Loaded {len(df)} elements from {file_name}")
            else:
                logger.warning(f"File not found: {file_name}")
        
        if all_data:
            self.combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"✓ Combined data: {len(self.combined_data)} total records")
            return True
        
        logger.error("No data files found!")
        return False
    
    def extract_key_metrics(self):
        """Extract key financial metrics for analysis."""
        logger.info("Extracting key financial metrics...")
        
        # Define key financial metrics to track
        key_elements = {
            'Revenue': ['RevenueFromOperations', 'Revenue'],
            'Other Income': ['OtherIncome'],
            'Total Income': ['Income'],
            'Operating Expenses': ['EmployeeBenefitExpense', 'OtherExpenses'],
            'Finance Costs': ['FinanceCosts'],
            'Depreciation': ['DepreciationDepletionAndAmortisationExpense'],
            'Profit Before Tax': ['ProfitBeforeTax'],
            'Tax Expense': ['CurrentTax', 'TaxExpense'],
            'Net Profit': ['ProfitAfterTax', 'NetProfit'],
            'Total Assets': ['Assets'],
            'Total Liabilities': ['Liabilities'],
            'Equity': ['Equity'],
            'Cash Flow from Operations': ['CashFlowFromOperatingActivities'],
            'EPS': ['BasicEarningsPerShareInRupees', 'EarningsPerShare']
        }
        
        metrics_data = []
        
        for quarter in self.quarterly_files:
            quarter_name = quarter.replace('.xlsx', '')
            quarter_data = self.combined_data[self.combined_data['Quarter'] == quarter_name]
            
            if quarter_data.empty:
                continue
            
            quarter_metrics = {'Quarter': quarter_name}
            
            # Extract period info
            if not quarter_data.empty:
                period = quarter_data['Period'].iloc[0]
                quarter_metrics['Period'] = period
                
                # Extract year and quarter number
                if 'To' in period:
                    end_date = period.split('To')[1].strip()
                    if '2022' in end_date:
                        quarter_metrics['Financial_Year'] = 'FY2022-23'
                    elif '2023' in end_date:
                        quarter_metrics['Financial_Year'] = 'FY2023-24'  
                    elif '2024' in end_date:
                        quarter_metrics['Financial_Year'] = 'FY2023-24'
            
            # Extract financial values
            for metric_name, element_names in key_elements.items():
                value = None
                for element_name in element_names:
                    matching_rows = quarter_data[
                        quarter_data['Element Name'].str.contains(element_name, case=False, na=False)
                    ]
                    if not matching_rows.empty:
                        # Get the first non-null numeric value
                        for _, row in matching_rows.iterrows():
                            try:
                                val = pd.to_numeric(row['Fact Value'], errors='coerce')
                                if pd.notna(val):
                                    value = val
                                    break
                            except:
                                continue
                        if value is not None:
                            break
                
                quarter_metrics[metric_name] = value
            
            metrics_data.append(quarter_metrics)
        
        self.key_metrics = pd.DataFrame(metrics_data)
        logger.info(f"✓ Extracted metrics for {len(self.key_metrics)} quarters")
        return self.key_metrics
    
    def generate_financial_summary(self):
        """Generate a comprehensive financial summary."""
        if self.key_metrics is None or self.key_metrics.empty:
            logger.error("No key metrics available. Run extract_key_metrics() first.")
            return None
        
        logger.info("Generating financial summary...")
        
        summary = {
            'company_name': 'Jaiprakash Associates Limited',
            'reporting_period': 'April 2022 to March 2024 (2 Years)',
            'total_quarters': len(self.key_metrics),
            'financial_years': self.key_metrics['Financial_Year'].unique().tolist()
        }
        
        # Calculate yearly aggregates
        yearly_summary = self.key_metrics.groupby('Financial_Year').agg({
            'Revenue': 'sum',
            'Total Income': 'sum', 
            'Profit Before Tax': 'sum',
            'Net Profit': 'sum',
            'Finance Costs': 'sum'
        }).round(0)
        
        summary['yearly_performance'] = yearly_summary.to_dict()
        
        # Growth analysis
        if len(yearly_summary) >= 2:
            fy_2022_23 = yearly_summary.loc['FY2022-23'] if 'FY2022-23' in yearly_summary.index else None
            fy_2023_24 = yearly_summary.loc['FY2023-24'] if 'FY2023-24' in yearly_summary.index else None
            
            if fy_2022_23 is not None and fy_2023_24 is not None:
                growth_analysis = {}
                for metric in ['Revenue', 'Total Income', 'Profit Before Tax', 'Net Profit']:
                    if pd.notna(fy_2022_23[metric]) and pd.notna(fy_2023_24[metric]) and fy_2022_23[metric] != 0:
                        growth_rate = ((fy_2023_24[metric] - fy_2022_23[metric]) / abs(fy_2022_23[metric])) * 100
                        growth_analysis[metric] = round(growth_rate, 2)
                
                summary['year_over_year_growth'] = growth_analysis
        
        return summary
    
    def create_trend_visualizations(self, output_dir='reports'):
        """Create trend visualizations for key metrics."""
        if self.key_metrics is None or self.key_metrics.empty:
            logger.error("No key metrics available for visualization.")
            return False
        
        logger.info("Creating trend visualizations...")
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Jaiprakash Associates Limited - 2-Year Financial Trends', fontsize=16, fontweight='bold')
        
        # Revenue Trend
        ax1 = axes[0, 0]
        revenue_data = self.key_metrics.dropna(subset=['Revenue'])
        if not revenue_data.empty:
            ax1.plot(range(len(revenue_data)), revenue_data['Revenue']/1e9, marker='o', linewidth=2)
            ax1.set_title('Quarterly Revenue Trend')
            ax1.set_ylabel('Revenue (₹ Billions)')
            ax1.set_xlabel('Quarter')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
        
        # Profit Trend
        ax2 = axes[0, 1]
        profit_data = self.key_metrics.dropna(subset=['Net Profit'])
        if not profit_data.empty:
            ax2.plot(range(len(profit_data)), profit_data['Net Profit']/1e9, marker='s', color='green', linewidth=2)
            ax2.set_title('Quarterly Net Profit Trend')
            ax2.set_ylabel('Net Profit (₹ Billions)')
            ax2.set_xlabel('Quarter')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
        
        # Income vs Expenses
        ax3 = axes[1, 0]
        income_data = self.key_metrics.dropna(subset=['Total Income'])
        if not income_data.empty:
            ax3.bar(range(len(income_data)), income_data['Total Income']/1e9, alpha=0.7, label='Total Income', color='blue')
            # Note: We'd need total expenses data for a complete comparison
            ax3.set_title('Quarterly Income Trend')
            ax3.set_ylabel('Amount (₹ Billions)')
            ax3.set_xlabel('Quarter')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Finance Costs Trend
        ax4 = axes[1, 1]
        finance_data = self.key_metrics.dropna(subset=['Finance Costs'])
        if not finance_data.empty:
            ax4.plot(range(len(finance_data)), finance_data['Finance Costs']/1e9, marker='d', color='red', linewidth=2)
            ax4.set_title('Quarterly Finance Costs Trend')
            ax4.set_ylabel('Finance Costs (₹ Billions)')
            ax4.set_xlabel('Quarter')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_path = output_path / 'financial_trends.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved trend charts to {chart_path}")
        plt.close()
        
        return True
    
    def export_combined_report(self, output_dir='reports'):
        """Export the combined financial report to Excel."""
        logger.info("Exporting combined financial report...")
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = output_path / f'Jaiprakash_Associates_2Year_Report_{timestamp}.xlsx'
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Raw combined data
            if self.combined_data is not None:
                self.combined_data.to_excel(writer, sheet_name='Raw_Data', index=False)
            
            # Key metrics summary
            if self.key_metrics is not None:
                self.key_metrics.to_excel(writer, sheet_name='Key_Metrics', index=False)
            
            # Financial summary
            summary = self.generate_financial_summary()
            if summary:
                summary_df = pd.DataFrame([summary])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"✓ Exported combined report to {excel_path}")
        return excel_path
    
    def run_complete_analysis(self):
        """Run the complete financial analysis workflow."""
        logger.info("Starting complete financial analysis...")
        
        # Load data
        if not self.load_quarterly_data():
            return False
        
        # Extract metrics
        self.extract_key_metrics()
        
        # Generate summary
        summary = self.generate_financial_summary()
        
        # Create visualizations
        self.create_trend_visualizations()
        
        # Export report
        report_path = self.export_combined_report()
        
        # Print summary
        print("\n" + "="*60)
        print("JAIPRAKASH ASSOCIATES LIMITED - 2-YEAR FINANCIAL ANALYSIS")
        print("="*60)
        
        if summary:
            print(f"Company: {summary['company_name']}")
            print(f"Period: {summary['reporting_period']}")
            print(f"Quarters Analyzed: {summary['total_quarters']}")
            print(f"Financial Years: {', '.join(summary['financial_years'])}")
            
            if 'yearly_performance' in summary:
                print("\n📊 YEARLY PERFORMANCE:")
                for fy, metrics in summary['yearly_performance'].items():
                    print(f"\n{fy}:")
                    for metric, value in metrics.items():
                        if pd.notna(value):
                            print(f"  {metric}: ₹{value/1e9:.2f} Billion" if abs(value) >= 1e9 else f"  {metric}: ₹{value/1e6:.2f} Million")
            
            if 'year_over_year_growth' in summary:
                print("\n📈 YEAR-OVER-YEAR GROWTH:")
                for metric, growth in summary['year_over_year_growth'].items():
                    print(f"  {metric}: {growth:+.2f}%")
        
        print(f"\n✅ Complete report exported to: {report_path}")
        print("✅ Trend charts saved to: reports/financial_trends.png")
        
        return True

def main():
    """Main function to run the financial analysis."""
    combiner = FinancialReportCombiner()
    success = combiner.run_complete_analysis()
    
    if success:
        print("\n🎉 Financial analysis completed successfully!")
    else:
        print("\n❌ Financial analysis failed. Please check the logs.")

if __name__ == "__main__":
    main()
