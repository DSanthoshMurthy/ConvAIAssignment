import json
import os
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer

class FinancialDatasetProcessor:
    def __init__(self, data_path: str):
        """Initialize the dataset processor.
        
        Args:
            data_path: Path to the xbrl_qa_pairs.json file
        """
        self.data_path = data_path
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.max_length = 512  # DistilBERT's maximum sequence length
        
        # Define expert categories and their keywords
        self.expert_categories = {
            'company_info': ['name', 'company', 'period', 'year', 'quarter'],
            'revenue_metrics': ['revenue', 'income', 'total income'],
            'expense_metrics': ['expense', 'cost', 'depreciation', 'finance costs', 'employee benefit'],
            'profitability': ['profit', 'loss', 'eps', 'earnings', 'comprehensive'],
            'segment_analysis': ['segment', 'percentage', 'ratio', 'factors']
        }
    
    def load_data(self) -> List[Dict]:
        """Load Q/A pairs from JSON file."""
        with open(self.data_path, 'r') as f:
            return json.load(f)
    
    def categorize_question(self, question: str) -> str:
        """Categorize question into one of the expert domains."""
        question = question.lower()
        
        # Count keyword matches for each category
        category_scores = {
            category: sum(1 for keyword in keywords if keyword in question)
            for category, keywords in self.expert_categories.items()
        }
        
        # Return category with highest score, default to 'income_statement' if no matches
        return max(category_scores.items(), key=lambda x: x[1])[0]
    
    def prepare_for_training(self, qa_pairs: List[Dict]) -> Tuple[Dict, Dict, Dict]:
        """Prepare data for DistilBERT fine-tuning."""
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(qa_pairs)
        
        # Add expert categories
        df['expert_category'] = df['question'].apply(self.categorize_question)
        
        # Create features
        features = []
        for _, row in df.iterrows():
            # Tokenize question and answer
            question_tokens = self.tokenizer(
                row['question'],
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            answer_tokens = self.tokenizer(
                row['answer'],
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            features.append({
                'input_ids': question_tokens['input_ids'].squeeze(),
                'attention_mask': question_tokens['attention_mask'].squeeze(),
                'labels': answer_tokens['input_ids'].squeeze(),
                'expert_category': row['expert_category']
            })
        
        # Split into train/val/test (70/15/15)
        train_features, temp_features = train_test_split(features, test_size=0.3, random_state=42)
        val_features, test_features = train_test_split(temp_features, test_size=0.5, random_state=42)
        
        return {
            'train': train_features,
            'validation': val_features,
            'test': test_features
        }
    
    def get_expert_distribution(self, features: List[Dict]) -> Dict[str, int]:
        """Get distribution of questions across expert categories."""
        distribution = {}
        for feature in features:
            category = feature['expert_category']
            distribution[category] = distribution.get(category, 0) + 1
        return distribution

def main():
    # Initialize processor
    processor = FinancialDatasetProcessor('data/processed/xbrl_qa_pairs.json')
    
    # Load and process data
    qa_pairs = processor.load_data()
    datasets = processor.prepare_for_training(qa_pairs)
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total Q/A pairs: {len(qa_pairs)}")
    print("\nExpert Distribution in Training Set:")
    train_dist = processor.get_expert_distribution(datasets['train'])
    for category, count in train_dist.items():
        print(f"{category}: {count} questions")

if __name__ == "__main__":
    main()
