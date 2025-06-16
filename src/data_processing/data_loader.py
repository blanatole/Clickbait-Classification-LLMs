"""
Webis-Clickbait-17 Dataset Loader
================================

Handles loading and extraction of the Webis-Clickbait-17 dataset including:
- Loading instances and truth labels from JSONL files
- Extracting text features while removing media components
- Data validation and quality checks
- Memory-efficient processing for large datasets
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from tqdm import tqdm
import yaml

from ..utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class WebisClickbaitLoader:
    """
    Loader for Webis-Clickbait-17 dataset with text feature extraction
    and media feature removal as recommended in the README.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            config_path: Path to data configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.text_fields = self.config['processing']['text_fields']
        self.primary_text_field = self.config['processing']['primary_text_field']
        
        # Statistics tracking
        self.stats = {
            'total_instances': 0,
            'total_truth_labels': 0,
            'missing_text_fields': 0,
            'empty_text_instances': 0,
            'label_distribution': {}
        }
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = "configs/data_config.yaml"
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default settings.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default configuration if config file is not found."""
        return {
            'processing': {
                'text_fields': [
                    "postText", "targetTitle", "targetDescription", 
                    "targetParagraphs", "targetKeywords"
                ],
                'primary_text_field': "postText"
            },
            'validation': {
                'check_missing_values': True,
                'check_text_length': True,
                'min_text_length': 5,
                'max_text_length': 1000,
                'check_duplicate_ids': True,
                'check_label_distribution': True
            }
        }
    
    def load_instances(self, instances_path: str) -> List[Dict]:
        """
        Load instances from JSONL file.
        
        Args:
            instances_path: Path to instances.jsonl file
            
        Returns:
            List of instance dictionaries
        """
        logger.info(f"Loading instances from {instances_path}")
        instances = []
        
        try:
            with open(instances_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(tqdm(f, desc="Loading instances")):
                    try:
                        instance = json.loads(line.strip())
                        instances.append(instance)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num + 1}: {e}")
                        continue
                        
        except FileNotFoundError:
            logger.error(f"Instances file not found: {instances_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading instances: {e}")
            raise
            
        self.stats['total_instances'] = len(instances)
        logger.info(f"Loaded {len(instances)} instances")
        return instances
    
    def load_truth_labels(self, truth_path: str) -> List[Dict]:
        """
        Load truth labels from JSONL file.
        
        Args:
            truth_path: Path to truth.jsonl file
            
        Returns:
            List of truth label dictionaries
        """
        logger.info(f"Loading truth labels from {truth_path}")
        truth_labels = []
        
        try:
            with open(truth_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(tqdm(f, desc="Loading truth labels")):
                    try:
                        truth = json.loads(line.strip())
                        truth_labels.append(truth)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num + 1}: {e}")
                        continue
                        
        except FileNotFoundError:
            logger.error(f"Truth labels file not found: {truth_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading truth labels: {e}")
            raise
            
        self.stats['total_truth_labels'] = len(truth_labels)
        logger.info(f"Loaded {len(truth_labels)} truth labels")
        return truth_labels
    
    def extract_text_features(self, instances: List[Dict]) -> pd.DataFrame:
        """
        Extract text features from instances, removing media components.
        
        Args:
            instances: List of instance dictionaries
            
        Returns:
            DataFrame with extracted text features
        """
        logger.info("Extracting text features and removing media components")
        
        extracted_data = []
        missing_text_count = 0
        empty_text_count = 0
        
        for instance in tqdm(instances, desc="Extracting text features"):
            # Extract ID
            item_id = instance.get('id', '')
            
            # Extract individual text components
            extracted_item = {'id': item_id}
            
            for field in self.text_fields:
                content = instance.get(field, '')
                
                # Handle list-type fields (like targetParagraphs)
                if isinstance(content, list):
                    content = ' '.join(str(item) for item in content if item)
                elif not isinstance(content, str):
                    content = str(content) if content else ''
                
                extracted_item[field] = content
                
                # Track missing fields
                if not content and field == self.primary_text_field:
                    missing_text_count += 1
            
            # Create combined text field
            combined_texts = []
            for field in self.text_fields:
                if extracted_item.get(field):
                    combined_texts.append(extracted_item[field])
            
            extracted_item['combined_text'] = ' '.join(combined_texts)
            
            # Check for empty primary text
            if not extracted_item.get(self.primary_text_field, '').strip():
                empty_text_count += 1
            
            # Add metadata
            extracted_item['timestamp'] = instance.get('postTimestamp', '')
            
            extracted_data.append(extracted_item)
        
        # Update statistics
        self.stats['missing_text_fields'] = missing_text_count
        self.stats['empty_text_instances'] = empty_text_count
        
        logger.info(f"Extracted text features from {len(extracted_data)} instances")
        logger.info(f"Missing primary text fields: {missing_text_count}")
        logger.info(f"Empty primary text instances: {empty_text_count}")
        
        return pd.DataFrame(extracted_data)
    
    def align_with_truth_labels(self, 
                               features_df: pd.DataFrame, 
                               truth_labels: List[Dict]) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Align features with truth labels by ID.
        
        Args:
            features_df: DataFrame with extracted features
            truth_labels: List of truth label dictionaries
            
        Returns:
            Tuple of (aligned_features_df, aligned_truth_labels)
        """
        logger.info("Aligning features with truth labels")
        
        # Create truth labels mapping
        truth_map = {str(item['id']): item for item in truth_labels}
        
        # Filter features to only include those with truth labels
        aligned_features = []
        aligned_truth = []
        missing_labels = 0
        
        for _, row in features_df.iterrows():
            row_id = str(row['id'])
            if row_id in truth_map:
                aligned_features.append(row.to_dict())
                aligned_truth.append(truth_map[row_id])
            else:
                missing_labels += 1
        
        logger.info(f"Aligned {len(aligned_features)} instances")
        logger.info(f"Missing truth labels: {missing_labels}")
        
        return pd.DataFrame(aligned_features), aligned_truth
    
    def validate_data(self, 
                     features_df: pd.DataFrame, 
                     truth_labels: List[Dict]) -> bool:
        """
        Validate the loaded data according to configuration.
        
        Args:
            features_df: DataFrame with features
            truth_labels: List of truth labels
            
        Returns:
            True if validation passes, False otherwise
        """
        logger.info("Validating loaded data")
        validation_config = self.config.get('validation', {})
        issues_found = []
        
        # Check for missing values
        if validation_config.get('check_missing_values', True):
            missing_primary = features_df[self.primary_text_field].isnull().sum()
            if missing_primary > 0:
                issues_found.append(f"Missing values in primary text field: {missing_primary}")
        
        # Check text length
        if validation_config.get('check_text_length', True):
            min_length = validation_config.get('min_text_length', 5)
            max_length = validation_config.get('max_text_length', 1000)
            
            primary_lengths = features_df[self.primary_text_field].str.len()
            too_short = (primary_lengths < min_length).sum()
            too_long = (primary_lengths > max_length).sum()
            
            if too_short > 0:
                issues_found.append(f"Texts shorter than {min_length} chars: {too_short}")
            if too_long > 0:
                issues_found.append(f"Texts longer than {max_length} chars: {too_long}")
        
        # Check duplicate IDs
        if validation_config.get('check_duplicate_ids', True):
            duplicate_ids = features_df['id'].duplicated().sum()
            if duplicate_ids > 0:
                issues_found.append(f"Duplicate IDs found: {duplicate_ids}")
        
        # Check label distribution
        if validation_config.get('check_label_distribution', True):
            truth_classes = [item.get('truthClass', 'unknown') for item in truth_labels]
            class_counts = pd.Series(truth_classes).value_counts()
            self.stats['label_distribution'] = class_counts.to_dict()
            
            logger.info(f"Label distribution: {class_counts.to_dict()}")
            
            # Check for severe imbalance (more than 10:1 ratio)
            if len(class_counts) >= 2:
                ratio = class_counts.max() / class_counts.min()
                if ratio > 10:
                    issues_found.append(f"Severe class imbalance detected: {ratio:.2f}:1")
        
        # Report validation results
        if issues_found:
            logger.warning("Data validation issues found:")
            for issue in issues_found:
                logger.warning(f"  - {issue}")
            return False
        else:
            logger.info("Data validation passed successfully")
            return True
    
    def load_dataset(self, 
                    instances_path: str, 
                    truth_path: str,
                    validate: bool = True) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Complete pipeline to load and process dataset.
        
        Args:
            instances_path: Path to instances JSONL file
            truth_path: Path to truth labels JSONL file
            validate: Whether to run data validation
            
        Returns:
            Tuple of (features_dataframe, truth_labels_list)
        """
        logger.info("Starting complete dataset loading pipeline")
        
        # Load raw data
        instances = self.load_instances(instances_path)
        truth_labels = self.load_truth_labels(truth_path)
        
        # Extract text features
        features_df = self.extract_text_features(instances)
        
        # Align with truth labels
        aligned_features, aligned_truth = self.align_with_truth_labels(
            features_df, truth_labels
        )
        
        # Validate if requested
        if validate:
            self.validate_data(aligned_features, aligned_truth)
        
        logger.info("Dataset loading pipeline completed successfully")
        return aligned_features, aligned_truth
    
    def get_statistics(self) -> Dict:
        """Return statistics about the loaded data."""
        return self.stats.copy()
    
    def save_processed_data(self,
                           features_df: pd.DataFrame,
                           truth_labels: List[Dict],
                           output_dir: str) -> None:
        """
        Save processed data to files.
        
        Args:
            features_df: Features DataFrame
            truth_labels: Truth labels list
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save features
        features_path = output_path / "features.csv"
        features_df.to_csv(features_path, index=False)
        logger.info(f"Features saved to {features_path}")
        
        # Save truth labels
        truth_path = output_path / "truth_labels.json"
        with open(truth_path, 'w', encoding='utf-8') as f:
            json.dump(truth_labels, f, indent=2, ensure_ascii=False)
        logger.info(f"Truth labels saved to {truth_path}")
        
        # Save statistics
        stats_path = output_path / "statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Statistics saved to {stats_path}")


def main():
    """Example usage of WebisClickbaitLoader."""
    
    # Initialize loader
    loader = WebisClickbaitLoader()
    
    # Load training data
    train_features, train_labels = loader.load_dataset(
        instances_path="data/clickbait17-validation-170630/instances.jsonl",
        truth_path="data/clickbait17-validation-170630/truth.jsonl"
    )
    
    # Load test data
    test_features, test_labels = loader.load_dataset(
        instances_path="data/clickbait17-test-170720/instances.jsonl", 
        truth_path="data/clickbait17-test-170720/truth.jsonl"
    )
    
    # Save processed data
    loader.save_processed_data(train_features, train_labels, "data/processed/train")
    loader.save_processed_data(test_features, test_labels, "data/processed/test")
    
    # Print statistics
    print("Dataset Statistics:")
    print(json.dumps(loader.get_statistics(), indent=2))


if __name__ == "__main__":
    main() 