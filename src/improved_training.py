#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Training Script - Xử lý cân bằng dữ liệu và feature engineering
Dựa trên insights từ phân tích dữ liệu JSON
"""

import json
import os
import re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
import wandb
from datetime import datetime

# Import utilities
from utils import load_data, check_gpu_compatibility, calculate_metrics

class FeatureEngineer:
    """Feature engineering based on data analysis insights"""
    
    def __init__(self):
        # Keywords từ phân tích dữ liệu
        self.clickbait_keywords = ['tips', 'you', 'ways', 'reasons', 'facts', 'incredible', 'why', 'what']
        self.question_words = ['how', 'what', 'why', 'when', 'where', 'who']
        self.superlatives = ['best', 'worst', 'most', 'least', 'biggest', 'smallest', 'amazing', 'incredible']
        self.time_urgency = ['now', 'today', 'immediately', 'urgent', 'never', 'always']
        self.personal_pronouns = ['you', 'your', 'we', 'us', 'our', 'i', 'my', 'me']
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract engineered features from text"""
        text_lower = text.lower()
        words = text_lower.split()
        
        features = {
            # Length features
            'text_length': len(text),
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            
            # Punctuation features
            'question_marks': text.count('?'),
            'exclamation_marks': text.count('!'),
            'punctuation_ratio': (text.count('?') + text.count('!')) / len(text) if len(text) > 0 else 0,
            
            # Keyword features
            'clickbait_keywords_count': sum(1 for kw in self.clickbait_keywords if kw in text_lower),
            'question_words_count': sum(1 for qw in self.question_words if qw in text_lower),
            'superlatives_count': sum(1 for sup in self.superlatives if sup in text_lower),
            'time_urgency_count': sum(1 for tu in self.time_urgency if tu in text_lower),
            'personal_pronouns_count': sum(1 for pp in self.personal_pronouns if pp in words),
            
            # Number features
            'has_numbers': 1 if re.search(r'\d', text) else 0,
            'numbers_count': len(re.findall(r'\d+', text)),
            
            # Formatting features
            'has_quotes': 1 if '"' in text or "'" in text else 0,
            'uppercase_words': sum(1 for word in words if word.isupper() and len(word) > 1),
            'title_case': 1 if text.istitle() else 0,
            
            # Ratio features
            'pronoun_ratio': sum(1 for pp in self.personal_pronouns if pp in words) / len(words) if words else 0,
            'question_ratio': sum(1 for qw in self.question_words if qw in text_lower) / len(words) if words else 0,
        }
        
        return features

class ImbalancedDatasetHandler:
    """Xử lý cân bằng dữ liệu"""
    
    def __init__(self, strategy='weighted_loss'):
        """
        strategy: 'weighted_loss', 'oversample', 'undersample', 'focal_loss'
        """
        self.strategy = strategy
    
    def get_class_weights(self, labels: List[int]) -> torch.Tensor:
        """Tính class weights cho weighted loss"""
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        return torch.FloatTensor(class_weights)
    
    def oversample_minority(self, data: List[Dict], target_ratio: float = 0.4) -> List[Dict]:
        """Oversample minority class (clickbait)"""
        clickbait_data = [item for item in data if item['label'] == 1]
        no_clickbait_data = [item for item in data if item['label'] == 0]
        
        current_ratio = len(clickbait_data) / len(data)
        
        if current_ratio < target_ratio:
            # Tính số lượng cần oversample
            target_clickbait = int(len(no_clickbait_data) * target_ratio / (1 - target_ratio))
            oversample_count = target_clickbait - len(clickbait_data)
            
            # Random oversample
            np.random.seed(42)
            oversampled = np.random.choice(clickbait_data, oversample_count, replace=True).tolist()
            
            balanced_data = no_clickbait_data + clickbait_data + oversampled
            np.random.shuffle(balanced_data)
            
            print(f"📊 Oversampling: {len(clickbait_data)} -> {target_clickbait} clickbait samples")
            print(f"📊 New ratio: {target_clickbait / len(balanced_data):.3f}")
            
            return balanced_data
        
        return data
    
    def undersample_majority(self, data: List[Dict], target_ratio: float = 0.4) -> List[Dict]:
        """Undersample majority class (no-clickbait)"""
        clickbait_data = [item for item in data if item['label'] == 1]
        no_clickbait_data = [item for item in data if item['label'] == 0]
        
        current_ratio = len(clickbait_data) / len(data)
        
        if current_ratio < target_ratio:
            # Tính số lượng cần giữ lại
            target_no_clickbait = int(len(clickbait_data) * (1 - target_ratio) / target_ratio)
            
            # Random undersample
            np.random.seed(42)
            undersampled = np.random.choice(no_clickbait_data, target_no_clickbait, replace=False).tolist()
            
            balanced_data = clickbait_data + undersampled
            np.random.shuffle(balanced_data)
            
            print(f"📊 Undersampling: {len(no_clickbait_data)} -> {target_no_clickbait} no-clickbait samples")
            print(f"📊 New ratio: {len(clickbait_data) / len(balanced_data):.3f}")
            
            return balanced_data
        
        return data

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class CustomTrainer(Trainer):
    """Custom Trainer với weighted loss hoặc focal loss"""
    
    def __init__(self, *args, class_weights=None, use_focal_loss=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        
        if use_focal_loss:
            self.focal_loss = FocalLoss(weight=class_weights)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.use_focal_loss:
            loss = self.focal_loss(logits, labels)
        else:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def prepare_enhanced_dataset(data: List[Dict], tokenizer, feature_engineer: FeatureEngineer, max_length: int = 512):
    """Prepare dataset with engineered features"""
    
    texts = []
    labels = []
    features_list = []
    
    for item in data:
        text = item['text']
        label = item['label']
        
        # Extract engineered features
        features = feature_engineer.extract_features(text)
        
        texts.append(text)
        labels.append(label)
        features_list.append(features)
    
    # Tokenize
    encoding = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Convert to dataset
    dataset_dict = {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'labels': torch.tensor(labels)
    }
    
    # Add engineered features
    feature_names = list(features_list[0].keys())
    for feature_name in feature_names:
        feature_values = [features[feature_name] for features in features_list]
        dataset_dict[f'feat_{feature_name}'] = torch.tensor(feature_values, dtype=torch.float32)
    
    return Dataset.from_dict(dataset_dict), feature_names

class EnhancedClassificationModel(nn.Module):
    """Enhanced model with feature fusion"""
    
    def __init__(self, base_model, num_features: int, hidden_size: int = 768):
        super().__init__()
        self.base_model = base_model
        self.feature_projection = nn.Linear(num_features, hidden_size // 4)
        self.classifier = nn.Linear(hidden_size + hidden_size // 4, 2)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Extract engineered features
        feature_inputs = []
        for key, value in kwargs.items():
            if key.startswith('feat_'):
                feature_inputs.append(value.unsqueeze(-1))
        
        # Base model forward
        base_outputs = self.base_model.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get pooled output
        pooled_output = base_outputs.last_hidden_state[:, 0]  # CLS token
        
        # Feature fusion
        if feature_inputs:
            features_tensor = torch.cat(feature_inputs, dim=-1)
            feature_emb = self.feature_projection(features_tensor)
            feature_emb = self.dropout(feature_emb)
            
            # Combine text and features
            combined = torch.cat([pooled_output, feature_emb], dim=-1)
        else:
            combined = pooled_output
        
        # Classification
        logits = self.classifier(self.dropout(combined))
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {'loss': loss, 'logits': logits}

def train_improved_model(
    config_name: str = "tinybert_demo",
    use_balancing: str = "weighted_loss",  # "weighted_loss", "oversample", "undersample", "focal_loss"
    target_ratio: float = 0.35,  # Target clickbait ratio after balancing
    use_feature_fusion: bool = True,
    output_dir: str = "outputs/improved_model"
):
    """
    Train improved model with data balancing and feature engineering
    """
    
    print("🚀 IMPROVED CLICKBAIT CLASSIFICATION TRAINING")
    print("=" * 60)
    print(f"📊 Data balancing strategy: {use_balancing}")
    print(f"📊 Target clickbait ratio: {target_ratio}")
    print(f"🔧 Feature fusion: {use_feature_fusion}")
    
    # Load configuration
    from configs.model_configs import get_model_config
    config = get_model_config(config_name)
    
    # Check GPU compatibility
    device, gpu_info = check_gpu_compatibility(config['min_gpu_memory'])
    print(f"\n🔧 Using device: {device}")
    if gpu_info:
        print(f"🔧 GPU: {gpu_info}")
    
    # Initialize components
    feature_engineer = FeatureEngineer()
    data_handler = ImbalancedDatasetHandler(strategy=use_balancing)
    
    # Load data
    print(f"\n📁 Loading training data...")
    train_data = load_data("data/train/data.jsonl")
    val_data = load_data("data/val/data.jsonl")
    
    print(f"✅ Original train data: {len(train_data)} samples")
    original_ratio = sum(1 for item in train_data if item['label'] == 1) / len(train_data)
    print(f"📊 Original clickbait ratio: {original_ratio:.3f}")
    
    # Apply data balancing
    if use_balancing == "oversample":
        train_data = data_handler.oversample_minority(train_data, target_ratio)
    elif use_balancing == "undersample":
        train_data = data_handler.undersample_majority(train_data, target_ratio)
    
    new_ratio = sum(1 for item in train_data if item['label'] == 1) / len(train_data)
    print(f"✅ Balanced train data: {len(train_data)} samples")
    print(f"📊 New clickbait ratio: {new_ratio:.3f}")
    
    # Initialize tokenizer and model
    print(f"\n🤖 Loading model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare datasets
    print(f"\n📊 Preparing datasets with feature engineering...")
    train_dataset, feature_names = prepare_enhanced_dataset(train_data, tokenizer, feature_engineer)
    val_dataset, _ = prepare_enhanced_dataset(val_data, tokenizer, feature_engineer)
    
    print(f"✅ Engineered features: {len(feature_names)}")
    print(f"📋 Features: {', '.join(feature_names[:5])}...")
    
    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config['model_name'],
        num_labels=2
    )
    
    # Enhanced model with feature fusion
    if use_feature_fusion:
        model = EnhancedClassificationModel(base_model, len(feature_names))
        print(f"🔧 Using enhanced model with feature fusion")
    else:
        model = base_model
        print(f"🔧 Using base model without feature fusion")
    
    model.to(device)
    
    # Calculate class weights
    train_labels = [item['label'] for item in train_data]
    class_weights = None
    use_focal = False
    
    if use_balancing == "weighted_loss":
        class_weights = data_handler.get_class_weights(train_labels).to(device)
        print(f"📊 Class weights: {class_weights.tolist()}")
    elif use_balancing == "focal_loss":
        class_weights = data_handler.get_class_weights(train_labels).to(device)
        use_focal = True
        print(f"📊 Using Focal Loss with weights: {class_weights.tolist()}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get('epochs', 3),
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=["wandb"] if config.get('use_wandb', False) else [],
        dataloader_num_workers=0 if device.type == 'cpu' else 2,
        fp16=device.type == 'cuda' and config.get('use_fp16', False),
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Custom trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=calculate_metrics,
        class_weights=class_weights,
        use_focal_loss=use_focal
    )
    
    # Initialize wandb if enabled
    if config.get('use_wandb', False):
        wandb.init(
            project="clickbait-improved",
            config={
                **config,
                'balancing_strategy': use_balancing,
                'target_ratio': target_ratio,
                'feature_fusion': use_feature_fusion,
                'original_ratio': original_ratio,
                'new_ratio': new_ratio
            }
        )
    
    # Training
    print(f"\n🎯 Starting training...")
    start_time = datetime.now()
    
    trainer.train()
    
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"⏱️ Training completed in: {training_time}")
    
    # Save model
    print(f"\n💾 Saving improved model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save feature engineer config
    feature_config = {
        'feature_names': feature_names,
        'clickbait_keywords': feature_engineer.clickbait_keywords,
        'question_words': feature_engineer.question_words,
        'superlatives': feature_engineer.superlatives,
        'time_urgency': feature_engineer.time_urgency,
        'personal_pronouns': feature_engineer.personal_pronouns
    }
    
    with open(f"{output_dir}/feature_config.json", 'w') as f:
        json.dump(feature_config, f, indent=2)
    
    # Final evaluation
    print(f"\n📊 Final evaluation...")
    eval_results = trainer.evaluate()
    
    print(f"\n✅ IMPROVED TRAINING COMPLETED!")
    print(f"📊 Final metrics:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
    
    print(f"\n📁 Model saved to: {output_dir}")
    print(f"🔧 Features config saved to: {output_dir}/feature_config.json")
    
    return trainer, eval_results

def main():
    """Main function with different balancing strategies"""
    
    # Test different strategies
    strategies = [
        ("weighted_loss", 0.35),
        ("oversample", 0.35),
        ("focal_loss", 0.35)
    ]
    
    results = {}
    
    for strategy, ratio in strategies:
        print(f"\n{'='*80}")
        print(f"🧪 TESTING STRATEGY: {strategy.upper()}")
        print(f"{'='*80}")
        
        output_dir = f"outputs/improved_{strategy}"
        
        try:
            trainer, eval_results = train_improved_model(
                config_name="tinybert_demo",
                use_balancing=strategy,
                target_ratio=ratio,
                use_feature_fusion=True,
                output_dir=output_dir
            )
            
            results[strategy] = eval_results
            
        except Exception as e:
            print(f"❌ Error with strategy {strategy}: {e}")
            results[strategy] = {"error": str(e)}
    
    # Compare results
    print(f"\n{'='*80}")
    print(f"📊 STRATEGY COMPARISON")
    print(f"{'='*80}")
    
    for strategy, result in results.items():
        if "error" not in result:
            f1 = result.get('eval_f1', 0)
            acc = result.get('eval_accuracy', 0)
            print(f"{strategy.upper():15} | F1: {f1:.4f} | Accuracy: {acc:.4f}")
        else:
            print(f"{strategy.upper():15} | ERROR: {result['error']}")

if __name__ == "__main__":
    main() 