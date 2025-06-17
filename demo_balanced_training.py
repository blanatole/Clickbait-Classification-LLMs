#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo Balanced Training - Cân bằng dữ liệu đơn giản
Dựa trên insights từ phân tích JSON
"""

import sys
import os
import json
import re
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
from datetime import datetime

# Import utilities
sys.path.append('src')
from utils import load_data, check_gpu_compatibility

def analyze_data_balance(data: List[Dict]) -> Dict:
    """Phân tích cân bằng dữ liệu"""
    
    clickbait_count = sum(1 for item in data if item['label'] == 1)
    total_count = len(data)
    no_clickbait_count = total_count - clickbait_count
    
    return {
        'total': total_count,
        'clickbait': clickbait_count,
        'no_clickbait': no_clickbait_count,
        'clickbait_ratio': clickbait_count / total_count,
        'imbalance_ratio': no_clickbait_count / clickbait_count if clickbait_count > 0 else float('inf')
    }

def oversample_minority_class(data: List[Dict], target_ratio: float = 0.4) -> List[Dict]:
    """Oversample minority class để cân bằng dữ liệu"""
    
    clickbait_data = [item for item in data if item['label'] == 1]
    no_clickbait_data = [item for item in data if item['label'] == 0]
    
    current_ratio = len(clickbait_data) / len(data)
    
    if current_ratio < target_ratio:
        # Tính số lượng clickbait cần có để đạt target_ratio
        target_clickbait = int(len(no_clickbait_data) * target_ratio / (1 - target_ratio))
        oversample_count = target_clickbait - len(clickbait_data)
        
        if oversample_count > 0:
            # Random oversample với seed cố định
            np.random.seed(42)
            oversampled_indices = np.random.choice(len(clickbait_data), oversample_count, replace=True)
            oversampled_data = [clickbait_data[i] for i in oversampled_indices]
            
            # Combine dữ liệu
            balanced_data = no_clickbait_data + clickbait_data + oversampled_data
            np.random.shuffle(balanced_data)
            
            print(f"📊 Oversampling: {len(clickbait_data)} -> {target_clickbait} clickbait samples")
            print(f"📊 Total samples: {len(data)} -> {len(balanced_data)}")
            
            return balanced_data
    
    return data

def extract_clickbait_features(text: str) -> Dict[str, float]:
    """Extract features dựa trên insights từ phân tích dữ liệu"""
    
    # Keywords từ phân tích JSON
    clickbait_indicators = ['tips', 'you', 'ways', 'reasons', 'facts', 'incredible', 'why', 'what']
    question_words = ['how', 'what', 'why', 'when', 'where', 'who']
    personal_pronouns = ['you', 'your', 'we', 'us', 'our', 'i', 'my', 'me']
    
    text_lower = text.lower()
    words = text_lower.split()
    
    features = {
        # Length features (clickbait ngắn hơn 34 ký tự)
        'text_length': len(text),
        'is_short': 1 if len(text) < 220 else 0,  # Dựa trên avg clickbait = 219
        
        # Punctuation features (clickbait có nhiều dấu hỏi và cảm hơn)
        'question_marks': text.count('?'),
        'exclamation_marks': text.count('!'),
        'has_question': 1 if '?' in text else 0,
        'has_exclamation': 1 if '!' in text else 0,
        
        # Keyword features
        'clickbait_keywords': sum(1 for word in clickbait_indicators if word in text_lower),
        'question_words': sum(1 for word in question_words if word in text_lower),
        'personal_pronouns': sum(1 for word in personal_pronouns if word in words),
        
        # Pattern features
        'has_numbers': 1 if re.search(r'\d', text) else 0,
        'uppercase_ratio': sum(1 for char in text if char.isupper()) / len(text) if text else 0,
    }
    
    return features

def create_enhanced_text(text: str, features: Dict[str, float]) -> str:
    """Tạo enhanced text với feature indicators"""
    
    # Thêm feature tokens vào text
    feature_tokens = []
    
    if features['is_short']:
        feature_tokens.append("[SHORT]")
    if features['has_question']:
        feature_tokens.append("[QUESTION]")
    if features['has_exclamation']:
        feature_tokens.append("[EXCLAMATION]")
    if features['clickbait_keywords'] > 2:
        feature_tokens.append("[CLICKBAIT_WORDS]")
    if features['personal_pronouns'] > 0:
        feature_tokens.append("[PERSONAL]")
    
    # Combine với text gốc
    enhanced_text = " ".join(feature_tokens) + " " + text if feature_tokens else text
    
    return enhanced_text

def prepare_balanced_dataset(data: List[Dict], tokenizer, use_feature_enhancement: bool = True):
    """Prepare dataset với feature enhancement"""
    
    texts = []
    labels = []
    
    for item in data:
        text = item['text']
        label = item['label']
        
        if use_feature_enhancement:
            # Extract features
            features = extract_clickbait_features(text)
            # Enhance text with feature tokens
            enhanced_text = create_enhanced_text(text, features)
            texts.append(enhanced_text)
        else:
            texts.append(text)
        
        labels.append(label)
    
    # Tokenize
    encoding = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=256,  # Shorter for demo
        return_tensors="pt"
    )
    
    return Dataset.from_dict({
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'labels': torch.tensor(labels)
    })

class WeightedTrainer(Trainer):
    """Trainer với class weights cho imbalanced data"""
    
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # Weighted CrossEntropyLoss
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = np.mean(predictions == labels)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def demo_balanced_training(
    strategy: str = "oversample",  # "oversample", "weighted_loss", "both"
    target_ratio: float = 0.35,
    use_enhancement: bool = True,
    model_name: str = "huawei-noah/TinyBERT_General_4L_312D"
):
    """
    Demo training với data balancing
    """
    
    print("🚀 DEMO BALANCED TRAINING FOR CLICKBAIT CLASSIFICATION")
    print("=" * 70)
    print(f"📊 Strategy: {strategy}")
    print(f"📊 Target ratio: {target_ratio}")
    print(f"🔧 Feature enhancement: {use_enhancement}")
    
    # Load data
    print(f"\n📁 Loading data...")
    train_data = load_data("data/train/data_demo.jsonl")  # Use demo data for speed
    val_data = load_data("data/val/data_demo.jsonl")
    
    # Analyze original balance
    print(f"\n📊 ORIGINAL DATA ANALYSIS:")
    train_balance = analyze_data_balance(train_data)
    val_balance = analyze_data_balance(val_data)
    
    print(f"Train: {train_balance['clickbait']}/{train_balance['total']} clickbait ({train_balance['clickbait_ratio']:.3f})")
    print(f"Val: {val_balance['clickbait']}/{val_balance['total']} clickbait ({val_balance['clickbait_ratio']:.3f})")
    print(f"Imbalance ratio: {train_balance['imbalance_ratio']:.1f}:1")
    
    # Apply balancing strategy
    original_train_size = len(train_data)
    
    if strategy in ["oversample", "both"]:
        print(f"\n🔄 Applying oversampling...")
        train_data = oversample_minority_class(train_data, target_ratio)
    
    # Analyze balanced data
    balanced_train_balance = analyze_data_balance(train_data)
    print(f"\n📊 BALANCED DATA ANALYSIS:")
    print(f"Train: {balanced_train_balance['clickbait']}/{balanced_train_balance['total']} clickbait ({balanced_train_balance['clickbait_ratio']:.3f})")
    print(f"Size change: {original_train_size} -> {len(train_data)} samples")
    
    # Initialize model and tokenizer
    print(f"\n🤖 Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens for feature enhancement
    if use_enhancement:
        special_tokens = ["[SHORT]", "[QUESTION]", "[EXCLAMATION]", "[CLICKBAIT_WORDS]", "[PERSONAL]"]
        tokenizer.add_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))
        print(f"✅ Added {len(special_tokens)} special feature tokens")
    
    # Prepare datasets
    print(f"\n📊 Preparing datasets...")
    train_dataset = prepare_balanced_dataset(train_data, tokenizer, use_enhancement)
    val_dataset = prepare_balanced_dataset(val_data, tokenizer, use_enhancement)
    
    print(f"✅ Train dataset: {len(train_dataset)} samples")
    print(f"✅ Val dataset: {len(val_dataset)} samples")
    
    # Calculate class weights if needed
    class_weights = None
    if strategy in ["weighted_loss", "both"]:
        train_labels = [item['label'] for item in train_data]
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights = torch.FloatTensor(class_weights)
        print(f"📊 Class weights: {class_weights.tolist()}")
    
    # Training arguments
    output_dir = f"outputs/demo_balanced_{strategy}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,  # Quick demo
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[],  # No wandb for demo
    )
    
    # Initialize trainer
    if class_weights is not None:
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
            class_weights=class_weights
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics
        )
    
    # Training
    print(f"\n🎯 Starting training...")
    start_time = datetime.now()
    
    trainer.train()
    
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"⏱️ Training completed in: {training_time}")
    
    # Final evaluation
    print(f"\n📊 Final evaluation...")
    eval_results = trainer.evaluate()
    
    print(f"\n✅ DEMO TRAINING COMPLETED!")
    print(f"📊 Results with {strategy} strategy:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save training info
    training_info = {
        'strategy': strategy,
        'target_ratio': target_ratio,
        'feature_enhancement': use_enhancement,
        'original_balance': train_balance,
        'balanced_balance': balanced_train_balance,
        'results': eval_results,
        'training_time': str(training_time)
    }
    
    with open(f"{output_dir}/training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2, default=str)
    
    print(f"\n📁 Model saved to: {output_dir}")
    
    return eval_results

def compare_strategies():
    """So sánh các strategies khác nhau"""
    
    strategies = [
        ("baseline", 0.24, False),      # No balancing
        ("oversample", 0.35, False),    # Oversample only
        ("weighted_loss", 0.24, False), # Weighted loss only
        ("oversample", 0.35, True),     # Oversample + features
        ("both", 0.35, True),           # Oversample + weighted + features
    ]
    
    results = {}
    
    print(f"\n{'='*80}")
    print(f"🧪 COMPARING STRATEGIES")
    print(f"{'='*80}")
    
    for strategy, ratio, features in strategies:
        print(f"\n{'='*50}")
        print(f"Testing: {strategy} (ratio={ratio}, features={features})")
        print(f"{'='*50}")
        
        try:
            result = demo_balanced_training(
                strategy=strategy if strategy != "baseline" else "oversample",
                target_ratio=0.24 if strategy == "baseline" else ratio,
                use_enhancement=features
            )
            results[f"{strategy}_r{ratio}_f{features}"] = result
            
        except Exception as e:
            print(f"❌ Error with {strategy}: {e}")
            results[f"{strategy}_r{ratio}_f{features}"] = {"error": str(e)}
    
    # Print comparison
    print(f"\n{'='*80}")
    print(f"📊 STRATEGY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Strategy':<20} | {'F1':<8} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8}")
    print(f"{'-'*70}")
    
    for strategy_key, result in results.items():
        if "error" not in result:
            f1 = result.get('eval_f1', 0)
            acc = result.get('eval_accuracy', 0)
            prec = result.get('eval_precision', 0)
            recall = result.get('eval_recall', 0)
            print(f"{strategy_key:<20} | {f1:<8.4f} | {acc:<8.4f} | {prec:<9.4f} | {recall:<8.4f}")
        else:
            print(f"{strategy_key:<20} | ERROR: {result['error']}")
    
    return results

def main():
    """Main function"""
    
    print("Demo: Single strategy test")
    demo_balanced_training(
        strategy="oversample",
        target_ratio=0.35,
        use_enhancement=True
    )

if __name__ == "__main__":
    main() 