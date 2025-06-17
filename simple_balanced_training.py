#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Balanced Training Demo - Cân bằng dữ liệu cho clickbait classification
Dựa trên insights từ phân tích JSON: 24.08% vs 75.92% imbalance
"""

import json
import os
import re
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
from datetime import datetime

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
    return data

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

def oversample_minority_class(data: List[Dict], target_ratio: float = 0.35) -> List[Dict]:
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
    
    # Keywords từ phân tích JSON - top discriminative keywords
    clickbait_indicators = ['tips', 'you', 'ways', 'reasons', 'facts', 'incredible', 'why', 'what']
    question_words = ['how', 'what', 'why', 'when', 'where', 'who']
    personal_pronouns = ['you', 'your', 'we', 'us', 'our', 'i', 'my', 'me']
    
    text_lower = text.lower()
    words = text_lower.split()
    
    features = {
        # Length features (clickbait trung bình 219 ký tự vs no-clickbait 253)
        'text_length': len(text),
        'is_short': 1 if len(text) < 220 else 0,
        
        # Punctuation features (clickbait: 17% có ?, 6.3% có !)
        'question_marks': text.count('?'),
        'exclamation_marks': text.count('!'),
        'has_question': 1 if '?' in text else 0,
        'has_exclamation': 1 if '!' in text else 0,
        
        # Keyword features từ analysis
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
    
    # Thêm feature tokens dựa trên analysis insights
    feature_tokens = []
    
    if features['is_short']:
        feature_tokens.append("[SHORT]")
    if features['has_question']:
        feature_tokens.append("[QUESTION]")
    if features['has_exclamation']:
        feature_tokens.append("[EXCLAMATION]")
    if features['clickbait_keywords'] > 1:
        feature_tokens.append("[CLICKBAIT_WORDS]")
    if features['personal_pronouns'] > 0:
        feature_tokens.append("[PERSONAL]")
    
    # Combine với text gốc
    enhanced_text = " ".join(feature_tokens) + " " + text if feature_tokens else text
    
    return enhanced_text

class WeightedTrainer(Trainer):
    """Trainer với class weights cho imbalanced data"""
    
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # Weighted CrossEntropyLoss để handle imbalance
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = np.mean(predictions == labels)
    
    # Also compute per-class metrics
    cb_precision, cb_recall, cb_f1, _ = precision_recall_fscore_support(labels, predictions, average=None)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'clickbait_f1': cb_f1[1] if len(cb_f1) > 1 else 0,  # F1 for clickbait class
        'no_clickbait_f1': cb_f1[0] if len(cb_f1) > 0 else 0  # F1 for no-clickbait class
    }

def demo_balanced_training():
    """
    Demo training với cân bằng dữ liệu và feature engineering
    """
    
    print("🚀 DEMO: BALANCED TRAINING FOR CLICKBAIT CLASSIFICATION")
    print("Dựa trên insights từ phân tích JSON: 24.08% vs 75.92% imbalance")
    print("=" * 80)
    
    # Load demo data
    print(f"\n📁 Loading demo data...")
    train_data = load_jsonl("data/train/data_demo.jsonl")
    val_data = load_jsonl("data/val/data_demo.jsonl")
    
    if not train_data or not val_data:
        print("❌ Could not load data files!")
        return
    
    # Analyze original imbalance
    print(f"\n📊 ORIGINAL DATA ANALYSIS:")
    train_balance = analyze_data_balance(train_data)
    val_balance = analyze_data_balance(val_data)
    
    print(f"Train: {train_balance['clickbait']}/{train_balance['total']} clickbait ({train_balance['clickbait_ratio']:.3f})")
    print(f"Val: {val_balance['clickbait']}/{val_balance['total']} clickbait ({val_balance['clickbait_ratio']:.3f})")
    print(f"Imbalance ratio: {train_balance['imbalance_ratio']:.1f}:1 (no-clickbait:clickbait)")
    
    # Show some examples with features
    print(f"\n🔍 FEATURE ANALYSIS EXAMPLES:")
    for i, item in enumerate(train_data[:3]):
        features = extract_clickbait_features(item['text'])
        enhanced = create_enhanced_text(item['text'], features)
        label_name = "CLICKBAIT" if item['label'] == 1 else "NO-CLICKBAIT"
        
        print(f"\nExample {i+1} ({label_name}):")
        print(f"Original: {item['text'][:100]}...")
        print(f"Enhanced: {enhanced[:100]}...")
        print(f"Features: length={features['text_length']}, ?={features['question_marks']}, !={features['exclamation_marks']}, keywords={features['clickbait_keywords']}")
    
    # Apply oversampling
    target_ratio = 0.35  # Target 35% clickbait instead of 24%
    print(f"\n🔄 Applying oversampling to achieve {target_ratio:.0%} clickbait ratio...")
    
    original_train_size = len(train_data)
    balanced_train_data = oversample_minority_class(train_data, target_ratio)
    
    # Analyze balanced data
    balanced_train_balance = analyze_data_balance(balanced_train_data)
    print(f"\n📊 BALANCED DATA ANALYSIS:")
    print(f"Train: {balanced_train_balance['clickbait']}/{balanced_train_balance['total']} clickbait ({balanced_train_balance['clickbait_ratio']:.3f})")
    print(f"Size change: {original_train_size} -> {len(balanced_train_data)} samples")
    print(f"New imbalance ratio: {balanced_train_balance['imbalance_ratio']:.1f}:1")
    
    # Initialize model and tokenizer
    model_name = "huawei-noah/TinyBERT_General_4L_312D"
    print(f"\n🤖 Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add special feature tokens
    special_tokens = ["[SHORT]", "[QUESTION]", "[EXCLAMATION]", "[CLICKBAIT_WORDS]", "[PERSONAL]"]
    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    print(f"✅ Added {len(special_tokens)} special feature tokens")
    
    # Prepare enhanced datasets
    print(f"\n📊 Preparing enhanced datasets...")
    
    # Train dataset (balanced + enhanced)
    train_texts = []
    train_labels = []
    
    for item in balanced_train_data:
        features = extract_clickbait_features(item['text'])
        enhanced_text = create_enhanced_text(item['text'], features)
        train_texts.append(enhanced_text)
        train_labels.append(item['label'])
    
    # Val dataset (original + enhanced)
    val_texts = []
    val_labels = []
    
    for item in val_data:
        features = extract_clickbait_features(item['text'])
        enhanced_text = create_enhanced_text(item['text'], features)
        val_texts.append(enhanced_text)
        val_labels.append(item['label'])
    
    # Tokenize
    train_encoding = tokenizer(train_texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
    val_encoding = tokenizer(val_texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        'input_ids': train_encoding['input_ids'],
        'attention_mask': train_encoding['attention_mask'],
        'labels': torch.tensor(train_labels)
    })
    
    val_dataset = Dataset.from_dict({
        'input_ids': val_encoding['input_ids'],
        'attention_mask': val_encoding['attention_mask'],
        'labels': torch.tensor(val_labels)
    })
    
    print(f"✅ Train dataset: {len(train_dataset)} samples (enhanced)")
    print(f"✅ Val dataset: {len(val_dataset)} samples (enhanced)")
    
    # Calculate class weights for weighted loss
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.FloatTensor(class_weights)
    print(f"📊 Class weights: {class_weights.tolist()}")
    
    # Training arguments
    output_dir = "outputs/demo_balanced_enhanced"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,  # Quick demo
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=20,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[],
    )
    
    # Initialize weighted trainer
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
    
    # Training
    print(f"\n🎯 Starting training with balanced data + feature enhancement + weighted loss...")
    start_time = datetime.now()
    
    trainer.train()
    
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"⏱️ Training completed in: {training_time}")
    
    # Final evaluation
    print(f"\n📊 Final evaluation on original (imbalanced) validation set...")
    eval_results = trainer.evaluate()
    
    print(f"\n✅ DEMO COMPLETED!")
    print(f"📊 Results with balanced training:")
    print(f"   Overall F1: {eval_results['eval_f1']:.4f}")
    print(f"   Overall Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"   Clickbait F1: {eval_results['eval_clickbait_f1']:.4f}")
    print(f"   No-clickbait F1: {eval_results['eval_no_clickbait_f1']:.4f}")
    print(f"   Precision: {eval_results['eval_precision']:.4f}")
    print(f"   Recall: {eval_results['eval_recall']:.4f}")
    
    # Save model and results
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save training summary
    training_summary = {
        'original_balance': train_balance,
        'balanced_balance': balanced_train_balance,
        'target_ratio': target_ratio,
        'strategies_used': [
            'Oversampling minority class',
            'Feature engineering with special tokens',
            'Weighted CrossEntropyLoss',
            'Enhanced text with feature indicators'
        ],
        'results': eval_results,
        'training_time': str(training_time),
        'insights_applied': [
            'Clickbait shorter than no-clickbait (219 vs 253 chars)',
            'Clickbait has more question marks (17% vs 6.5%)',
            'Clickbait has more exclamation marks (6.3% vs 3.2%)',
            'Keywords: tips, you, ways, reasons more common in clickbait',
            'Personal pronouns more frequent in clickbait (34.3% vs 14.3%)'
        ]
    }
    
    with open(f"{output_dir}/training_summary.json", 'w') as f:
        json.dump(training_summary, f, indent=2, default=str)
    
    print(f"\n📁 Model saved to: {output_dir}")
    print(f"📋 Training summary saved to: {output_dir}/training_summary.json")
    
    print(f"\n💡 KEY IMPROVEMENTS APPLIED:")
    print(f"   🎯 Balanced training data: {train_balance['clickbait_ratio']:.3f} -> {balanced_train_balance['clickbait_ratio']:.3f}")
    print(f"   🔧 Feature engineering: Added {len(special_tokens)} special tokens")
    print(f"   ⚖️ Weighted loss: Class weights {class_weights.tolist()}")
    print(f"   📈 Expected improvement: Better minority class (clickbait) detection")
    
    return eval_results

def main():
    """Main function"""
    demo_balanced_training()

if __name__ == "__main__":
    main() 