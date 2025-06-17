#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions cho Clickbait Classification
"""

import json
import torch
import numpy as np
import random
import os
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClickbaitDataset(Dataset):
    """Dataset class cho clickbait classification"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.data = self.load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load data from JSONL file"""
        data = []
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")
            raise
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_device():
    """Get available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    return device

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'eval_accuracy': accuracy,
        'eval_f1': f1,
        'eval_precision': precision,
        'eval_recall': recall
    }

def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Print metrics in a nice format"""
    if prefix:
        prefix = f"{prefix} "
    
    # Handle both with and without 'eval_' prefix
    accuracy = metrics.get('eval_accuracy', metrics.get('accuracy', 0))
    f1 = metrics.get('eval_f1', metrics.get('f1', 0))
    precision = metrics.get('eval_precision', metrics.get('precision', 0))
    recall = metrics.get('eval_recall', metrics.get('recall', 0))
    
    print(f"\n📊 {prefix}METRICS:")
    print(f"   • Accuracy:  {accuracy:.4f}")
    print(f"   • F1-Score:  {f1:.4f}")
    print(f"   • Precision: {precision:.4f}")
    print(f"   • Recall:    {recall:.4f}")

def print_confusion_matrix(y_true, y_pred, labels=['No-Clickbait', 'Clickbait']):
    """Print confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n🔍 CONFUSION MATRIX:")
    print(f"                  Predicted")
    print(f"                {labels[0]:>12} {labels[1]:>12}")
    print(f"Actual {labels[0]:>8} {cm[0,0]:>12} {cm[0,1]:>12}")
    print(f"       {labels[1]:>8} {cm[1,0]:>12} {cm[1,1]:>12}")

def save_predictions(predictions: List[int], 
                    true_labels: List[int], 
                    texts: List[str], 
                    output_path: str):
    """Save predictions with texts for error analysis"""
    results = []
    for pred, true_label, text in zip(predictions, true_labels, texts):
        results.append({
            'text': text,
            'predicted': int(pred),
            'true_label': int(true_label),
            'correct': bool(pred == true_label),
            'predicted_class': 'clickbait' if pred == 1 else 'no-clickbait',
            'true_class': 'clickbait' if true_label == 1 else 'no-clickbait'
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    logger.info(f"Predictions saved to {output_path}")

def error_analysis(predictions: List[int], 
                  true_labels: List[int], 
                  texts: List[str], 
                  n_examples: int = 10):
    """Perform error analysis"""
    
    # Find errors
    errors = []
    for i, (pred, true_label, text) in enumerate(zip(predictions, true_labels, texts)):
        if pred != true_label:
            errors.append({
                'index': i,
                'text': text,
                'predicted': pred,
                'true_label': true_label,
                'predicted_class': 'clickbait' if pred == 1 else 'no-clickbait',
                'true_class': 'clickbait' if true_label == 1 else 'no-clickbait'
            })
    
    print(f"\n❌ ERROR ANALYSIS:")
    print(f"Total errors: {len(errors)} out of {len(predictions)} ({len(errors)/len(predictions)*100:.1f}%)")
    
    if errors:
        print(f"\n🔍 First {min(n_examples, len(errors))} errors:")
        for i, error in enumerate(errors[:n_examples]):
            print(f"\n{i+1}. Text: {error['text'][:100]}...")
            print(f"   Predicted: {error['predicted_class']}")
            print(f"   True:      {error['true_class']}")
    
    return errors

def get_model_info(model_path: str) -> Dict[str, Any]:
    """Get model information"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'model_path': model_path,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'vocab_size': tokenizer.vocab_size,
            'max_position_embeddings': getattr(model.config, 'max_position_embeddings', 'Unknown'),
            'hidden_size': getattr(model.config, 'hidden_size', 'Unknown'),
            'num_hidden_layers': getattr(model.config, 'num_hidden_layers', 'Unknown'),
            'num_attention_heads': getattr(model.config, 'num_attention_heads', 'Unknown'),
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return {'error': str(e)}

def estimate_gpu_memory(batch_size: int, 
                       max_length: int, 
                       hidden_size: int = 768, 
                       num_layers: int = 12) -> float:
    """Estimate GPU memory usage (rough estimation)"""
    # Rough estimation based on common transformer architectures
    # This is a simplified calculation
    
    # Input tensors
    input_memory = batch_size * max_length * 4  # 4 bytes per float32
    
    # Model parameters (rough estimate)
    model_memory = hidden_size * hidden_size * num_layers * 4 * 12  # Multiple matrices per layer
    
    # Gradients (same size as parameters)
    gradient_memory = model_memory
    
    # Optimizer states (Adam uses ~2x parameter memory)
    optimizer_memory = model_memory * 2
    
    # Activations (rough estimate)
    activation_memory = batch_size * max_length * hidden_size * num_layers * 4
    
    total_bytes = input_memory + model_memory + gradient_memory + optimizer_memory + activation_memory
    total_gb = total_bytes / (1024**3)
    
    return total_gb

def check_gpu_compatibility(config, verbose: bool = True):
    """Check if current GPU is compatible with config"""
    if not torch.cuda.is_available():
        if verbose:
            print("⚠️ No GPU available, will use CPU (very slow)")
        return False
    
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    gpu_name = torch.cuda.get_device_name(0)
    
    # Extract memory requirement from config
    memory_req = config.gpu_memory_required
    if 'GB' in memory_req:
        # Parse memory requirement (e.g., "4-6GB" -> 6.0)
        numbers = [float(x) for x in memory_req.replace('GB', '').replace('-', ' ').split() if x.replace('.', '').isdigit()]
        required_memory = max(numbers) if numbers else 8.0
    else:
        required_memory = 8.0  # Default assumption
    
    compatible = gpu_memory_gb >= required_memory
    
    if verbose:
        status = "✅" if compatible else "❌"
        print(f"{status} GPU: {gpu_name} ({gpu_memory_gb:.1f}GB)")
        print(f"   Required: {memory_req}")
        print(f"   Compatible: {'Yes' if compatible else 'No'}")
        
        if not compatible:
            print(f"   💡 Suggestion: Use a model with lower memory requirements")
    
    return compatible

def create_data_loaders(train_path: str, 
                       val_path: str, 
                       test_path: str, 
                       tokenizer, 
                       config,
                       demo_mode: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders"""
    
    # Use demo data if in demo mode
    suffix = "_demo.jsonl" if demo_mode else ".jsonl"
    train_path = train_path.replace(".jsonl", suffix)
    val_path = val_path.replace(".jsonl", suffix)
    test_path = test_path.replace(".jsonl", suffix)
    
    # Create datasets
    train_dataset = ClickbaitDataset(train_path, tokenizer, config.max_length)
    val_dataset = ClickbaitDataset(val_path, tokenizer, config.max_length)
    test_dataset = ClickbaitDataset(test_path, tokenizer, config.max_length)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size_train,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size_eval,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size_eval,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    logger.info(f"Data loaders created:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

def format_time(seconds: float) -> str:
    """Format time in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def print_training_info(config, demo_mode: bool = False):
    """Print training information"""
    mode = "DEMO" if demo_mode else "FULL"
    
    print(f"\n🚀 TRAINING INFO ({mode} MODE)")
    print("=" * 50)
    print(f"Model: {config.model_name}")
    print(f"Base Model: {config.model_path}")
    print(f"Max Length: {config.max_length}")
    print(f"Train Batch Size: {config.batch_size_train}")
    print(f"Eval Batch Size: {config.batch_size_eval}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Epochs: {config.num_epochs}")
    print(f"GPU Memory Required: {config.gpu_memory_required}")
    print(f"FP16: {config.fp16}")
    print(f"Gradient Accumulation: {config.gradient_accumulation_steps}")

if __name__ == "__main__":
    # Test functions
    print("🔧 Testing utility functions...")
    
    # Test device detection
    device = get_device()
    
    # Test model info
    print("\n📊 Testing model info...")
    try:
        info = get_model_info("distilbert-base-uncased")
        print(f"Model parameters: {info['total_parameters']:,}")
    except Exception as e:
        print(f"Could not load model info: {e}")
    
    print("\n✅ Utils test completed!") 