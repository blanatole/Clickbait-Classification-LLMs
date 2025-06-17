#!/usr/bin/env python3
"""
Fine-tune BERT family models (BERT-base, BERT-large) 
Optimized for RTX A5000 (24 GB VRAM)
Compatible with PyTorch 2.5+ and latest transformers
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np


# Model configurations optimized for RTX A5000 (24GB)
MODEL_CONFIGS = {
    "bert-base-uncased": {
        "model_name": "bert-base-uncased",
        "batch_size": 48,  # Conservative for 24GB
        "learning_rate": 2e-5,
        "epochs": 4,
        "max_length": 128,
        "fp16": True,
        "gradient_accumulation_steps": 1
    },
    "bert-large-uncased": {
        "model_name": "bert-large-uncased",
        "batch_size": 16,  # Much larger model
        "learning_rate": 1e-5,
        "epochs": 3,
        "max_length": 128,
        "fp16": True,
        "gradient_accumulation_steps": 2  # Effective batch = 32
    }
}


def setup_model_and_tokenizer(config):
    """Setup model and tokenizer"""
    print(f"🤖 Loading {config['model_name']}...")
    
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=2
    )
    
    return model, tokenizer


def prepare_dataset(tokenizer, max_length=128):
    """Load and tokenize dataset"""
    print("📊 Loading and tokenizing dataset...")
    
    data_files = {
        "train": "data/train/data.jsonl",
        "validation": "data/val/data.jsonl",
        "test": "data/test/data.jsonl"
    }
    
    # Check if data files exist
    for split, file_path in data_files.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
    
    dataset = load_dataset("json", data_files=data_files)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=True
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average='weighted'),
        "f1_macro": f1_score(labels, predictions, average='macro'),
        "f1_binary": f1_score(labels, predictions, average='binary')
    }


def train_model(model_key, config, output_base_dir="outputs"):
    """Train a single model"""
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING {model_key.upper()}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Setup paths
    output_dir = f"{output_base_dir}/{model_key}-a5000"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Prepare dataset
    tokenized_dataset = prepare_dataset(tokenizer, config["max_length"])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["epochs"],
        weight_decay=0.01,
        warmup_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=config["fp16"],
        dataloader_num_workers=4,
        gradient_checkpointing=True,  # Save VRAM
        report_to=None,
        save_total_limit=2  # Keep only best 2 checkpoints
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print("🏋️ Starting training...")
    train_result = trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Evaluate on test set
    print("🧪 Evaluating on test set...")
    test_results = trainer.predict(tokenized_dataset["test"])
    test_predictions = test_results.predictions.argmax(axis=-1)
    test_labels = tokenized_dataset["test"]["label"]
    
    # Calculate metrics
    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_f1 = f1_score(test_labels, test_predictions, average='weighted')
    test_f1_macro = f1_score(test_labels, test_predictions, average='macro')
    test_f1_binary = f1_score(test_labels, test_predictions, average='binary')
    
    # Training time
    training_time = time.time() - start_time
    
    # Detailed classification report
    class_report = classification_report(
        test_labels, 
        test_predictions,
        target_names=["Not Clickbait", "Clickbait"],
        output_dict=True
    )
    
    # Save results
    results = {
        "model_name": config["model_name"],
        "model_key": model_key,
        "training_config": config,
        "training_time_seconds": training_time,
        "training_time_formatted": f"{training_time//3600:.0f}h {(training_time%3600)//60:.0f}m {training_time%60:.0f}s",
        "test_metrics": {
            "accuracy": float(test_accuracy),
            "f1_weighted": float(test_f1),
            "f1_macro": float(test_f1_macro),
            "f1_binary": float(test_f1_binary)
        },
        "classification_report": class_report,
        "train_loss": float(train_result.training_loss),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print(f"\n📊 RESULTS FOR {model_key.upper()}:")
    print(f"   Accuracy: {test_accuracy:.4f}")
    print(f"   F1 (weighted): {test_f1:.4f}")
    print(f"   F1 (macro): {test_f1_macro:.4f}")
    print(f"   F1 (binary): {test_f1_binary:.4f}")
    print(f"   Training time: {results['training_time_formatted']}")
    print(f"   Model saved to: {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train BERT family models on RTX A5000")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()) + ["all"], 
                       default="all", help="Model to train")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🔧 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        if "A5000" not in gpu_name and gpu_memory < 20:
            print("⚠️  Warning: This script is optimized for RTX A5000 (24GB)")
            print("   Consider reducing batch sizes for smaller GPUs")
    else:
        print("❌ No CUDA GPU detected!")
        return
    
    # Train models
    all_results = {}
    
    if args.model == "all":
        models_to_train = list(MODEL_CONFIGS.keys())
    else:
        models_to_train = [args.model]
    
    print(f"🎯 Training {len(models_to_train)} model(s): {', '.join(models_to_train)}")
    
    for model_key in models_to_train:
        try:
            config = MODEL_CONFIGS[model_key]
            results = train_model(model_key, config, args.output_dir)
            all_results[model_key] = results
            
            # Clean up GPU memory
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Error training {model_key}: {e}")
            continue
    
    # Save summary results
    summary_file = f"{args.output_dir}/bert_family_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("📋 TRAINING SUMMARY")
    print(f"{'='*80}")
    
    for model_key, results in all_results.items():
        metrics = results["test_metrics"]
        print(f"{model_key:20} | F1: {metrics['f1_weighted']:.4f} | "
              f"Acc: {metrics['accuracy']:.4f} | "
              f"Time: {results['training_time_formatted']}")
    
    print(f"\n✅ All results saved to: {summary_file}")


if __name__ == "__main__":
    main() 