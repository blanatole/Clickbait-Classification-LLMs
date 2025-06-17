#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo Training Script cho RTX 3050
Train models nhẹ với demo dataset để test nhanh
"""

import os
import sys
import time
import torch
import argparse
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.model_configs import get_config, get_demo_configs, print_config_summary
from src.utils import (
    set_seed, get_device, compute_metrics, print_metrics, 
    check_gpu_compatibility, create_data_loaders, print_training_info,
    print_confusion_matrix, error_analysis, format_time
)

def main():
    parser = argparse.ArgumentParser(description="Demo Training for RTX 3050")
    parser.add_argument("--model", type=str, default="tinybert", 
                       choices=list(get_demo_configs().keys()),
                       help="Model to train")
    parser.add_argument("--data_dir", type=str, default="data", 
                       help="Data directory")
    parser.add_argument("--output_dir", type=str, default="outputs", 
                       help="Output directory")
    parser.add_argument("--demo_size", type=int, default=100,
                       help="Size of demo dataset (already created)")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    parser.add_argument("--quick_test", action="store_true",
                       help="Very quick test with 1 epoch")
    
    args = parser.parse_args()
    
    print("🚀 CLICKBAIT DEMO TRAINING")
    print(f"📱 Optimized for RTX 3050 (8GB VRAM)")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Get device info
    device = get_device()
    
    # Load configuration
    config = get_config(args.model)
    
    # Check GPU compatibility
    print(f"\n🔍 Checking GPU compatibility...")
    compatible = check_gpu_compatibility(config)
    if not compatible and torch.cuda.is_available():
        print("⚠️ GPU might not have enough memory. Continuing anyway...")
    
    # Quick test mode
    if args.quick_test:
        print("⚡ Quick test mode: reducing epochs and steps")
        config.num_epochs = 1
        config.save_steps = 50
        config.eval_steps = 25
        config.logging_steps = 10
    
    # Print training info
    print_training_info(config, demo_mode=True)
    
    # Setup paths
    train_path = os.path.join(args.data_dir, "train", "data.jsonl")
    val_path = os.path.join(args.data_dir, "val", "data.jsonl")
    test_path = os.path.join(args.data_dir, "test", "data.jsonl")
    
    output_dir = os.path.join(args.output_dir, f"{config.model_name}_demo")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📂 Setting up model...")
    start_time = time.time()
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_path, 
            num_labels=2
        )
        
        # Add pad token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"✅ Model loaded: {config.model_path}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Parameters: {total_params:,} total, {trainable_params:,} trainable")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create data loaders (demo mode)
    print(f"\n📊 Loading demo datasets...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            train_path, val_path, test_path, tokenizer, config, demo_mode=True
        )
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size_train,
        per_device_eval_batch_size=config.batch_size_eval,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        fp16=config.fp16 and torch.cuda.is_available(),
        logging_dir=f'{output_dir}/logs',
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps", 
        save_steps=config.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=False,
        report_to="none",  # Disable wandb for demo
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)]
    )
    
    # Start training
    print(f"\n🚀 Starting training...")
    print(f"   Demo dataset: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Batch size: {config.batch_size_train}")
    
    try:
        train_start = time.time()
        
        # Train the model
        trainer.train()
        
        train_time = time.time() - train_start
        print(f"\n✅ Training completed in {format_time(train_time)}")
        
        # Save the model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        print(f"💾 Model saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Evaluation on test set
    print(f"\n📊 Evaluating on test set...")
    try:
        eval_start = time.time()
        
        # Evaluate
        eval_results = trainer.evaluate(eval_dataset=test_loader.dataset)
        
        eval_time = time.time() - eval_start
        print(f"⏱️ Evaluation completed in {format_time(eval_time)}")
        
        # Print metrics
        print_metrics(eval_results, "Test")
        
        # Get predictions for error analysis
        predictions = trainer.predict(test_loader.dataset)
        y_pred = predictions.predictions.argmax(-1)
        y_true = predictions.label_ids
        
        # Get texts for error analysis
        test_texts = [item['text'] for item in test_loader.dataset.data]
        
        # Print confusion matrix
        print_confusion_matrix(y_true, y_pred)
        
        # Error analysis (just a few examples for demo)
        error_analysis(y_pred, y_true, test_texts, n_examples=5)
        
        # Save predictions
        from src.utils import save_predictions
        pred_file = os.path.join(output_dir, "demo_predictions.jsonl")
        save_predictions(y_pred, y_true, test_texts, pred_file)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n🎉 DEMO TRAINING COMPLETED!")
    print(f"⏱️ Total time: {format_time(total_time)}")
    print(f"📊 Final F1 Score: {eval_results['eval_f1']:.4f}")
    print(f"📊 Final Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"💾 Model saved to: {output_dir}")
    print(f"\n💡 This was a demo with {len(train_loader.dataset)} samples.")
    print(f"   For full training with {args.demo_size*300}+ samples, use train_full.py")

if __name__ == "__main__":
    main() 