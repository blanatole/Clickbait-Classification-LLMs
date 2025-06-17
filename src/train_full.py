#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Training Script cho A5000/A6000
Train models mạnh với full dataset để đạt performance cao nhất
"""

import os
import sys
import time
import torch
import argparse
import wandb
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.model_configs import get_config, get_full_configs, print_config_summary
from src.utils import (
    set_seed, get_device, compute_metrics, print_metrics, 
    check_gpu_compatibility, create_data_loaders, print_training_info,
    print_confusion_matrix, error_analysis, format_time, save_predictions
)

def main():
    parser = argparse.ArgumentParser(description="Full Training for A5000/A6000")
    parser.add_argument("--model", type=str, default="deberta_v3", 
                       choices=list(get_full_configs().keys()),
                       help="Model to train")
    parser.add_argument("--data_dir", type=str, default="data", 
                       help="Data directory")
    parser.add_argument("--output_dir", type=str, default="outputs", 
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="clickbait-classification",
                       help="W&B project name")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Resume training from checkpoint")
    parser.add_argument("--push_to_hub", action="store_true",
                       help="Push model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default=None,
                       help="Model ID for Hugging Face Hub")
    
    args = parser.parse_args()
    
    print("🚀 CLICKBAIT FULL TRAINING")
    print(f"🔥 Optimized for A5000/A6000 (24GB+ VRAM)")
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
    if not compatible:
        print("❌ Current GPU is not compatible with this model!")
        print("💡 Please use a model from demo configs or upgrade your GPU")
        return
    
    # Print training info
    print_training_info(config, demo_mode=False)
    
    # Setup paths
    train_path = os.path.join(args.data_dir, "train", "data.jsonl")
    val_path = os.path.join(args.data_dir, "val", "data.jsonl")
    test_path = os.path.join(args.data_dir, "test", "data.jsonl")
    
    output_dir = os.path.join(args.output_dir, f"{config.model_name}_full")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{config.model_name}_full",
            config=vars(config),
            tags=["full-training", config.model_name.split('_')[0]]
        )
        print("📊 Weights & Biases initialized")
    
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
        
        # Estimate memory usage
        from src.utils import estimate_gpu_memory
        estimated_memory = estimate_gpu_memory(
            config.batch_size_train, 
            config.max_length,
            getattr(model.config, 'hidden_size', 768),
            getattr(model.config, 'num_hidden_layers', 12)
        )
        print(f"   Estimated GPU memory: {estimated_memory:.1f}GB")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create data loaders (full mode)
    print(f"\n📊 Loading full datasets...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            train_path, val_path, test_path, tokenizer, config, demo_mode=False
        )
        
        print(f"📈 Dataset sizes:")
        print(f"   Train: {len(train_loader.dataset):,} samples")
        print(f"   Val:   {len(val_loader.dataset):,} samples")
        print(f"   Test:  {len(test_loader.dataset):,} samples")
        
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
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=False,
        report_to="wandb" if args.use_wandb else "none",
        run_name=f"{config.model_name}_full",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_strategy="every_save" if args.push_to_hub else "end",
        # Performance optimizations
        dataloader_pin_memory=True,
        group_by_length=True,  # Group similar length sequences
        tf32=True if torch.cuda.is_available() else False,  # Use TF32 on A100
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
    print(f"\n🚀 Starting full training...")
    print(f"   Full dataset: {len(train_loader.dataset):,} train, {len(val_loader.dataset):,} val")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Effective batch size: {config.batch_size_train * config.gradient_accumulation_steps}")
    print(f"   Total steps: ~{len(train_loader) * config.num_epochs // config.gradient_accumulation_steps}")
    
    try:
        train_start = time.time()
        
        # Train the model
        if args.resume_from_checkpoint:
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            trainer.train()
        
        train_time = time.time() - train_start
        print(f"\n✅ Training completed in {format_time(train_time)}")
        
        # Save the model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        print(f"💾 Model saved to: {output_dir}")
        
        # Log training time to wandb
        if args.use_wandb:
            wandb.log({"training_time_seconds": train_time})
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if args.use_wandb:
            wandb.finish(exit_code=1)
        return
    
    # Evaluation on test set
    print(f"\n📊 Final evaluation on test set...")
    try:
        eval_start = time.time()
        
        # Evaluate
        eval_results = trainer.evaluate(eval_dataset=test_loader.dataset)
        
        eval_time = time.time() - eval_start
        print(f"⏱️ Evaluation completed in {format_time(eval_time)}")
        
        # Print metrics
        print_metrics(eval_results, "Final Test")
        
        # Get predictions for comprehensive analysis
        print(f"\n🔍 Getting predictions for error analysis...")
        predictions = trainer.predict(test_loader.dataset)
        y_pred = predictions.predictions.argmax(-1)
        y_true = predictions.label_ids
        
        # Get texts for error analysis
        test_texts = [item['text'] for item in test_loader.dataset.data]
        
        # Print confusion matrix
        print_confusion_matrix(y_true, y_pred)
        
        # Comprehensive error analysis
        errors = error_analysis(y_pred, y_true, test_texts, n_examples=20)
        
        # Save detailed predictions
        pred_file = os.path.join(output_dir, "full_predictions.jsonl")
        save_predictions(y_pred, y_true, test_texts, pred_file)
        
        # Save error analysis
        error_file = os.path.join(output_dir, "error_analysis.jsonl")
        import json
        with open(error_file, 'w', encoding='utf-8') as f:
            for error in errors:
                f.write(json.dumps(error, ensure_ascii=False) + '\n')
        print(f"💾 Error analysis saved to: {error_file}")
        
        # Log final metrics to wandb
        if args.use_wandb:
            wandb.log({
                "final_test_accuracy": eval_results['eval_accuracy'],
                "final_test_f1": eval_results['eval_f1'],
                "final_test_precision": eval_results['eval_precision'],
                "final_test_recall": eval_results['eval_recall'],
                "eval_time_seconds": eval_time,
                "total_errors": len(errors),
                "error_rate": len(errors) / len(y_true)
            })
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        if args.use_wandb:
            wandb.finish(exit_code=1)
        return
    
    # Final summary
    total_time = time.time() - start_time
    
    print(f"\n🎉 FULL TRAINING COMPLETED!")
    print("=" * 60)
    print(f"⏱️ Total time: {format_time(total_time)}")
    print(f"📊 Final Results:")
    print(f"   • Accuracy:  {eval_results['eval_accuracy']:.4f}")
    print(f"   • F1-Score:  {eval_results['eval_f1']:.4f}")
    print(f"   • Precision: {eval_results['eval_precision']:.4f}")
    print(f"   • Recall:    {eval_results['eval_recall']:.4f}")
    print(f"💾 Model saved to: {output_dir}")
    print(f"📊 Predictions saved to: {pred_file}")
    
    if args.push_to_hub and args.hub_model_id:
        print(f"🤗 Model pushed to Hub: {args.hub_model_id}")
    
    # Performance comparison
    print(f"\n📈 PERFORMANCE COMPARISON:")
    if eval_results['eval_f1'] >= 0.75:
        print("🏆 Excellent performance (F1 ≥ 0.75)")
    elif eval_results['eval_f1'] >= 0.70:
        print("✅ Good performance (F1 ≥ 0.70)")
    elif eval_results['eval_f1'] >= 0.65:
        print("⚠️ Acceptable performance (F1 ≥ 0.65)")
    else:
        print("❌ Below expected performance (F1 < 0.65)")
        print("💡 Consider trying different hyperparameters or models")
    
    # Cleanup
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main() 