#!/usr/bin/env python3
"""
Fine-tune Large Language Models with LoRA/QLoRA and RS-LoRA
Optimized for RTX A6000 (48GB VRAM) - Maximum Performance Configuration

Supports:
- Mistral-7B-Instruct: Full precision, large batches, high LoRA rank
- Llama-3.1-8B-Instruct: Full precision with RS-LoRA
- Llama-3.1-8B-Instruct-Quantized: 4-bit quantization for extreme batch sizes

RTX A6000 Optimizations:
- No quantization for maximum quality (where VRAM permits)
- Large batch sizes (32-48) to utilize 48GB VRAM
- Extended sequence lengths (512-768 tokens)
- High LoRA rank (128-256) for maximum parameter efficiency
- RS-LoRA support for enhanced performance
- Cosine learning rate schedule with proper warmup
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
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Model configurations optimized for RTX A6000 (48GB VRAM) - MAXIMUM PERFORMANCE
LLM_CONFIGS = {
    "mistral-7b-instruct": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "batch_size": 32,  # Large batch to utilize 48GB VRAM
        "learning_rate": 1e-4,  # Stable learning rate for quality
        "epochs": 5,  # More epochs for better convergence
        "max_length": 512,  # Longer sequences for better context
        "quantization": "none",  # No quantization to maximize performance
        "lora_r": 128,  # High rank for maximum parameter efficiency
        "lora_alpha": 256,  # 2x lora_r for optimal scaling
        "lora_dropout": 0.05,  # Lower dropout for more parameters
        "gradient_accumulation_steps": 1,  # Direct training with large batch
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "use_rslora": False  # Standard LoRA
    },
    "llama3-8b": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct", 
        "batch_size": 24,  # Slightly smaller for 8B model
        "learning_rate": 5e-5,  # Lower LR for larger model
        "epochs": 5,  # More epochs for better convergence
        "max_length": 512,  # Longer sequences for better context
        "quantization": "none",  # No quantization to maximize performance
        "lora_r": 128,  # High rank for maximum parameter efficiency
        "lora_alpha": 256,  # 2x lora_r for optimal scaling
        "lora_dropout": 0.05,  # Lower dropout for more parameters
        "gradient_accumulation_steps": 1,  # Direct training with large batch
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "use_rslora": True  # RS-LoRA for better performance
    },
    "llama3-8b-quantized": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "batch_size": 48,  # Even larger batch with quantization
        "learning_rate": 8e-5,  # Slightly higher for quantized training
        "epochs": 6,  # More epochs to compensate for quantization
        "max_length": 768,  # Very long sequences
        "quantization": "4bit",  # Use quantization for extreme batch sizes
        "lora_r": 256,  # Very high rank
        "lora_alpha": 512,  # 2x lora_r for optimal scaling
        "lora_dropout": 0.03,  # Very low dropout
        "gradient_accumulation_steps": 1,
        "weight_decay": 0.008,
        "warmup_ratio": 0.15,
        "lr_scheduler_type": "cosine",
        "use_rslora": True  # RS-LoRA for better performance
    }
}

torch.backends.cuda.matmul.allow_tf32 = True

def get_quantization_config(quantization_type):
    """Get quantization configuration"""
    if quantization_type == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization_type == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif quantization_type == "none" or quantization_type is None:
        return None
    else:
        return None


def setup_model_and_tokenizer(config):
    """Setup model and tokenizer with LoRA"""
    print(f"🤖 Loading {config['model_name']} with {config['quantization']} quantization...")
    
    # Quantization config
    quantization_config = get_quantization_config(config["quantization"])
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    
    # Set pad token if not exists - crucial for batch processing
    original_vocab_size = len(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"✅ Set pad_token to eos_token: {tokenizer.pad_token}")
    
    # Ensure padding side is correct for classification
    tokenizer.padding_side = "right"
    
    # Load model with quantization
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=2,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype= "auto"
    )

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Only resize embeddings if tokenizer vocab changed
    if len(tokenizer) != original_vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        print(f"✅ Resized embeddings from {original_vocab_size} to {len(tokenizer)}")
    else:
        print(f"✅ No need to resize embeddings (vocab size: {len(tokenizer)})")
    
    # LoRA configuration - Support for both standard LoRA and RS-LoRA
    lora_kwargs = {
        "task_type": TaskType.SEQ_CLS,
        "r": config["lora_r"],
        "lora_alpha": config["lora_alpha"],
        "lora_dropout": config.get("lora_dropout", 0.05),
        "target_modules": [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        "bias": "none"
    }
    
    # Add RS-LoRA support if specified
    if config.get("use_rslora", False):
        lora_kwargs["use_rslora"] = True
        print("✅ Using RS-LoRA configuration for enhanced performance")
    else:
        print("✅ Using standard LoRA configuration")
    
    lora_config = LoraConfig(**lora_kwargs)
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def preprocess(example, tokenizer, max_length=128):
    """Preprocess example for classification task - OPTIMIZED FOR SPEED"""
    # Clean up text - remove [SEP] tokens and extra whitespace
    text = example["text"]
    if "[SEP]" in text:
        # Take the first part before [SEP] as main text
        text = text.split("[SEP]")[0].strip()
    
    # Basic text cleaning and early truncation for speed
    text = " ".join(text.split())  # Normalize whitespace
    
    # Truncate text early to save tokenization time
    words = text.split()
    if len(words) > max_length // 2:  # Rough estimation
        text = " ".join(words[:max_length // 2])
    
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    enc["labels"] = int(example["label"])
    return enc

def prepare_dataset(tokenizer, max_length=128, quick_train=False):
    """Prepare all dataset splits for training - with QUICK TRAIN option"""
    file_map = {
        "train": "shared/data/train/data.jsonl",
        "validation": "shared/data/val/data.jsonl", 
        "test": "shared/data/test/data.jsonl"
    }
    
    datasets = {}
    for split, file_path in file_map.items():
        print(f"Loading {split} dataset from {file_path}...")
        data = load_jsonl(file_path)
        
        # Quick train mode: use only subset for faster training
        if quick_train:
            if split == "train":
                data = data[:5000]  # Use only 5K samples for training
            elif split == "validation":
                data = data[:500]   # Use only 500 samples for validation
            elif split == "test":
                data = data[:500]   # Use only 500 samples for test
            print(f"🚀 QUICK TRAIN MODE: Using {len(data)} samples")
        
        ds = Dataset.from_list(data)
        ds = ds.map(lambda ex: preprocess(ex, tokenizer, max_length))
        datasets[split] = ds
        print(f"✅ Loaded {len(ds)} examples for {split}")
    
    return datasets


def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "f1_macro": f1_score(labels, preds, average='macro'),
        "f1_binary": f1_score(labels, preds, average='binary')
    }


def train_model(model_key, config, output_base_dir="outputs", quick_train=False):
    """Train a single model with LoRA - SPEED OPTIMIZED"""
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING {model_key.upper()} WITH LORA {'(QUICK MODE)' if quick_train else ''}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Setup paths
    output_dir = f"{output_base_dir}/{model_key}-lora-cuda"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(config)
        
        # Prepare dataset
        tokenized_dataset = prepare_dataset(tokenizer, config["max_length"], quick_train)
        
        # Training arguments - OPTIMIZED FOR RTX A6000 (48GB VRAM)
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch", 
            logging_steps=25,  # Balanced logging frequency
            learning_rate=config["learning_rate"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            num_train_epochs=config["epochs"],
            weight_decay=config.get("weight_decay", 0.01),
            warmup_ratio=config.get("warmup_ratio", 0.1),
            lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            bf16=True,  # Use bfloat16 for A6000 compatibility
            dataloader_num_workers=8,  # More workers for large batches
            gradient_checkpointing=True,
            report_to=None,  # Disable wandb by default 
            save_total_limit=3,  # Keep more checkpoints for quality
            dataloader_pin_memory=True,
            optim="adamw_torch",  # Standard AdamW for maximum performance
            max_grad_norm=1.0,
            logging_dir=f"{output_dir}/logs",
            disable_tqdm=False,
            dataloader_persistent_workers=True,
            # Additional A6000 optimizations
            fp16_full_eval=False,  # Use bf16 for eval too
            ddp_find_unused_parameters=False,  # For multi-GPU efficiency
            dataloader_drop_last=True,  # Consistent batch sizes
            group_by_length=True,  # Optimize sequence lengths
            length_column_name="length",  # For sequence length grouping
            prediction_loss_only=False,
            # Memory and performance optimizations
            torch_compile=False,  # Stable training
            seed=42  # Reproducible results
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # More patience for quality training
        )
        
        # Train
        print("🏋️ Starting training...")
        train_result = trainer.train()
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Try to merge and save merged model
        try:
            print("🔗 Merging LoRA weights...")
            merged_model = model.merge_and_unload()
            merged_dir = f"{output_dir}_merged"
            merged_model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            print(f"✅ Merged model saved to: {merged_dir}")
        except Exception as e:
            print(f"⚠️ Could not merge LoRA weights: {e}")
        
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
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise e


def main():
    parser = argparse.ArgumentParser(description="Train LLMs with LoRA - OPTIMIZED FOR RTX A6000")
    parser.add_argument("--model", 
                       choices=["mistral-7b-instruct", "llama3-8b", "llama3-8b-quantized", "all"], 
                       default="all", 
                       help="Model to train: mistral-7b-instruct, llama3-8b (no quant), llama3-8b-quantized (max batch), or all")
    parser.add_argument("--output_dir", default="fine-tuning/outputs", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick training with subset of data")
    
    args = parser.parse_args()
    
    # Check GPU configuration
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"🔧 CUDA GPUs detected: {gpu_count}")
        
        total_memory = 0
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            total_memory += gpu_memory
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        print(f"   Total GPU Memory: {total_memory:.1f} GB")
        
        # RTX A6000 specific optimizations
        if "A6000" in gpu_name:
            print("✅ RTX A6000 detected! Using optimized configuration for maximum performance")
            print("   • No quantization for full precision")
            print("   • Large batch sizes to utilize 48GB VRAM") 
            print("   • Extended sequences for better context")
        elif gpu_memory >= 40:
            print("✅ High-end GPU detected! Using performance configuration")
        elif gpu_memory < 16:
            print("⚠️  Warning: GPU has limited VRAM. Consider using quantized models")
            print("   Recommended: Use 'llama3-8b-quantized' configuration")
    else:
        print("❌ No CUDA GPU detected!")
        return
    
    # Train models
    all_results = {}
    
    if args.model == "all":
        models_to_train = list(LLM_CONFIGS.keys())
    else:
        models_to_train = [args.model]
    
    print(f"🎯 Training {len(models_to_train)} model(s): {', '.join(models_to_train)}")
    
    for model_key in models_to_train:
        try:
            config = LLM_CONFIGS[model_key]
            results = train_model(model_key, config, args.output_dir, args.quick)
            all_results[model_key] = results
            
            # Clean up GPU memory
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Error training {model_key}: {e}")
            continue
    
    # Save summary results
    summary_file = f"{args.output_dir}/llm_lora_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("📋 LLM LORA TRAINING SUMMARY")
    print(f"{'='*80}")
    
    for model_key, results in all_results.items():
        metrics = results["test_metrics"]
        print(f"{model_key:20} | F1: {metrics['f1_weighted']:.4f} | "
              f"Acc: {metrics['accuracy']:.4f} | "
              f"Time: {results['training_time_formatted']}")
    
    print(f"\n✅ All results saved to: {summary_file}")


if __name__ == "__main__":
    main() 