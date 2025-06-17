#!/usr/bin/env python3
"""
Fine-tune Large Language Models with LoRA/QLoRA
Optimized for RTX A5000 (24 GB VRAM) - supports Mistral, Llama models
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
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np


# Model configurations optimized for RTX A5000 (24GB) with QLoRA
LLM_CONFIGS = {
    "mistral-7b-v0.3": {
        "model_name": "mistralai/Mistral-7B-v0.3",
        "batch_size": 10,
        "learning_rate": 5e-6,
        "epochs": 3,
        "max_length": 256,
        "quantization": "4bit",
        "lora_r": 8,
        "lora_alpha": 16,
        "gradient_accumulation_steps": 6  # Effective batch = 60
    },
    "mistral-7b-instruct": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "batch_size": 10,
        "learning_rate": 3e-6,  # Lower for instruct model
        "epochs": 2,  # Fewer epochs for instruct
        "max_length": 256,
        "quantization": "4bit",
        "lora_r": 8,
        "lora_alpha": 16,
        "gradient_accumulation_steps": 6
    },
    "llama2-7b": {
        "model_name": "meta-llama/Llama-2-7b-hf",
        "batch_size": 10,
        "learning_rate": 5e-6,
        "epochs": 3,
        "max_length": 256,
        "quantization": "4bit",
        "lora_r": 8,
        "lora_alpha": 16,
        "gradient_accumulation_steps": 6
    },
    "llama3-8b": {
        "model_name": "meta-llama/Meta-Llama-3-8B",
        "batch_size": 8,  # Slightly larger model
        "learning_rate": 4e-6,
        "epochs": 3,
        "max_length": 256,
        "quantization": "4bit",
        "lora_r": 8,
        "lora_alpha": 16,
        "gradient_accumulation_steps": 8  # Effective batch = 64
    },
    "llama2-13b": {
        "model_name": "meta-llama/Llama-2-13b-hf",
        "batch_size": 5,  # Much larger model
        "learning_rate": 3e-6,
        "epochs": 2,
        "max_length": 256,
        "quantization": "8bit",  # 8-bit for 13B
        "lora_r": 8,
        "lora_alpha": 16,
        "gradient_accumulation_steps": 12  # Effective batch = 60
    }
}


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
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer


def prepare_dataset(tokenizer, max_length=256):
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
        # Ensure padding token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",  # Always pad to max_length
            return_tensors=None  # Don't return tensors here
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
    """Train a single model with LoRA"""
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING {model_key.upper()} WITH LORA")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Setup paths
    output_dir = f"{output_base_dir}/{model_key}-lora-a5000"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(config)
        
        # Prepare dataset
        tokenized_dataset = prepare_dataset(tokenizer, config["max_length"])
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=20,
            learning_rate=config["learning_rate"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            num_train_epochs=config["epochs"],
            weight_decay=0.01,
            warmup_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            bf16=True,  # Use bfloat16 for better numerical stability
            dataloader_num_workers=2,  # Lower for memory
            gradient_checkpointing=True,
            report_to=None,
            save_total_limit=1,  # Keep only best checkpoint
            dataloader_pin_memory=False,  # Save memory
            optim="adamw_torch"
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
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
    parser = argparse.ArgumentParser(description="Train LLMs with LoRA on RTX A5000")
    parser.add_argument("--model", choices=list(LLM_CONFIGS.keys()) + ["all"], 
                       default="mistral-7b-v0.2", help="Model to train")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🔧 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        if gpu_memory < 16:
            print("⚠️  Warning: LLM training requires at least 16GB VRAM")
            print("   Consider using smaller models or reduce batch sizes")
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
            results = train_model(model_key, config, args.output_dir)
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