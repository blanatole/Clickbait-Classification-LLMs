#!/usr/bin/env python3
"""
Setup and check environment for clickbait classification fine-tuning
Following the step-by-step guide for Webis-Clickbait-17 dataset
"""

import torch
import sys
import subprocess
import pkg_resources
import os
from pathlib import Path


def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("⚠️  Warning: Python 3.8+ is recommended")
    else:
        print("✅ Python version is compatible")
    print()


def check_gpu():
    """Check GPU availability and specs"""
    print("🔧 Checking GPU availability...")
    
    if torch.cuda.is_available():
        print("✅ CUDA is available!")
        
        device_count = torch.cuda.device_count()
        print(f"Number of GPUs: {device_count}")
        
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            print(f"GPU {i}: {gpu_name}")
            print(f"Memory: {gpu_memory:.1f} GB")
            
            # Memory recommendations
            if gpu_memory >= 48:
                print("✅ Excellent! Can handle large models with LoRA (13B+)")
            elif gpu_memory >= 24:
                print("✅ Good! Can handle LoRA training (7B models)")
            elif gpu_memory >= 16:
                print("✅ OK! Can handle BERT/DeBERTa fine-tuning")
            elif gpu_memory >= 8:
                print("✅ OK! Can handle BERT fine-tuning")
            else:
                print("⚠️  Limited memory. Consider smaller batch sizes or models")
        
        # Test GPU
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            torch.cuda.synchronize()
            print("✅ GPU test successful")
        except Exception as e:
            print(f"❌ GPU test failed: {e}")
    
    else:
        print("❌ CUDA is not available. Training will use CPU (very slow)")
        print("Consider:")
        print("- Installing PyTorch with CUDA support")
        print("- Using Google Colab or cloud instances with GPU")
    
    print()


def check_required_packages():
    """Check if required packages are installed"""
    print("📦 Checking required packages...")
    
    required_packages = [
        "transformers",
        "datasets", 
        "peft",
        "accelerate",
        "scikit-learn",
        "tqdm",
        "torch"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            pkg_resources.get_distribution(package)
            print(f"✅ {package}")
        except pkg_resources.DistributionNotFound:
            print(f"❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
    else:
        print("\n✅ All required packages are installed!")
    
    print()


def check_data_structure():
    """Check if data structure is correct"""
    print("📁 Checking data structure...")
    
    expected_files = [
                "shared/data/train/data.jsonl",
        "shared/data/val/data.jsonl",
        "shared/data/test/data.jsonl"
    ]
    
    all_exist = True
    for file_path in expected_files:
        if os.path.exists(file_path):
            # Count lines
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            print(f"✅ {file_path} ({line_count:,} samples)")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_exist = False
    
    if all_exist:
        print("✅ Data structure is correct!")
    else:
        print("⚠️  Some data files are missing!")
    
    print()


def create_output_dirs():
    """Create output directories"""
    print("📂 Creating output directories...")
    
    dirs = ["fine-tuning/outputs", "fine-tuning/outputs/checkpoints", "fine-tuning/outputs/logs", 
            "prompting/outputs", "prompting/results"]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ {dir_path}")
    
    print()


def print_training_commands():
    """Print example training commands"""
    print("🚀 Ready to start training! Example commands:")
    print("="*60)
    print()
    
    print("1️⃣  Train BERT family models:")
    print("   python fine-tuning/scripts/train_bert_family.py --model all")
    print()
    
    print("2️⃣  Train LLM with LoRA:")
    print("   python fine-tuning/scripts/train_llm_lora.py --model mistral-7b-instruct")
    print("   python fine-tuning/scripts/train_llm_lora.py --model llama3-8b")
    print()
    
    print("3️⃣  Train individual models:")
    print("   python fine-tuning/scripts/train_bert_family.py --model bert-base-uncased")
    print("   python fine-tuning/scripts/train_bert_family.py --model bert-large-uncased")
    print()
    
    print("4️⃣  Prompting approaches:")
    print("   python prompting/scripts/prompting_example.py")
    print("   python prompting/scripts/improved_prompting.py")
    print()
    
    print("5️⃣  Run full benchmark suite:")
    print("   python shared/scripts/run_all_experiments.py")
    print()
    
    print("6️⃣  Generate comparison results:")
    print("   python shared/scripts/benchmark_results.py --save_csv")
    print()
    
    print("💡 Expected results:")
    print("   - BERT-large: F1 ≈ 0.73, Accuracy ≈ 87%")
    print("   - Mistral-7B-Instruct: F1 ≈ 0.74, Accuracy ≈ 87%")
    print("   - Llama-3.1-8B-Instruct: F1 ≈ 0.74, Accuracy ≈ 87%")
    print("   - Training time: 45-150 minutes (with GPU)")
    print()


def main():
    print("🔍 CLICKBAIT CLASSIFICATION ENVIRONMENT SETUP")
    print("="*60)
    print()
    
    check_python_version()
    check_gpu()
    check_required_packages()
    check_data_structure()
    create_output_dirs()
    
    print("="*60)
    print("📋 ENVIRONMENT CHECK COMPLETE")
    print("="*60)
    print()
    
    print_training_commands()


if __name__ == "__main__":
    main() 