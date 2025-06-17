#!/usr/bin/env python3
"""
Quick Start Guide for RTX A5000 Clickbait Classification Training
Interactive script to guide users through the training process
"""

import subprocess
import sys
import os
from pathlib import Path


def print_header():
    """Print welcome header"""
    print("🎯" + "="*78 + "🎯")
    print("   CLICKBAIT CLASSIFICATION - RTX A5000 TRAINING SUITE")
    print("🎯" + "="*78 + "🎯")
    print()
    print("This interactive guide will help you train state-of-the-art models")
    print("for clickbait classification on your RTX A5000 GPU.")
    print()


def check_prerequisites():
    """Check if all prerequisites are met"""
    print("🔧 CHECKING PREREQUISITES")
    print("-" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 8:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor} (need 3.8+)")
        return False
    
    # Check data files
    data_files = [
        "data/train/data.jsonl",
        "data/val/data.jsonl",
        "data/test/data.jsonl"
    ]
    
    for file_path in data_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            return False
    
    # Check if in correct directory
    if not Path("scripts").exists():
        print("❌ Please run this script from the project root directory")
        return False
    
    print("✅ All prerequisites met!")
    return True


def show_model_options():
    """Show available model options"""
    print("\n📋 AVAILABLE MODELS FOR RTX A5000 (24GB)")
    print("-" * 50)
    
    models = [
        {
            "category": "🔥 BERT Family (Recommended for beginners)",
            "models": [
                ("1", "BERT-base-uncased", "110M params, ~45min, F1≈0.70"),
                ("2", "DeBERTa-v3-base", "184M params, ~60min, F1≈0.72"),
                ("3", "BERT-large-uncased", "340M params, ~90min, F1≈0.73"),
            ]
        },
        {
            "category": "🚀 Large Language Models with LoRA",
            "models": [
                ("4", "Mistral-7B-v0.2", "7B params, ~120min, F1≈0.72"),
                ("5", "Mistral-7B-Instruct", "7B params, ~90min, F1≈0.73"),
                ("6", "Llama-2-7B", "7B params, ~120min, F1≈0.70"),
                ("7", "Llama-3-8B", "8B params, ~150min, F1≈0.74"),
                ("8", "Llama-2-13B", "13B params, ~180min, F1≈0.75"),
            ]
        }
    ]
    
    for category_info in models:
        print(f"\n{category_info['category']}")
        for option, name, description in category_info['models']:
            print(f"   {option}. {name:<20} | {description}")
    
    print(f"\n   9. All BERT family models")
    print(f"   10. All LLM models")
    print(f"   11. ALL models (full benchmark)")


def get_user_choice():
    """Get user's model choice"""
    print("\n🎯 SELECT TRAINING OPTION")
    print("-" * 30)
    
    while True:
        choice = input("Enter your choice (1-11): ").strip()
        
        if choice in [str(i) for i in range(1, 12)]:
            return int(choice)
        else:
            print("❌ Invalid choice. Please enter a number between 1-11.")


def run_training(choice):
    """Run the selected training option"""
    
    commands = {
        1: "python scripts/train_bert_family.py --model bert-base-uncased",
        2: "python scripts/train_bert_family.py --model deberta-v3-base", 
        3: "python scripts/train_bert_family.py --model bert-large-uncased",
        4: "python scripts/train_llm_lora.py --model mistral-7b-v0.2",
        5: "python scripts/train_llm_lora.py --model mistral-7b-instruct",
        6: "python scripts/train_llm_lora.py --model llama2-7b",
        7: "python scripts/train_llm_lora.py --model llama3-8b",
        8: "python scripts/train_llm_lora.py --model llama2-13b",
        9: "python scripts/train_bert_family.py --model all",
        10: "python scripts/train_llm_lora.py --model all",
        11: "python scripts/run_all_experiments.py"
    }
    
    model_names = {
        1: "BERT-base-uncased",
        2: "DeBERTa-v3-base",
        3: "BERT-large-uncased", 
        4: "Mistral-7B-v0.2",
        5: "Mistral-7B-Instruct",
        6: "Llama-2-7B",
        7: "Llama-3-8B",
        8: "Llama-2-13B",
        9: "All BERT family models",
        10: "All LLM models",
        11: "All models (full benchmark)"
    }
    
    print(f"\n🚀 STARTING TRAINING: {model_names[choice]}")
    print("=" * 60)
    
    command = commands[choice]
    print(f"Command: {command}")
    print()
    
    # Confirm before starting
    confirm = input("Proceed with training? (y/n): ").lower()
    if confirm != 'y':
        print("❌ Training cancelled.")
        return
    
    print("🏋️ Training started! This may take a while...")
    print("💡 Tip: You can monitor GPU usage with 'nvidia-smi' in another terminal")
    print()
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        print("\n✅ Training completed successfully!")
        
        # Offer to run benchmark
        if choice in [9, 10, 11]:  # Multiple models trained
            benchmark = input("\nGenerate benchmark comparison? (y/n): ").lower()
            if benchmark == 'y':
                print("📊 Generating benchmark results...")
                subprocess.run("python scripts/benchmark_results.py --save_csv", shell=True)
                print("✅ Benchmark results generated!")
        
    except subprocess.CalledProcessError:
        print("\n❌ Training failed! Check the error messages above.")
        return
    
    print(f"\n📁 Results saved to: outputs/")
    print("🎉 Training complete! Check the outputs directory for your trained models.")


def main():
    """Main function"""
    print_header()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above and try again.")
        return
    
    # Show model options
    show_model_options()
    
    # Get user choice
    choice = get_user_choice()
    
    # Run training
    run_training(choice)
    
    print("\n🎯 Quick Start Guide completed!")
    print("📚 For more advanced options, check docs/FINE_TUNING_GUIDE.md")


if __name__ == "__main__":
    main() 