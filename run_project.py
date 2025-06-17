#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Project Runner cho Clickbait Classification
Chạy toàn bộ quy trình từ preprocessing đến demo
"""

import os
import sys
import argparse
import subprocess
from typing import List

def run_command(command: List[str], description: str = "") -> bool:
    """Run a command and return success status"""
    if description:
        print(f"\n🔄 {description}")
        print(f"Command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, shell=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_requirements():
    """Check if all requirements are installed"""
    print("🔍 Checking requirements...")
    
    try:
        import torch
        import transformers
        import datasets
        import sklearn
        print("✅ Core requirements installed")
        
        # Check GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("⚠️ No GPU available - will use CPU (slower)")
        
        return True
    except ImportError as e:
        print(f"❌ Missing requirement: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def setup_data():
    """Setup and preprocess data"""
    print("\n📊 Setting up data...")
    
    # Check if data exists
    data_paths = ["data/train/instances.jsonl", "data/val/instances.jsonl", "data/test/instances.jsonl"]
    missing_data = [path for path in data_paths if not os.path.exists(path)]
    
    if missing_data:
        print("❌ Data files missing:")
        for path in missing_data:
            print(f"   - {path}")
        print("Please ensure data is properly placed in data/ directory")
        return False
    
    # Run preprocessing
    return run_command(
        ["python", "src/data_preprocessor.py"],
        "Running data preprocessing..."
    )

def train_demo_model(model_name: str = "tinybert", quick: bool = True):
    """Train demo model"""
    print(f"\n🚀 Training demo model: {model_name}")
    
    command = ["python", "src/train_demo.py", "--model", model_name]
    if quick:
        command.append("--quick_test")
    
    return run_command(command, f"Training {model_name} demo model...")

def train_full_model(model_name: str = "deberta_v3", use_wandb: bool = False):
    """Train full model"""
    print(f"\n🔥 Training full model: {model_name}")
    
    command = ["python", "src/train_full.py", "--model", model_name]
    if use_wandb:
        command.append("--use_wandb")
    
    return run_command(command, f"Training {model_name} full model...")

def test_inference(model_path: str):
    """Test inference"""
    print(f"\n🎯 Testing inference with {model_path}")
    
    return run_command(
        ["python", "src/inference.py", "--model_path", model_path, "--test_examples"],
        "Testing inference with examples..."
    )

def launch_demo(model_path: str = "outputs/tinybert_rtx3050_demo"):
    """Launch demo UI"""
    print(f"\n🌐 Launching demo UI...")
    
    try:
        import gradio
        print(f"Demo will use model: {model_path}")
        print("Starting web interface...")
        subprocess.run(["python", "demo_ui.py"], check=True)
    except ImportError:
        print("❌ Gradio not installed. Install with: pip install gradio")
        return False
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
        return True

def main():
    parser = argparse.ArgumentParser(description="Clickbait Classification Project Runner")
    parser.add_argument("--mode", type=str, default="demo", 
                       choices=["setup", "demo", "full", "inference", "ui", "all"],
                       help="Mode to run")
    parser.add_argument("--model", type=str, default="tinybert",
                       help="Model to use for demo mode")
    parser.add_argument("--full_model", type=str, default="deberta_v3",
                       help="Model to use for full mode")
    parser.add_argument("--quick", action="store_true", default=True,
                       help="Use quick test mode for demo")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for full training")
    parser.add_argument("--model_path", type=str, default="outputs/tinybert_rtx3050_demo",
                       help="Model path for inference/demo")
    
    args = parser.parse_args()
    
    print("🎯 CLICKBAIT CLASSIFICATION PROJECT")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        return
    
    success = True
    
    if args.mode == "setup" or args.mode == "all":
        success &= setup_data()
    
    if args.mode == "demo" or args.mode == "all":
        if success:
            success &= train_demo_model(args.model, args.quick)
            if success:
                success &= test_inference(f"outputs/{args.model}_rtx3050_demo")
    
    if args.mode == "full":
        if success:
            success &= train_full_model(args.full_model, args.use_wandb)
            if success:
                success &= test_inference(f"outputs/{args.full_model}_full")
    
    if args.mode == "inference":
        success &= test_inference(args.model_path)
    
    if args.mode == "ui":
        launch_demo(args.model_path)
    
    if args.mode == "all" and success:
        print(f"\n🎉 All steps completed successfully!")
        print(f"\n💡 Next steps:")
        print(f"   1. Test model interactively")
        print(f"   python src/inference.py --model_path {args.model_path} --interactive")
        print(f"   2. Launch web demo")
        print(f"   python demo_ui.py")
        print(f"   3. Train with more epochs")
        print(f"   python src/train_demo.py --model {args.model}")
    
    # Final summary
    print(f"\n📋 PROJECT SUMMARY")
    print("=" * 40)
    print(f"Mode: {args.mode}")
    print(f"Status: {'✅ Success' if success else '❌ Failed'}")
    
    if os.path.exists("outputs"):
        print(f"\n📁 Available models:")
        for item in os.listdir("outputs"):
            if os.path.isdir(os.path.join("outputs", item)):
                print(f"   - {item}")
    
    if success and args.mode in ["demo", "all"]:
        print(f"\n🚀 Quick start commands:")
        print(f"   # Test model interactively")
        print(f"   python src/inference.py --model_path {args.model_path} --interactive")
        print(f"   ")
        print(f"   # Launch web demo")
        print(f"   python demo_ui.py")
        print(f"   ")
        print(f"   # Train with more epochs")
        print(f"   python src/train_demo.py --model {args.model}")

if __name__ == "__main__":
    main() 