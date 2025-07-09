#!/usr/bin/env python3
"""
Quick test training script with minimal data to verify configuration
"""

import os
import torch
from train_llm_lora import LLM_CONFIGS, train_model

def main():
    print("🧪 QUICK TRAINING TEST")
    print("="*50)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ No GPU available!")
        return
        
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"🔧 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Test with Mistral first (smaller)
    model_key = "mistral-7b-instruct"
    config = LLM_CONFIGS[model_key]
    
    # Override for very quick training
    config_test = config.copy()
    config_test["epochs"] = 1  # Only 1 epoch
    
    print(f"🚀 Testing {model_key} with 1 epoch...")
    
    try:
        # Run with quick=True to use subset of data
        results = train_model(model_key, config_test, "fine-tuning/outputs/test", quick_train=True)
        print("✅ Test training completed successfully!")
        print(f"📊 Results: Accuracy={results['test_metrics']['accuracy']:.4f}")
        
    except Exception as e:
        print(f"❌ Test training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 