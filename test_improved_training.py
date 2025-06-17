#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Improved Training - Demo các strategies cân bằng dữ liệu
"""

import sys
import os
sys.path.append('src')

from improved_training import train_improved_model
import argparse

def main():
    parser = argparse.ArgumentParser(description="Test Improved Training Strategies")
    parser.add_argument("--strategy", type=str, default="weighted_loss", 
                       choices=["weighted_loss", "oversample", "undersample", "focal_loss"],
                       help="Data balancing strategy")
    parser.add_argument("--ratio", type=float, default=0.35, 
                       help="Target clickbait ratio")
    parser.add_argument("--features", action="store_true", 
                       help="Use feature fusion")
    parser.add_argument("--config", type=str, default="tinybert_demo",
                       help="Model configuration")
    
    args = parser.parse_args()
    
    print(f"🧪 Testing Strategy: {args.strategy}")
    print(f"📊 Target Ratio: {args.ratio}")
    print(f"🔧 Feature Fusion: {args.features}")
    
    output_dir = f"outputs/test_{args.strategy}"
    
    try:
        trainer, results = train_improved_model(
            config_name=args.config,
            use_balancing=args.strategy,
            target_ratio=args.ratio,
            use_feature_fusion=args.features,
            output_dir=output_dir
        )
        
        print(f"\n✅ Test completed successfully!")
        print(f"📊 Results:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 