#!/usr/bin/env python3
"""
Standalone evaluation script for English prompting methods
Methods: Zero-shot, Few-shot, Chain-of-Thought (CoT)
Run: python scripts/run_evaluation.py
"""

from prompting_example import (
    initialize_llm, run_evaluation_comparison, 
    load_evaluation_data, evaluate_prompting_method, calculate_metrics
)
import os
from dotenv import load_dotenv
import argparse

def main():
    # Load environment variables
    load_dotenv()
    
    # Create parser for command line arguments
    parser = argparse.ArgumentParser(description='Evaluate English prompting methods for clickbait classification')
    parser.add_argument('--limit', type=int, default=20, 
                       help='Number of test samples (default: 20)')
    parser.add_argument('--methods', nargs='+', 
                       choices=['zero_shot', 'few_shot', 'cot'], 
                       default=['zero_shot', 'few_shot', 'cot'],
                       help='Methods to evaluate')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between API calls in seconds')
    
    args = parser.parse_args()
    
    # Check environment variables
    if not os.environ.get("SHUBI_API_KEY") or not os.environ.get("SHUBI_URL"):
        print("❌ Error: Please set SHUBI_API_KEY and SHUBI_URL in .env file")
        print("Create .env file from sample.env template and fill in API details")
        return
    
    print("🚀 Initializing ChatOpenAI...")
    try:
        llm = initialize_llm()
        print("✅ Initialization successful!\n")
    except Exception as e:
        print(f"❌ LLM initialization error: {e}")
        return
    
    print("=" * 70)
    print("        🎯 ENGLISH PROMPTING METHODS EVALUATION")
    print("=" * 70)
    print(f"📊 Test samples: {args.limit}")
    print(f"🔧 Methods: {', '.join(args.methods)}")
    print(f"⏱️ Delay: {args.delay}s")
    print("=" * 70)
    
    # Load test data
    data_path = os.path.join("data", "test", "data_demo.jsonl")
    test_data = load_evaluation_data(data_path, limit=args.limit)
    
    if not test_data:
        print("❌ Unable to load test data")
        print("Check file: data/test/data_demo.jsonl")
        return
    
    print(f"✅ Loaded {len(test_data)} test samples\n")
    
    # Run evaluation for each method
    results = []
    
    for method in args.methods:
        print(f"\n{'='*50}")
        print(f"🧪 TESTING METHOD: {method.upper()}")
        print(f"{'='*50}")
        
        try:
            predictions, true_labels = evaluate_prompting_method(
                llm, test_data, method, delay=args.delay
            )
            metrics = calculate_metrics(predictions, true_labels, method)
            if metrics:
                results.append(metrics)
                
        except KeyboardInterrupt:
            print("\n⚠️ Stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in {method} evaluation: {e}")
    
    # Display comparison results
    if results:
        print("\n" + "=" * 70)
        print("           📈 FINAL EVALUATION COMPARISON")
        print("=" * 70)
        
        import pandas as pd
        df = pd.DataFrame(results)
        
        # Sort by F1-Score
        df = df.sort_values('f1_score', ascending=False)
        
        print(df.to_string(index=False, float_format='%.3f'))
        
        # Find best method
        best_method = df.iloc[0]
        print(f"\n🏆 BEST METHOD: {best_method['method']}")
        print(f"   📊 F1-Score: {best_method['f1_score']:.3f}")
        print(f"   🎯 Accuracy: {best_method['accuracy']:.3f}")
        print(f"   ✅ Valid samples: {best_method['valid_samples']}/{best_method['total_samples']}")
        
        # Save results
        output_file = "evaluation_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        # Method analysis
        print(f"\n📋 METHOD ANALYSIS:")
        for _, row in df.iterrows():
            success_rate = (row['valid_samples'] / row['total_samples']) * 100
            print(f"   {row['method']}: {row['f1_score']:.3f} F1 ({success_rate:.1f}% success rate)")
        
    else:
        print("❌ No valid results to compare")

if __name__ == "__main__":
    main() 