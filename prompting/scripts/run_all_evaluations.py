#!/usr/bin/env python3
"""
Run all prompting methods evaluation and comparison
Compare Zero-shot, Few-shot, and Chain-of-Thought prompting methods
"""

import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Import functions from individual scripts
from zero_shot_prompting import (
    zero_shot_prompting,
    load_evaluation_data,
    initialize_llm
)
from few_shot_prompting import few_shot_prompting
from chain_of_thought_prompting import chain_of_thought_prompting

load_dotenv()

def calculate_all_metrics(predictions, true_labels, method_name):
    """Calculate all evaluation metrics"""
    # Filter valid predictions
    valid_indices = [i for i, pred in enumerate(predictions) if pred != -1]
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_true_labels = [true_labels[i] for i in valid_indices]
    
    if len(valid_predictions) == 0:
        print(f"❌ No valid predictions for method {method_name}")
        return None
    
    # Calculate all metrics
    accuracy = accuracy_score(valid_true_labels, valid_predictions)
    
    # Macro averages
    precision_macro = precision_score(valid_true_labels, valid_predictions, average='macro', zero_division=0)
    recall_macro = recall_score(valid_true_labels, valid_predictions, average='macro', zero_division=0)
    f1_macro = f1_score(valid_true_labels, valid_predictions, average='macro', zero_division=0)
    
    # Weighted averages
    precision_weighted = precision_score(valid_true_labels, valid_predictions, average='weighted', zero_division=0)
    recall_weighted = recall_score(valid_true_labels, valid_predictions, average='weighted', zero_division=0)
    f1_weighted = f1_score(valid_true_labels, valid_predictions, average='weighted', zero_division=0)
    
    # Detailed report
    report = classification_report(valid_true_labels, valid_predictions, target_names=['No-clickbait', 'Clickbait'])
    
    print(f"\n📊 RESULTS FOR METHOD: {method_name.upper()}")
    print("=" * 60)
    print(f"Valid samples: {len(valid_predictions)}/{len(predictions)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nMacro Averages:")
    print(f"  Precision (macro): {precision_macro:.4f}")
    print(f"  Recall (macro): {recall_macro:.4f}")
    print(f"  F1-Score (macro): {f1_macro:.4f}")
    print(f"\nWeighted Averages:")
    print(f"  Precision (weighted): {precision_weighted:.4f}")
    print(f"  Recall (weighted): {recall_weighted:.4f}")
    print(f"  F1-Score (weighted): {f1_weighted:.4f}")
    print("\nDetailed Classification Report:")
    print(report)
    
    return {
        'method': method_name,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'valid_samples': len(valid_predictions),
        'total_samples': len(predictions)
    }

def evaluate_method(client, data, method_func, method_name, delay=1.0):
    """Evaluate a specific prompting method"""
    predictions = []
    true_labels = []
    
    print(f"\n{'='*60}")
    print(f"🧪 EVALUATING: {method_name.upper()}")
    print(f"{'='*60}")
    
    for i, item in enumerate(data):
        title = item['title']
        true_label = item['label']
        
        print(f"[{i+1}/{len(data)}] {title[:50]}...")
        
        prediction = method_func(client, title)
        predictions.append(prediction)
        true_labels.append(true_label)
        
        if prediction != -1:
            label_map = {0: "No-clickbait", 1: "Clickbait"}
            label_name = label_map.get(prediction, "Unknown")
            print(f"  → {prediction} - {label_name}")
        else:
            print(f"  → Response parsing error")
        
        time.sleep(delay)
    
    return predictions, true_labels

def run_comprehensive_evaluation(data_limit=30):
    """Run evaluation of 3 main prompting methods"""
    print("🚀 COMPREHENSIVE PROMPTING METHODS EVALUATION")
    print("=" * 70)
    
    # Check environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: Please set OPENAI_API_KEY in .env file")
        print("   Get your API key from: https://platform.openai.com/api-keys")
        return
    
    # Initialize OpenAI client
    print("🚀 Initializing OpenAI client...")
    try:
        client = initialize_llm()
        print("✅ Initialization successful!\n")
    except Exception as e:
        print(f"❌ OpenAI client initialization error: {e}")
        return
    
    # Load test data
    data_path = "../../shared/data/test/data.jsonl"
    test_data = load_evaluation_data(data_path, limit=data_limit)
    
    if not test_data:
        print("❌ Unable to load test data")
        return
    
    print(f"✅ Loaded {len(test_data)} test samples\n")
    
    # Define 3 main methods to evaluate
    methods = {
        "zero_shot": {
            "func": zero_shot_prompting,
            "delay": 1.0,
            "description": "Zero-shot prompting with expert guidelines"
        },
        "few_shot": {
            "func": few_shot_prompting, 
            "delay": 1.0,
            "description": "Few-shot prompting with curated examples"
        },
        "chain_of_thought": {
            "func": chain_of_thought_prompting,
            "delay": 1.5,
            "description": "Chain-of-thought with step-by-step analysis"
        }
    }
    
    all_results = []
    
    for method_name, method_info in methods.items():
        try:
            print(f"\n{'='*70}")
            print(f"🔄 Starting {method_name} evaluation...")
            print(f"📝 Method: {method_info['description']}")
            
            predictions, true_labels = evaluate_method(
                client, test_data, method_info['func'], method_name, 
                delay=method_info['delay']
            )
            
            metrics = calculate_all_metrics(predictions, true_labels, method_name)
            
            if metrics:
                all_results.append(metrics)
                print(f"✅ {method_name} completed successfully!")
            else:
                print(f"⚠️ {method_name} failed to generate valid metrics")
                
        except KeyboardInterrupt:
            print("\n⚠️ Evaluation stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in {method_name} evaluation: {e}")
            continue
    
    # Final comparison and analysis
    if all_results:
        print(f"\n{'='*70}")
        print("🏆 FINAL COMPARISON RESULTS")
        print(f"{'='*70}")
        
        df = pd.DataFrame(all_results)
        
        # Sort by F1-macro for display
        df_sorted = df.sort_values('f1_macro', ascending=False)
        print(df_sorted.to_string(index=False, float_format='%.4f'))
        
        # Find best methods
        best_accuracy = df.loc[df['accuracy'].idxmax()]
        best_f1_macro = df.loc[df['f1_macro'].idxmax()]
        best_f1_weighted = df.loc[df['f1_weighted'].idxmax()]
        
        print(f"\n🥇 BEST PERFORMING METHODS:")
        print(f"Best Accuracy: {best_accuracy['method']} ({best_accuracy['accuracy']:.4f})")
        print(f"Best F1-Macro: {best_f1_macro['method']} ({best_f1_macro['f1_macro']:.4f})")
        print(f"Best F1-Weighted: {best_f1_weighted['method']} ({best_f1_weighted['f1_weighted']:.4f})")
        
        # Performance analysis
        print(f"\n📈 PERFORMANCE ANALYSIS:")
        for _, row in df_sorted.iterrows():
            method = row['method']
            valid_rate = row['valid_samples'] / row['total_samples'] * 100
            print(f"{method.upper()}: Accuracy={row['accuracy']:.3f}, F1-Macro={row['f1_macro']:.3f}, Valid Rate={valid_rate:.1f}%")
        
        # Save results
        output_file = "prompting/outputs/comprehensive_evaluation_results.csv"
        os.makedirs("prompting/outputs", exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\n📄 Results saved to: {output_file}")
        
        return df
    else:
        print("❌ No valid results to compare")
        return None

def quick_demo():
    """Quick demo of all 3 methods"""
    print("🎯 QUICK DEMO - ALL 3 METHODS")
    print("=" * 50)
    
    # Check environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: Please set OPENAI_API_KEY in .env file")
        return
    
    try:
        client = initialize_llm()
        print("✅ OpenAI client initialized\n")
    except Exception as e:
        print(f"❌ OpenAI client error: {e}")
        return
    
    test_headlines = [
        "Federal Reserve raises interest rates by 0.25%",
        "You won't BELIEVE what this celebrity did next!",
        "Scientists discover new exoplanet in habitable zone"
    ]
    
    methods = {
        "Zero-shot": zero_shot_prompting,
        "Few-shot": few_shot_prompting,
        "Chain-of-thought": chain_of_thought_prompting
    }
    
    for headline in test_headlines:
        print(f"\n📰 Headline: {headline}")
        print("-" * 50)
        
        for method_name, method_func in methods.items():
            try:
                result = method_func(client, headline)
                label = "Clickbait" if result == 1 else "Not Clickbait" if result == 0 else "Error"
                print(f"{method_name:15}: {result} ({label})")
                time.sleep(0.5)
            except Exception as e:
                print(f"{method_name:15}: Error - {e}")

def main():
    """Main function with options"""
    print("🎯 PROMPTING METHODS EVALUATION SUITE (OpenAI)")
    print("=" * 60)
    
    print("\nOptions:")
    print("1. Quick demo (3 headlines, all methods)")
    print("2. Comprehensive evaluation (default: 30 samples)")
    print("3. Custom evaluation (specify sample size)")
    
    choice = input("\nEnter your choice (1/2/3) or press Enter for demo: ").strip()
    
    if choice == "2":
        print("\n🚀 Running comprehensive evaluation...")
        run_comprehensive_evaluation(data_limit=30)
    elif choice == "3":
        try:
            limit = int(input("Enter number of samples to evaluate: "))
            print(f"\n🚀 Running evaluation with {limit} samples...")
            run_comprehensive_evaluation(data_limit=limit)
        except ValueError:
            print("❌ Invalid number. Running with default 30 samples...")
            run_comprehensive_evaluation(data_limit=30)
    else:
        print("\n🚀 Running quick demo...")
        quick_demo()

if __name__ == "__main__":
    main() 