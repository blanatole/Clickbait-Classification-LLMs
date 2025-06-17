#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Script cho Clickbait Classification
Đánh giá model trên test set với metrics chi tiết
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import ClickbaitClassifier
from src.utils import set_seed, print_metrics, print_confusion_matrix

def evaluate_model(model_path: str, test_file: str, output_dir: str = None):
    """Evaluate model on test set"""
    
    print("📊 CLICKBAIT MODEL EVALUATION")
    print("=" * 50)
    
    # Load classifier
    try:
        classifier = ClickbaitClassifier(model_path)
        print(f"✅ Model loaded from: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load test data
    try:
        test_data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    test_data.append(json.loads(line))
        
        print(f"✅ Test data loaded: {len(test_data)} samples")
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return
    
    # Extract texts and labels
    texts = [item['text'] for item in test_data]
    true_labels = [item['label'] for item in test_data]
    
    print(f"📊 Test set info:")
    clickbait_count = sum(true_labels)
    print(f"   Total samples: {len(true_labels)}")
    print(f"   Clickbait: {clickbait_count} ({clickbait_count/len(true_labels)*100:.1f}%)")
    print(f"   No-clickbait: {len(true_labels) - clickbait_count} ({(len(true_labels) - clickbait_count)/len(true_labels)*100:.1f}%)")
    
    # Make predictions
    print(f"\n🔄 Making predictions...")
    try:
        results = classifier.predict_batch(texts, batch_size=32)
        predictions = [r['prediction'] for r in results]
        probabilities = [r['probabilities']['clickbait'] for r in results]
        
        print(f"✅ Predictions completed")
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        return
    
    # Calculate metrics
    print(f"\n📈 Calculating metrics...")
    
    # Basic metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        true_labels, predictions, average=None
    )
    
    # AUC score
    try:
        auc = roc_auc_score(true_labels, probabilities)
    except:
        auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Print results
    print(f"\n📊 EVALUATION RESULTS")
    print("=" * 30)
    
    metrics = {
        'eval_accuracy': accuracy,
        'eval_f1': f1,
        'eval_precision': precision,
        'eval_recall': recall
    }
    print_metrics(metrics, "Overall")
    
    print(f"\n📊 Detailed Metrics:")
    print(f"   • AUC Score: {auc:.4f}")
    
    print(f"\n📊 Per-Class Metrics:")
    class_names = ['No-Clickbait', 'Clickbait']
    for i, class_name in enumerate(class_names):
        print(f"   {class_name}:")
        print(f"     - Precision: {precision_per_class[i]:.4f}")
        print(f"     - Recall:    {recall_per_class[i]:.4f}")
        print(f"     - F1-Score:  {f1_per_class[i]:.4f}")
        print(f"     - Support:   {support[i]}")
    
    # Confusion matrix
    print_confusion_matrix(true_labels, predictions)
    
    # Save results if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        all_metrics = {
            'overall': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'auc': float(auc)
            },
            'per_class': {
                'no_clickbait': {
                    'precision': float(precision_per_class[0]),
                    'recall': float(recall_per_class[0]),
                    'f1': float(f1_per_class[0]),
                    'support': int(support[0])
                },
                'clickbait': {
                    'precision': float(precision_per_class[1]),
                    'recall': float(recall_per_class[1]),
                    'f1': float(f1_per_class[1]),
                    'support': int(support[1])
                }
            },
            'confusion_matrix': cm.tolist()
        }
        
        metrics_file = os.path.join(output_dir, 'evaluation_metrics.json')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        print(f"💾 Metrics saved to: {metrics_file}")
        
        # Save predictions
        prediction_results = []
        for i, (text, true_label, pred, prob) in enumerate(zip(texts, true_labels, predictions, probabilities)):
            prediction_results.append({
                'id': i,
                'text': text,
                'true_label': true_label,
                'predicted': pred,
                'probability': prob,
                'correct': true_label == pred,
                'true_class': 'clickbait' if true_label == 1 else 'no-clickbait',
                'predicted_class': 'clickbait' if pred == 1 else 'no-clickbait'
            })
        
        predictions_file = os.path.join(output_dir, 'predictions.jsonl')
        with open(predictions_file, 'w', encoding='utf-8') as f:
            for result in prediction_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"💾 Predictions saved to: {predictions_file}")
        
        # Plot and save confusion matrix
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            cm_file = os.path.join(output_dir, 'confusion_matrix.png')
            plt.savefig(cm_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 Confusion matrix plot saved to: {cm_file}")
        except Exception as e:
            print(f"⚠️ Could not save confusion matrix plot: {e}")
        
        # Error analysis
        errors = [r for r in prediction_results if not r['correct']]
        
        print(f"\n❌ Error Analysis:")
        print(f"   Total errors: {len(errors)} out of {len(prediction_results)} ({len(errors)/len(prediction_results)*100:.1f}%)")
        
        if errors:
            # False positives (predicted clickbait, actually not)
            false_positives = [e for e in errors if e['predicted'] == 1]
            # False negatives (predicted not clickbait, actually clickbait)
            false_negatives = [e for e in errors if e['predicted'] == 0]
            
            print(f"   False Positives: {len(false_positives)} (predicted clickbait, actually not)")
            print(f"   False Negatives: {len(false_negatives)} (predicted not clickbait, actually clickbait)")
            
            # Save errors
            errors_file = os.path.join(output_dir, 'errors.jsonl')
            with open(errors_file, 'w', encoding='utf-8') as f:
                for error in errors:
                    f.write(json.dumps(error, ensure_ascii=False) + '\n')
            print(f"💾 Error analysis saved to: {errors_file}")
            
            # Show some examples
            print(f"\n🔍 Sample errors:")
            for i, error in enumerate(errors[:5]):
                print(f"   {i+1}. Text: {error['text'][:80]}...")
                print(f"      True: {error['true_class']}, Predicted: {error['predicted_class']}")
    
    return metrics

def compare_models(model_paths: list, test_file: str, output_dir: str = None):
    """Compare multiple models"""
    print(f"\n🔄 Comparing {len(model_paths)} models...")
    
    results = {}
    
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print(f"\n📊 Evaluating {model_name}...")
        
        try:
            # Create sub-directory for this model
            model_output_dir = os.path.join(output_dir, model_name) if output_dir else None
            metrics = evaluate_model(model_path, test_file, model_output_dir)
            results[model_name] = metrics
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            results[model_name] = None
    
    # Compare results
    if len(results) > 1:
        print(f"\n📊 MODEL COMPARISON")
        print("=" * 50)
        
        print(f"{'Model':<25} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}")
        print("-" * 65)
        
        for model_name, metrics in results.items():
            if metrics:
                print(f"{model_name:<25} {metrics['eval_accuracy']:<10.4f} {metrics['eval_f1']:<10.4f} {metrics['eval_precision']:<10.4f} {metrics['eval_recall']:<10.4f}")
            else:
                print(f"{model_name:<25} {'Failed':<40}")
        
        # Find best model
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            best_model = max(valid_results.keys(), key=lambda k: valid_results[k]['eval_f1'])
            print(f"\n🏆 Best model by F1-score: {best_model} (F1: {valid_results[best_model]['eval_f1']:.4f})")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Clickbait Classification Model")
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--test_file", type=str, default="data/test/data.jsonl",
                       help="Path to test data file")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory to save results")
    parser.add_argument("--compare_models", nargs='+', default=None,
                       help="List of model paths to compare")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Check test file exists
    if not os.path.exists(args.test_file):
        print(f"❌ Test file not found: {args.test_file}")
        return
    
    if args.compare_models:
        # Compare multiple models
        compare_models(args.compare_models, args.test_file, args.output_dir)
    else:
        # Evaluate single model
        if not os.path.exists(args.model_dir):
            print(f"❌ Model directory not found: {args.model_dir}")
            return
        
        evaluate_model(args.model_dir, args.test_file, args.output_dir)
    
    print(f"\n✅ Evaluation completed!")

if __name__ == "__main__":
    main() 