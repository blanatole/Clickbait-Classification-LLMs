#!/usr/bin/env python3
"""
Evaluate fine-tuned models on the test set
Following the step-by-step guide for Webis-Clickbait-17 dataset
"""

import argparse
import json
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
import numpy as np


def load_model_and_tokenizer(model_dir):
    """Load model and tokenizer from directory"""
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return model, tokenizer


def evaluate_model(model_dir, test_file, output_file=None):
    """Evaluate model on test dataset"""
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_dir)
    
    # Create pipeline
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if model.device.type == 'cuda' else -1
    )
    
    # Load test data
    print(f"Loading test data from {test_file}...")
    test_dataset = load_dataset("json", data_files={"test": test_file})["test"]
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Get predictions
    print("Getting predictions...")
    texts = test_dataset["text"]
    true_labels = test_dataset["label"]
    
    # Predict in batches for efficiency
    batch_size = 32
    predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_preds = classifier(batch_texts)
        
        # Convert predictions to binary labels (assuming LABEL_1 is clickbait)
        batch_labels = [1 if pred["label"] == "LABEL_1" else 0 for pred in batch_preds]
        predictions.extend(batch_labels)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"Processed {i + len(batch_texts)} / {len(texts)} samples")
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    precision, recall, f1_micro, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted'
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Model: {model_dir}")
    print(f"Test file: {test_file}")
    print(f"Test samples: {len(test_dataset)}")
    print("-"*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("-"*50)
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(
        true_labels, 
        predictions,
        target_names=["No Clickbait", "Clickbait"],
        digits=4
    ))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predictions)
    print(f"True Negatives:  {cm[0,0]}")
    print(f"False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}")
    print(f"True Positives:  {cm[1,1]}")
    
    # Save results
    results = {
        "model_path": model_dir,
        "test_file": test_file,
        "test_samples": len(test_dataset),
        "metrics": {
            "accuracy": float(accuracy),
            "f1_weighted": float(f1),
            "precision_weighted": float(precision),
            "recall_weighted": float(recall)
        },
        "confusion_matrix": {
            "true_negatives": int(cm[0,0]),
            "false_positives": int(cm[0,1]),
            "false_negatives": int(cm[1,0]),
            "true_positives": int(cm[1,1])
        }
    }
    
    if output_file:
        print(f"\nSaving results to {output_file}...")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
    
    return results


def demo_inference(model_dir, texts):
    """Demo inference on sample texts"""
    print(f"\nRunning demo inference with model: {model_dir}")
    print("-"*50)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_dir)
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if model.device.type == 'cuda' else -1
    )
    
    for i, text in enumerate(texts):
        result = classifier(text)
        label = "Clickbait" if result[0]["label"] == "LABEL_1" else "No Clickbait"
        confidence = result[0]["score"]
        
        print(f"Text {i+1}: {text[:100]}...")
        print(f"Prediction: {label} (confidence: {confidence:.3f})")
        print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned clickbait classification model")
    parser.add_argument("--model_dir", required=True, help="Path to the fine-tuned model directory")
    parser.add_argument("--test_file", default="data/test/data.jsonl", help="Path to test data file")
    parser.add_argument("--output_file", help="Path to save evaluation results (JSON)")
    parser.add_argument("--demo", action="store_true", help="Run demo inference on sample texts")
    
    args = parser.parse_args()
    
    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory {args.model_dir} does not exist!")
        return
    
    # Check if test file exists
    if not os.path.exists(args.test_file):
        print(f"Error: Test file {args.test_file} does not exist!")
        return
    
    # Evaluate model
    results = evaluate_model(args.model_dir, args.test_file, args.output_file)
    
    # Demo inference if requested
    if args.demo:
        demo_texts = [
            "Bạn sẽ không tin điều xảy ra tiếp theo khi cô gái này mở cửa...",
            "Chính phủ công bố kế hoạch kinh tế mới cho năm 2024",
            "7 bí mật mà các nhà hàng không muốn bạn biết!",
            "Nghiên cứu mới về biến đổi khí hậu được công bố trên tạp chí Nature"
        ]
        demo_inference(args.model_dir, demo_texts)


if __name__ == "__main__":
    main() 