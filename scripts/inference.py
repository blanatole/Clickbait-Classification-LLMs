#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference Script cho Clickbait Classification
Test model đã train với examples hoặc interactive mode
"""

import os
import sys
import json
import torch
import argparse
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_device, set_seed

class ClickbaitClassifier:
    """Clickbait Classifier for inference"""
    
    def __init__(self, model_path: str, device: torch.device = None):
        self.model_path = model_path
        self.device = device or get_device()
        
        print(f"Loading model from: {model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict(self, text: str, return_probabilities: bool = False) -> Dict[str, Any]:
        """Predict clickbait for a single text"""
        
        # Tokenize
        inputs = self.tokenizer(
            text, 
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Get prediction
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
        
        result = {
            'text': text,
            'predicted_class': 'clickbait' if predicted_class == 1 else 'no-clickbait',
            'confidence': confidence,
            'prediction': predicted_class
        }
        
        if return_probabilities:
            result['probabilities'] = {
                'no-clickbait': probabilities[0][0].item(),
                'clickbait': probabilities[0][1].item()
            }
        
        return result
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Predict clickbait for multiple texts"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Process results
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            confidences = torch.max(probabilities, dim=-1)[0].cpu().numpy()
            probs = probabilities.cpu().numpy()
            
            for j, text in enumerate(batch_texts):
                result = {
                    'text': text,
                    'predicted_class': 'clickbait' if predictions[j] == 1 else 'no-clickbait',
                    'confidence': float(confidences[j]),
                    'prediction': int(predictions[j]),
                    'probabilities': {
                        'no-clickbait': float(probs[j][0]),
                        'clickbait': float(probs[j][1])
                    }
                }
                results.append(result)
        
        return results

def test_examples():
    """Test examples for clickbait detection"""
    examples = [
        # Clickbait examples
        "You Won't Believe What Happened Next!",
        "10 Shocking Facts That Will Change Your Life Forever",
        "This Simple Trick Will Save You Thousands of Dollars",
        "Doctors Hate This One Simple Trick",
        "The Secret That Big Companies Don't Want You to Know",
        "What Happens Next Will Shock You",
        "Number 7 Will Blow Your Mind",
        "This Video Will Make You Cry",
        
        # Non-clickbait examples
        "Study Shows Correlation Between Exercise and Mental Health",
        "New COVID-19 Variant Detected in Multiple Countries",
        "Federal Reserve Raises Interest Rates by 0.25%",
        "Scientists Discover New Species in Amazon Rainforest",
        "Apple Reports Quarterly Earnings Beat Expectations",
        "Climate Change Report Released by UN Panel",
        "Local School District Announces Budget Cuts",
        "Research Team Publishes Findings on Cancer Treatment",
        
        # Mixed/Ambiguous examples
        "How to Increase Your Productivity in 5 Simple Steps",
        "The Truth About Social Media and Mental Health",
        "Why Everyone is Talking About This New Technology",
        "Things You Need to Know About Online Privacy",
    ]
    
    return examples

def interactive_mode(classifier: ClickbaitClassifier):
    """Interactive mode for testing custom inputs"""
    print("\n🎯 INTERACTIVE MODE")
    print("Enter text to classify (or 'quit' to exit):")
    print("-" * 50)
    
    while True:
        try:
            text = input("\n📝 Enter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                print("Please enter some text.")
                continue
            
            # Predict
            result = classifier.predict(text, return_probabilities=True)
            
            # Display result
            print(f"\n📊 Result:")
            print(f"   Text: {text}")
            print(f"   Prediction: {result['predicted_class']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Probabilities:")
            print(f"     • No-clickbait: {result['probabilities']['no-clickbait']:.3f}")
            print(f"     • Clickbait:    {result['probabilities']['clickbait']:.3f}")
            
            # Visual indicator
            if result['predicted_class'] == 'clickbait':
                if result['confidence'] > 0.8:
                    print("   🔴 Strong clickbait signal")
                elif result['confidence'] > 0.6:
                    print("   🟡 Moderate clickbait signal")
                else:
                    print("   🟢 Weak clickbait signal")
            else:
                if result['confidence'] > 0.8:
                    print("   ✅ Clearly not clickbait")
                elif result['confidence'] > 0.6:
                    print("   🟡 Probably not clickbait")
                else:
                    print("   🔴 Uncertain classification")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Interactive mode ended.")

def main():
    parser = argparse.ArgumentParser(description="Clickbait Classification Inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--test_examples", action="store_true",
                       help="Test with predefined examples")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--input_file", type=str, default=None,
                       help="Input JSONL file with texts to classify")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output JSONL file for results")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for inference")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    print("🎯 CLICKBAIT CLASSIFICATION INFERENCE")
    print("=" * 50)
    
    # Set seed
    set_seed(args.seed)
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model path not found: {args.model_path}")
        return
    
    # Load classifier
    try:
        classifier = ClickbaitClassifier(args.model_path)
    except Exception as e:
        print(f"❌ Failed to load classifier: {e}")
        return
    
    # Test examples
    if args.test_examples:
        print("\n🧪 TESTING PREDEFINED EXAMPLES")
        print("-" * 30)
        
        examples = test_examples()
        results = classifier.predict_batch(examples)
        
        # Group by prediction
        clickbait_results = [r for r in results if r['predicted_class'] == 'clickbait']
        no_clickbait_results = [r for r in results if r['predicted_class'] == 'no-clickbait']
        
        print(f"\n🔴 CLASSIFIED AS CLICKBAIT ({len(clickbait_results)}):")
        for result in sorted(clickbait_results, key=lambda x: x['confidence'], reverse=True):
            print(f"   {result['confidence']:.3f} | {result['text']}")
        
        print(f"\n✅ CLASSIFIED AS NO-CLICKBAIT ({len(no_clickbait_results)}):")
        for result in sorted(no_clickbait_results, key=lambda x: x['confidence'], reverse=True):
            print(f"   {result['confidence']:.3f} | {result['text']}")
        
        # Summary
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        print(f"\n📊 Summary:")
        print(f"   Total examples: {len(examples)}")
        print(f"   Clickbait: {len(clickbait_results)} ({len(clickbait_results)/len(examples)*100:.1f}%)")
        print(f"   No-clickbait: {len(no_clickbait_results)} ({len(no_clickbait_results)/len(examples)*100:.1f}%)")
        print(f"   Average confidence: {avg_confidence:.3f}")
    
    # Process input file
    if args.input_file:
        print(f"\n📂 PROCESSING INPUT FILE: {args.input_file}")
        
        if not os.path.exists(args.input_file):
            print(f"❌ Input file not found: {args.input_file}")
            return
        
        # Load input data
        texts = []
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        texts.append(data.get('text', ''))
        except Exception as e:
            print(f"❌ Error reading input file: {e}")
            return
        
        if not texts:
            print("❌ No texts found in input file")
            return
        
        print(f"Processing {len(texts)} texts...")
        
        # Predict
        results = classifier.predict_batch(texts, batch_size=args.batch_size)
        
        # Save results
        if args.output_file:
            try:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    for result in results:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                print(f"✅ Results saved to: {args.output_file}")
            except Exception as e:
                print(f"❌ Error saving results: {e}")
        else:
            # Print results
            for result in results[:10]:  # Show first 10
                print(f"   {result['predicted_class']:12} | {result['confidence']:.3f} | {result['text'][:60]}...")
            if len(results) > 10:
                print(f"   ... and {len(results) - 10} more")
        
        # Summary
        clickbait_count = sum(1 for r in results if r['predicted_class'] == 'clickbait')
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        
        print(f"\n📊 Processing Summary:")
        print(f"   Total processed: {len(results)}")
        print(f"   Clickbait: {clickbait_count} ({clickbait_count/len(results)*100:.1f}%)")
        print(f"   No-clickbait: {len(results) - clickbait_count} ({(len(results) - clickbait_count)/len(results)*100:.1f}%)")
        print(f"   Average confidence: {avg_confidence:.3f}")
    
    # Interactive mode
    if args.interactive:
        interactive_mode(classifier)
    
    # Default: show usage
    if not any([args.test_examples, args.input_file, args.interactive]):
        print("\n💡 Usage examples:")
        print("   Test examples:    python src/inference.py --model_path outputs/model --test_examples")
        print("   Interactive mode: python src/inference.py --model_path outputs/model --interactive")
        print("   Process file:     python src/inference.py --model_path outputs/model --input_file data.jsonl --output_file results.jsonl")

if __name__ == "__main__":
    main() 