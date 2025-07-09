#!/usr/bin/env python3
"""
Chain-of-Thought Prompting for clickbait classification
Method: Chain-of-Thought prompting with systematic step-by-step analysis
"""

import os
import json
import time
import pandas as pd
import jsonlines
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

load_dotenv()

def initialize_llm():
    """Initialize OpenAI client with GPT-4o-mini"""
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    return client

def chain_of_thought_prompting(client, title):
    """Chain-of-thought prompting with systematic step-by-step analysis"""
    prompt = f"""Analyze this headline to classify if it's clickbait:

Headline: "{title}"

Think step by step:
1. Language: Emotional/sensational words?
2. Information: Clear facts or curiosity gaps?  
3. Intent: Inform or manipulate clicks?

Analysis: [Think through each step briefly]

Final answer: 0 (not clickbait) or 1 (clickbait)"""
    
    try:
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are an expert clickbait detection specialist. Think step by step but be concise."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Multiple parsing attempts
        if "final answer:" in response_text.lower():
            answer_part = response_text.lower().split("final answer:")[-1]
            if "1" in answer_part[:10]:  # Check first 10 chars after "final answer:"
                return 1
            elif "0" in answer_part[:10]:
                return 0
        
        return parse_prompt_response(response_text, "cot")
    except Exception as e:
        print(f"Error in chain-of-thought prompting: {e}")
        return -1

def parse_prompt_response(response_text, method="cot"):
    """Parse LLM response to extract prediction with improved flexibility"""
    original_text = response_text
    response_text = response_text.lower()
    

    
    # CoT specific parsing - look for final classification
    if method == "cot" or "cot" in method:
        # Look for "final:" patterns
        if 'final:' in response_text or 'final answer:' in response_text or 'final classification:' in response_text:
            final_part = response_text.split('final')[-1]
            if '1' in final_part and ('clickbait' in final_part or 'yes' in final_part):
                return 1
            elif '0' in final_part and ('no-clickbait' in final_part or 'no' in final_part):
                return 0
        
        # Look for step 5 patterns
        if 'step 5' in response_text:
            step5_part = response_text.split('step 5')[-1]
            if '1' in step5_part and 'clickbait' in step5_part:
                return 1
            elif '0' in step5_part and ('no-clickbait' in step5_part or 'not clickbait' in step5_part):
                return 0
        
        # Look for "answer:" patterns
        if 'answer:' in response_text:
            answer_part = response_text.split('answer:')[-1]
            if '1' in answer_part and ('clickbait' in answer_part or answer_part.strip().startswith('1')):
                return 1
            elif '0' in answer_part and ('no-clickbait' in answer_part or answer_part.strip().startswith('0')):
                return 0
    
    # Enhanced general patterns
    patterns_1 = [
        '[1]', 'answer: 1', 'classification: 1', 'label: 1',
        '1 (clickbait)', '1(clickbait)', 'result: 1',
        'output: 1', 'prediction: 1'
    ]
    
    patterns_0 = [
        '[0]', 'answer: 0', 'classification: 0', 'label: 0', 
        '0 (no-clickbait)', '0(no-clickbait)', 'result: 0',
        'output: 0', 'prediction: 0'
    ]
    
    # Check for explicit patterns
    for pattern in patterns_1:
        if pattern in response_text:
            return 1
    
    for pattern in patterns_0:
        if pattern in response_text:
            return 0
    
    # Fallback: look for standalone numbers at end of response
    lines = response_text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line == '1' or line.endswith(' 1') or line.startswith('1 '):
            return 1
        elif line == '0' or line.endswith(' 0') or line.startswith('0 '):
            return 0
    
    # Last resort: check for clickbait keywords
    if 'clickbait' in response_text and 'no-clickbait' not in response_text and 'not clickbait' not in response_text:
        return 1
    elif 'no-clickbait' in response_text or 'not clickbait' in response_text:
        return 0
    

    return -1  # Unable to parse

def load_evaluation_data(file_path, limit=50):
    """Load evaluation data from JSONL file"""
    data = []
    try:
        # Try direct path first
        if os.path.exists(file_path):
            current_path = file_path
        else:
            # Try relative path from current directory
            current_path = os.path.join("../../shared/data/test/data.jsonl")
            if not os.path.exists(current_path):
                # Try other paths
                possible_paths = [
                    "../../shared/data/test/data_demo.jsonl",
                    "../shared/data/test/data.jsonl",
                    "shared/data/test/data.jsonl",
                    "data/test/data.jsonl"
                ]
                current_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        current_path = path
                        break
        
        if current_path and os.path.exists(current_path):
            print(f"✅ Loading data from: {current_path}")
            with open(current_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= limit:
                        break
                    item = json.loads(line.strip())
                    # Extract title from text field if needed
                    if 'text' in item and '[SEP]' in item['text']:
                        text_parts = item['text'].split('[SEP]')
                        title = text_parts[0].strip().strip('"')
                        data.append({
                            'id': item.get('id', i),
                            'title': title,
                            'label': item['label'],
                            'truth_class': item.get('truth_class', 'unknown')
                        })
                    else:
                        data.append(item)
            print(f"✅ Loaded {len(data)} data samples")
        else:
            print("❌ Test data file not found")
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
    
    return data

def calculate_metrics(predictions, true_labels, method_name):
    """Calculate evaluation metrics"""
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

def evaluate_cot_method(client, data, delay=1.5):
    """Evaluate chain-of-thought prompting method"""
    predictions = []
    true_labels = []
    
    print(f"\n{'='*50}")
    print(f"🧪 EVALUATING: CHAIN-OF-THOUGHT PROMPTING")
    print(f"{'='*50}")
    
    for i, item in enumerate(data):
        title = item['title']
        true_label = item['label']
        
        print(f"[{i+1}/{len(data)}] {title[:60]}...")
        
        prediction = chain_of_thought_prompting(client, title)
        predictions.append(prediction)
        true_labels.append(true_label)
        
        if prediction != -1:
            label_map = {0: "No-clickbait", 1: "Clickbait"}
            label_name = label_map.get(prediction, "Unknown")
            print(f"  → {prediction} - {label_name}")
        else:
            print(f"  → Response parsing error")
        
        time.sleep(delay)  # Rate limiting (longer for CoT)
    
    return predictions, true_labels

def run_cot_evaluation(client, data_limit=20):
    """Run chain-of-thought evaluation"""
    print("🚀 CHAIN-OF-THOUGHT PROMPTING EVALUATION")
    print("=" * 60)
    
    # Load test data
    data_path = "../../shared/data/test/data.jsonl"
    test_data = load_evaluation_data(data_path, limit=data_limit)
    
    if not test_data:
        print("❌ Unable to load test data")
        return
    
    print(f"✅ Loaded {len(test_data)} test samples\n")
    
    try:
        predictions, true_labels = evaluate_cot_method(
            client, test_data, delay=1.5  # Longer delay for CoT
        )
        metrics = calculate_metrics(predictions, true_labels, "chain_of_thought")
        if metrics:
            print(f"✅ Chain-of-thought evaluation completed successfully!")
        else:
            print(f"⚠️ Chain-of-thought evaluation failed to generate valid metrics")
            
    except KeyboardInterrupt:
        print("\n⚠️ Stopped by user")
    except Exception as e:
        print(f"❌ Error in chain-of-thought evaluation: {e}")

def demo_cot_examples(client):
    """Demo with specific examples"""
    print("\n🎯 CHAIN-OF-THOUGHT PROMPTING DEMO")
    print("=" * 50)
    
    examples = [
        "Scientists discover breakthrough in cancer treatment",
        "You'll NEVER guess what happened when she opened the door",
        "Federal Reserve chairman announces interest rate decision",
        "This SHOCKING trick will change your life forever",
        "Apple stock rises 4% following earnings report"
    ]
    
    for i, title in enumerate(examples, 1):
        print(f"\n--- Example {i} ---")
        print(f"Headline: {title}")
        
        # Test chain-of-thought
        result = chain_of_thought_prompting(client, title)
        label = "Clickbait" if result == 1 else "Not Clickbait" if result == 0 else "Error"
        print(f"Result: {result} ({label})")
        
        time.sleep(1.0)  # Longer delay for CoT

def main():
    """Main function with examples"""
    print("🎯 CHAIN-OF-THOUGHT PROMPTING FOR CLICKBAIT CLASSIFICATION (OpenAI)")
    print("=" * 70)
    
    # Check environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: Please set OPENAI_API_KEY in .env file")
        print("   Get your API key from: https://platform.openai.com/api-keys")
        return
    
    print("🚀 Initializing OpenAI client...")
    try:
        client = initialize_llm()
        print("✅ Initialization successful!\n")
    except Exception as e:
        print(f"❌ OpenAI client initialization error: {e}")
        return
    
    # Run demo
    demo_cot_examples(client)
    
    # Run evaluation - test with 20 samples first (CoT takes longer)
    print("\n" + "="*70)
    print("Would you like to run full evaluation? (20 samples - CoT takes longer)")
    print("Press Enter to continue or Ctrl+C to stop...")
    input()
    
    run_cot_evaluation(client, data_limit=20)

if __name__ == "__main__":
    main() 