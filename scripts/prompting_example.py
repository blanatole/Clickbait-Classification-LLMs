#!/usr/bin/env python3
"""
English-only prompting examples for clickbait classification
Methods: Zero-shot, Few-shot, Chain-of-Thought (CoT)
Examples taken from training dataset
"""

import os
import json
import time
import pandas as pd
import jsonlines
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

load_dotenv()

def initialize_llm():
    """Initialize ChatOpenAI model with working configuration"""
    llm = ChatOpenAI(
        model="deepseek-v3",  # Tested and stable
        temperature=0,
        api_key=os.environ.get("SHUBI_API_KEY"), 
        base_url=os.environ.get("SHUBI_URL")
    )
    return llm

def zero_shot_prompting_example(llm):
    """Zero-shot prompting example - no examples provided"""
    
    zero_shot_prompt = """You are a content analysis expert. Classify the following news headline:

Label definitions:
- 0: No-clickbait (factual, objective, clear information)
- 1: Clickbait (sensational, uses emotional language, withholds information, exaggerated)

Headline to classify: "{title}"

Respond with only 0 or 1:"""

    # Test example from training data
    test_title = "Oscar's biggest loser finally wins... on 21st try"
    
    prompt = zero_shot_prompt.format(title=test_title)
    
    print("=== ZERO-SHOT PROMPTING EXAMPLE ===\n")
    print(f"Test headline: {test_title}\n")
    
    response = llm.invoke([HumanMessage(content=prompt)])
    print("Zero-shot result:")
    print(response.content)
    print()

def few_shot_prompting_example(llm):
    """Few-shot prompting example with training data examples"""
    
    few_shot_prompt = """You are a clickbait classification expert. Here are some examples from our training data:

Example 1:
Headline: "Trump vows 35% tax for US firms that move jobs overseas"
Label: 0 (No-clickbait)
Reason: Clear, factual information about specific policy with concrete details

Example 2:
Headline: "Bet you didn't know government jobs paid so well :p"
Label: 1 (Clickbait)
Reason: Teasing tone, withholds specific information, creates curiosity gap

Example 3:
Headline: "John Glenn, American Hero of the Space Age, Dies at 95"
Label: 0 (No-clickbait)
Reason: Straightforward reporting of news event with clear facts

Example 4:
Headline: "Trump says Happy New Year in the most Trump way"
Label: 1 (Clickbait)
Reason: Vague description, doesn't specify what "Trump way" means, clickbait structure

Now classify this headline: "{title}"
Response format:
Label: [0/1]
Reason: [brief explanation]"""

    test_title = "The curious case of the billion-dollar lithium mine that sold on the cheap"
    
    prompt = few_shot_prompt.format(title=test_title)
    
    print("=== FEW-SHOT PROMPTING EXAMPLE ===\n")
    print(f"Test headline: {test_title}\n")
    
    response = llm.invoke([HumanMessage(content=prompt)])
    print("Few-shot result:")
    print(response.content)
    print()

def chain_of_thought_example(llm):
    """Chain of Thought prompting example"""
    
    cot_prompt = """Classify this clickbait headline step by step:

Headline: "{title}"

Analyze using these steps:
1. Identify emotional or attention-grabbing keywords
2. Evaluate information specificity - what facts are provided vs. withheld
3. Check for curiosity gaps - does it create questions without answers?
4. Assess overall tone and structure
5. Provide final classification with reasoning

Response format:
Step 1 - Keywords: [analysis]
Step 2 - Information specificity: [analysis] 
Step 3 - Curiosity gaps: [analysis]
Step 4 - Tone/structure: [analysis]
Step 5 - Final classification: [0 (No-clickbait) / 1 (Clickbait)] with reasoning"""

    test_title = "Wow, Instagram has a lot of followers."
    
    prompt = cot_prompt.format(title=test_title)
    
    print("=== CHAIN OF THOUGHT PROMPTING ===\n")
    print(f"Test headline: {test_title}\n")
    
    response = llm.invoke([HumanMessage(content=prompt)])
    print("Chain of Thought result:")
    print(response.content)
    print()

def batch_classification_example(llm):
    """Batch classification with training data examples"""
    
    # Real examples from training data with mixed labels
    sample_titles = [
        "Live Nation CEO Michael Rapino: \"I don't want to be in the secondary biz at all\"",  # no-clickbait
        "What happens when you don't shave for a whole year? Natural beauty, explains this blogger 🙌",  # clickbait
        "Cherokee Nation files first-of-its-kind opioid lawsuit against Wal-Mart, CVS and Walgreens",  # no-clickbait
        "The best fast food, picked by the world's top chefs",  # clickbait
        "Trump: \"We must fight\" hard-line conservative Freedom Caucus in 2018 midterm elections",  # no-clickbait
        "This woman sneaking pictures of The Rock's butt is you, me, and your grandma"  # clickbait
    ]
    
    results = []
    
    print("=== BATCH CLASSIFICATION: TRAINING DATA EXAMPLES ===\n")
    
    for i, title in enumerate(sample_titles, 1):
        print(f"Headline {i}: {title}")
        
        # Use zero-shot classification for batch
        result = classify_with_zero_shot(llm, title)
        results.append({
            "title": title,
            "prediction": result
        })
        
        if result != -1:
            label_map = {0: "No-clickbait", 1: "Clickbait"}
            label = label_map.get(result, "Unknown")
            print(f"Result: {result} - {label}\n")
        else:
            print("Unable to parse response\n")
        
        time.sleep(0.5)  # Rate limiting
    
    return results

def classify_with_zero_shot(llm, title):
    """Simple zero-shot classification"""
    prompt = f"""Classify this news headline as clickbait or not:
- 0: No-clickbait (factual, clear information)
- 1: Clickbait (sensational, creates curiosity)

Headline: "{title}"

Respond with only 0 or 1:"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()
        
        if response_text == '1':
            return 1
        elif response_text == '0':
            return 0
        else:
            return -1
    except Exception:
        return -1

def load_evaluation_data(file_path, limit=20):
    """Load data from jsonl file for evaluation"""
    data = []
    try:
        with jsonlines.open(file_path) as reader:
            for i, obj in enumerate(reader):
                if i >= limit:
                    break
                # Extract title from text field
                text_parts = obj['text'].split('[SEP]')
                title = text_parts[0].strip().strip('"')
                
                data.append({
                    'id': obj['id'],
                    'title': title,
                    'label': obj['label'],
                    'truth_class': obj['truth_class']
                })
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
    return data

def parse_prompt_response(response_text, method="zero_shot"):
    """Parse LLM response to extract prediction"""
    response_text = response_text.lower()
    
    # CoT specific parsing - look for final classification
    if method == "cot":
        # Look for "final:" or "step 5" patterns
        if 'final:' in response_text:
            final_part = response_text.split('final:')[1]
            if '1 (clickbait)' in final_part or '1(clickbait)' in final_part:
                return 1
            elif '0 (no-clickbait)' in final_part or '0(no-clickbait)' in final_part:
                return 0
        
        # Backup: look for explicit classification statements
        if '1 (clickbait)' in response_text or 'final classification: 1' in response_text:
            return 1
        elif '0 (no-clickbait)' in response_text or 'final classification: 0' in response_text:
            return 0
    
    # General patterns
    if '[1]' in response_text or 'label: 1' in response_text or 'classification: 1' in response_text:
        return 1
    elif '[0]' in response_text or 'label: 0' in response_text or 'classification: 0' in response_text:
        return 0
    elif response_text.strip() == '1':
        return 1
    elif response_text.strip() == '0':
        return 0
    elif 'clickbait)' in response_text and 'no-clickbait)' not in response_text:
        return 1
    elif 'no-clickbait)' in response_text:
        return 0
    
    return -1  # Unable to parse

def evaluate_prompting_method(llm, data, method="zero_shot", delay=1.0):
    """Evaluate a specific prompting method"""
    
    predictions = []
    true_labels = []
    
    print(f"\n=== EVALUATING METHOD: {method.upper()} ===")
    print(f"Processing {len(data)} samples...\n")
    
    for i, item in enumerate(data):
        title = item['title']
        true_label = item['label']
        
        print(f"[{i+1}/{len(data)}] Processing: {title[:60]}...")
        
        try:
            if method == "zero_shot":
                prompt = f"""Classify this news headline:
- 0: No-clickbait
- 1: Clickbait

Headline: "{title}"

Respond with only 0 or 1:"""
                
            elif method == "few_shot":
                prompt = f"""Classify clickbait with examples:

Headline: "Trump vows 35% tax for US firms that move jobs overseas"
Label: 0

Headline: "Bet you didn't know government jobs paid so well :p"
Label: 1

Headline: "John Glenn, American Hero of the Space Age, Dies at 95"
Label: 0

Headline: "{title}"
Label:"""
                
            elif method == "cot":
                prompt = f"""Classify this headline step by step:

Headline: "{title}"

Step 1 - Keywords: [identify attention-grabbing words]
Step 2 - Information: [what's provided vs. withheld]
Step 3 - Curiosity: [does it create gaps?]

Final Classification: [0 (No-clickbait) or 1 (Clickbait)]"""
            
            response = llm.invoke([HumanMessage(content=prompt)])
            prediction = parse_prompt_response(response.content, method)
            
            predictions.append(prediction)
            true_labels.append(true_label)
            
            if prediction != -1:
                label_map = {0: "No-clickbait", 1: "Clickbait"}
                label_name = label_map.get(prediction, "Unknown")
                print(f"Prediction: {prediction} - {label_name}")
            else:
                print("Unable to parse response")
            
            time.sleep(delay)
            
        except Exception as e:
            print(f"Error: {e}")
            predictions.append(-1)
            true_labels.append(true_label)
    
    return predictions, true_labels

def calculate_metrics(predictions, true_labels, method_name):
    """Calculate evaluation metrics"""
    
    # Filter valid predictions
    valid_indices = [i for i, pred in enumerate(predictions) if pred != -1]
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_true_labels = [true_labels[i] for i in valid_indices]
    
    if len(valid_predictions) == 0:
        print(f"No valid predictions for {method_name}")
        return None
    
    # Calculate metrics
    accuracy = accuracy_score(valid_true_labels, valid_predictions)
    precision = precision_score(valid_true_labels, valid_predictions, average='weighted', zero_division=0)
    recall = recall_score(valid_true_labels, valid_predictions, average='weighted', zero_division=0)
    f1 = f1_score(valid_true_labels, valid_predictions, average='weighted', zero_division=0)
    
    print(f"\n=== EVALUATION RESULTS: {method_name.upper()} ===")
    print(f"Valid samples: {len(valid_predictions)}/{len(predictions)}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    if len(set(valid_true_labels)) > 1:  # Multiple classes
        print(f"\nClassification Report:")
        print(classification_report(valid_true_labels, valid_predictions, 
                                  target_names=['0: No-clickbait', '1: Clickbait']))
    
    return {
        'method': method_name,
        'valid_samples': len(valid_predictions),
        'total_samples': len(predictions),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def run_evaluation_comparison(llm, data_limit=15):
    """Run evaluation comparison of prompting methods"""
    
    print("\n" + "="*60)
    print("    PROMPTING METHODS EVALUATION COMPARISON")
    print("="*60)
    
    # Load test data
    data_path = os.path.join("data", "test", "data_demo.jsonl")
    test_data = load_evaluation_data(data_path, limit=data_limit)
    
    if not test_data:
        print("Unable to load test data. Check data/test/data_demo.jsonl")
        return
    
    print(f"Loaded {len(test_data)} test samples")
    
    results = []
    methods = ["zero_shot", "few_shot", "cot"]
    
    for method in methods:
        try:
            predictions, true_labels = evaluate_prompting_method(llm, test_data, method)
            metrics = calculate_metrics(predictions, true_labels, method)
            if metrics:
                results.append(metrics)
        except Exception as e:
            print(f"Error in {method} evaluation: {e}")
    
    # Compare results
    if results:
        print("\n" + "="*60)
        print("           FINAL COMPARISON RESULTS")
        print("="*60)
        
        df = pd.DataFrame(results)
        print(df.to_string(index=False, float_format='%.3f'))
        
        # Find best method
        best_f1 = df.loc[df['f1_score'].idxmax()]
        print(f"\nBest method (by F1-Score): {best_f1['method']} (F1: {best_f1['f1_score']:.3f})")

def main():
    """Run prompting examples"""
    
    # Check environment variables
    if not os.environ.get("SHUBI_API_KEY") or not os.environ.get("SHUBI_URL"):
        print("❌ Error: Please set SHUBI_API_KEY and SHUBI_URL in .env file")
        return
    
    print("🚀 Initializing ChatOpenAI...")
    try:
        llm = initialize_llm()
        print("✅ Initialization successful!\n")
    except Exception as e:
        print(f"❌ LLM initialization error: {e}")
        return
    
    print("=" * 60)
    print("      ENGLISH PROMPTING METHODS DEMONSTRATION")
    print("=" * 60)
    
    # Run individual method examples
    zero_shot_prompting_example(llm)
    few_shot_prompting_example(llm)
    chain_of_thought_example(llm)
    
    # Run batch classification
    batch_results = batch_classification_example(llm)
    
    print("=" * 60)
    print("✅ All prompting examples completed!")
    print("=" * 60)
    
    # Ask about evaluation
    print("\nWould you like to run evaluation comparison?")
    print("(This will take a few minutes with real data)")
    print("\nTo run evaluation, use: python scripts/run_evaluation.py")

if __name__ == "__main__":
    main() 