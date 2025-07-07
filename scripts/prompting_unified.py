#!/usr/bin/env python3
"""
Unified prompting methods for clickbait classification
Combines basic and improved prompting strategies
Methods: Zero-shot, Few-shot, Chain-of-Thought (CoT), Ensemble
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

# ==================== BASIC PROMPTING METHODS ====================

def zero_shot_prompting(llm, title):
    """Basic zero-shot prompting"""
    prompt = f"""You are a content analysis expert. Classify the following news headline:

Label definitions:
- 0: No-clickbait (factual, objective, clear information)
- 1: Clickbait (sensational, uses emotional language, withholds information, exaggerated)

Headline to classify: "{title}"

Respond with only 0 or 1:"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return parse_prompt_response(response.content, "zero_shot")
    except Exception:
        return -1

def few_shot_prompting(llm, title):
    """Basic few-shot prompting with examples"""
    prompt = f"""You are a clickbait classification expert. Here are some examples:

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
Reason: Vague description, doesn't specify what "Trump way" means

Now classify this headline: "{title}"
Response format:
Label: [0/1]
Reason: [brief explanation]"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return parse_prompt_response(response.content, "few_shot")
    except Exception:
        return -1

def chain_of_thought_prompting(llm, title):
    """Chain of Thought prompting"""
    prompt = f"""Classify this clickbait headline step by step:

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
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return parse_prompt_response(response.content, "cot")
    except Exception:
        return -1

# ==================== IMPROVED PROMPTING METHODS ====================

def improved_zero_shot(llm, title):
    """Improved zero-shot with better prompt engineering"""
    prompt = f"""You are an expert content analyst specializing in clickbait detection.

TASK: Classify this news headline as clickbait or not.

DEFINITIONS:
- NO-CLICKBAIT (0): Factual, objective, clear information, specific details, straightforward reporting
- CLICKBAIT (1): Sensational language, emotional hooks, vague descriptions, curiosity gaps, exaggerated claims

CLICKBAIT INDICATORS:
- Emotional words: "SHOCKING", "AMAZING", "INCREDIBLE", "WOW"
- Vague phrases: "this will surprise you", "you won't believe", "what happens next"
- Curiosity gaps: withholding key information, creating questions
- Exaggerated numbers: "99%", "everyone", "nobody knows"
- Question hooks: "Can you spot?", "What happens when?"

Headline: "{title}"

Analysis: First, identify if this headline uses any clickbait techniques. Then classify.

Classification (0 or 1):"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return parse_prompt_response(response.content, "improved_zero_shot")
    except Exception:
        return -1

def improved_few_shot(llm, title):
    """Improved few-shot with better examples and format"""
    prompt = f"""You are a clickbait classification expert. Classify headlines based on these examples:

EXAMPLES:

Headline: "Apple announces iPhone 15 with new features and pricing"
Classification: 0 (No-clickbait)
Reason: Clear, factual information about product announcement

Headline: "You won't BELIEVE what this celebrity did next!"
Classification: 1 (Clickbait) 
Reason: Emotional language "BELIEVE", vague "what", creates curiosity

Headline: "Federal Reserve raises interest rates by 0.25%"
Classification: 0 (No-clickbait)
Reason: Specific, factual economic news with concrete numbers

Headline: "This simple trick will change your life forever"
Classification: 1 (Clickbait)
Reason: Vague "simple trick", exaggerated "change your life forever"

Headline: "Breaking: Major earthquake hits California, magnitude 6.2"
Classification: 0 (No-clickbait)
Reason: Clear breaking news with specific details

Headline: "10 things that will SHOCK you about your smartphone"
Classification: 1 (Clickbait)
Reason: Emotional "SHOCK", list format, vague "things"

NOW CLASSIFY THIS HEADLINE:
Headline: "{title}"
Classification:"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return parse_prompt_response(response.content, "improved_few_shot")
    except Exception:
        return -1

def improved_cot(llm, title):
    """Improved Chain of Thought with structured analysis"""
    prompt = f"""Analyze this headline step by step to determine if it's clickbait:

Headline: "{title}"

STEP-BY-STEP ANALYSIS:

1. EMOTIONAL LANGUAGE CHECK:
   - Does it use sensational words? (SHOCK, AMAZING, INCREDIBLE, etc.)
   - Does it use emotional hooks? (WOW, OMG, etc.)

2. INFORMATION SPECIFICITY CHECK:
   - Does it provide clear, specific facts?
   - Or does it use vague descriptions?

3. CURIOSITY GAP CHECK:
   - Does it withhold key information to create curiosity?
   - Does it create questions without answers?

4. EXAGGERATION CHECK:
   - Does it use exaggerated claims or numbers?
   - Does it promise unrealistic outcomes?

5. STRUCTURE CHECK:
   - Is it straightforward reporting?
   - Or does it use clickbait patterns?

FINAL CLASSIFICATION:
Based on the above analysis, this headline is:
[0 = No-clickbait (factual, clear, specific)]
[1 = Clickbait (sensational, vague, creates curiosity)]

Answer:"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return parse_prompt_response(response.content, "improved_cot")
    except Exception:
        return -1

def ensemble_classification(llm, title):
    """Ensemble method - combine multiple approaches"""
    results = []
    
    # Get predictions from all improved methods
    zero_result = improved_zero_shot(llm, title)
    few_result = improved_few_shot(llm, title) 
    cot_result = improved_cot(llm, title)
    
    # Collect valid results
    if zero_result != -1:
        results.append(zero_result)
    if few_result != -1:
        results.append(few_result)
    if cot_result != -1:
        results.append(cot_result)
    
    # Majority voting
    if len(results) > 0:
        # Count votes
        vote_0 = results.count(0)
        vote_1 = results.count(1)
        
        if vote_0 > vote_1:
            return 0
        elif vote_1 > vote_0:
            return 1
        else:
            # Tie - use zero-shot as tiebreaker
            return zero_result if zero_result != -1 else results[0]
    
    return -1

# ==================== UTILITY FUNCTIONS ====================

def parse_prompt_response(response_text, method="zero_shot"):
    """Parse response from different prompting methods"""
    response_text = response_text.strip().lower()
    
    # Extract number from response
    if "0" in response_text and "1" not in response_text:
        return 0
    elif "1" in response_text and "0" not in response_text:
        return 1
    elif "no-clickbait" in response_text or "not clickbait" in response_text:
        return 0
    elif "clickbait" in response_text:
        return 1
    else:
        # Try to find the last number in the response
        import re
        numbers = re.findall(r'\b[01]\b', response_text)
        if numbers:
            return int(numbers[-1])
    
    return -1

def load_evaluation_data(file_path, limit=20):
    """Load data from jsonl file for evaluation"""
    data = []
    try:
        with jsonlines.open(file_path) as reader:
            for i, obj in enumerate(reader):
                if i >= limit:
                    break
                data.append(obj)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def evaluate_prompting_method(llm, data, method="zero_shot", delay=1.0):
    """Evaluate a specific prompting method"""
    predictions = []
    true_labels = []
    
    methods = {
        "zero_shot": zero_shot_prompting,
        "few_shot": few_shot_prompting,
        "cot": chain_of_thought_prompting,
        "improved_zero_shot": improved_zero_shot,
        "improved_few_shot": improved_few_shot,
        "improved_cot": improved_cot,
        "ensemble": ensemble_classification
    }
    
    if method not in methods:
        print(f"❌ Unknown method: {method}")
        return [], []
    
    method_func = methods[method]
    
    for i, item in enumerate(data):
        title = item['title']
        true_label = item['label']
        
        print(f"[{i+1}/{len(data)}] {title[:50]}...")
        
        prediction = method_func(llm, title)
        predictions.append(prediction)
        true_labels.append(true_label)
        
        if prediction != -1:
            label_map = {0: "No-clickbait", 1: "Clickbait"}
            label_name = label_map.get(prediction, "Unknown")
            print(f"  → {prediction} - {label_name}")
        else:
            print(f"  → Failed to parse response")
        
        time.sleep(delay)  # Rate limiting
    
    return predictions, true_labels

def calculate_metrics(predictions, true_labels, method_name):
    """Calculate evaluation metrics"""
    # Filter out failed predictions (-1)
    valid_pairs = [(pred, true) for pred, true in zip(predictions, true_labels) if pred != -1]
    
    if not valid_pairs:
        print(f"❌ No valid predictions for {method_name}")
        return None
    
    valid_predictions, valid_true_labels = zip(*valid_pairs)
    
    # Calculate metrics
    accuracy = accuracy_score(valid_true_labels, valid_predictions)
    precision = precision_score(valid_true_labels, valid_predictions, average='weighted')
    recall = recall_score(valid_true_labels, valid_predictions, average='weighted')
    f1 = f1_score(valid_true_labels, valid_predictions, average='weighted')
    
    return {
        "method": method_name,
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "valid_samples": len(valid_predictions),
        "total_samples": len(predictions)
    }

def run_evaluation_comparison(llm, data_limit=15):
    """Run comparison of all prompting methods"""
    print("🚀 EVALUATING ALL PROMPTING METHODS")
    print("=" * 60)
    
    # Load test data
    data_path = "data/test/data_demo.jsonl"
    test_data = load_evaluation_data(data_path, limit=data_limit)
    
    if not test_data:
        print("❌ Unable to load test data")
        return
    
    print(f"✅ Loaded {len(test_data)} test samples\n")
    
    methods = [
        "zero_shot", "few_shot", "cot",
        "improved_zero_shot", "improved_few_shot", "improved_cot", "ensemble"
    ]
    
    results = []
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"🧪 TESTING: {method.upper()}")
        print(f"{'='*50}")
        
        try:
            predictions, true_labels = evaluate_prompting_method(
                llm, test_data, method, delay=1.0
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
        output_file = "prompting_evaluation_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        # Method analysis
        print(f"\n📋 METHOD ANALYSIS:")
        for _, row in df.iterrows():
            success_rate = (row['valid_samples'] / row['total_samples']) * 100
            print(f"   {row['method']}: {row['f1_score']:.3f} F1 ({success_rate:.1f}% success rate)")
        
    else:
        print("❌ No valid results to compare")

def main():
    """Main function with examples"""
    print("🎯 UNIFIED PROMPTING METHODS FOR CLICKBAIT CLASSIFICATION")
    print("=" * 70)
    
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
    
    # Run evaluation
    run_evaluation_comparison(llm, data_limit=15)

if __name__ == "__main__":
    main() 