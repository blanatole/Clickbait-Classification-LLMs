#!/usr/bin/env python3
"""
Improved prompting strategies for better clickbait classification accuracy
"""

from prompting_example import initialize_llm, load_evaluation_data, parse_prompt_response
from langchain_core.messages import HumanMessage
import time
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

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
        return parse_prompt_response(response.content, "zero_shot")
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
        return parse_prompt_response(response.content, "few_shot")
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
        return parse_prompt_response(response.content, "cot")
    except Exception:
        return -1

def ensemble_classification(llm, title):
    """Ensemble method - combine multiple approaches"""
    results = []
    
    # Get predictions from all methods
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

def evaluate_improved_methods(llm, data_limit=15):
    """Evaluate improved prompting methods"""
    
    print("🚀 EVALUATING IMPROVED PROMPTING METHODS")
    print("=" * 60)
    
    # Load test data
    data_path = "data/test/data_demo.jsonl"
    test_data = load_evaluation_data(data_path, limit=data_limit)
    
    if not test_data:
        print("❌ Unable to load test data")
        return
    
    print(f"✅ Loaded {len(test_data)} test samples\n")
    
    methods = {
        "improved_zero_shot": improved_zero_shot,
        "improved_few_shot": improved_few_shot, 
        "improved_cot": improved_cot,
        "ensemble": ensemble_classification
    }
    
    results = []
    
    for method_name, method_func in methods.items():
        print(f"🧪 Testing: {method_name.upper()}")
        print("-" * 40)
        
        predictions = []
        true_labels = []
        
        for i, item in enumerate(test_data):
            title = item['title']
            true_label = item['label']
            
            print(f"[{i+1}/{len(test_data)}] {title[:50]}...")
            
            prediction = method_func(llm, title)
            predictions.append(prediction)
            true_labels.append(true_label)
            
            if prediction != -1:
                label_map = {0: "No-clickbait", 1: "Clickbait"}
                label_name = label_map.get(prediction, "Unknown")
                print(f"  → {prediction} - {label_name}")
            else:
                print(f"  → Parse error")
            
            time.sleep(0.5)  # Rate limiting
        
        # Calculate metrics
        valid_indices = [i for i, pred in enumerate(predictions) if pred != -1]
        valid_predictions = [predictions[i] for i in valid_indices]
        valid_true_labels = [true_labels[i] for i in valid_indices]
        
        if len(valid_predictions) > 0:
            accuracy = accuracy_score(valid_true_labels, valid_predictions)
            precision = precision_score(valid_true_labels, valid_predictions, average='weighted', zero_division=0)
            recall = recall_score(valid_true_labels, valid_predictions, average='weighted', zero_division=0)
            f1 = f1_score(valid_true_labels, valid_predictions, average='weighted', zero_division=0)
            
            print(f"\n📊 RESULTS: {method_name.upper()}")
            print(f"   Accuracy: {accuracy:.3f}")
            print(f"   Precision: {precision:.3f}")
            print(f"   Recall: {recall:.3f}")
            print(f"   F1-Score: {f1:.3f}")
            print(f"   Valid samples: {len(valid_predictions)}/{len(predictions)}")
            
            results.append({
                'method': method_name,
                'valid_samples': len(valid_predictions),
                'total_samples': len(predictions),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        
        print("\n" + "=" * 60)
    
    # Compare results
    if results:
        print("\n🏆 FINAL COMPARISON")
        print("=" * 60)
        
        df = pd.DataFrame(results)
        df = df.sort_values('f1_score', ascending=False)
        
        print(df.to_string(index=False, float_format='%.3f'))
        
        best_method = df.iloc[0]
        print(f"\n🥇 BEST METHOD: {best_method['method']}")
        print(f"   📊 F1-Score: {best_method['f1_score']:.3f}")
        print(f"   🎯 Accuracy: {best_method['accuracy']:.3f}")
        
        # Save results
        df.to_csv("improved_evaluation_results.csv", index=False)
        print(f"\n💾 Results saved to: improved_evaluation_results.csv")

def main():
    """Run improved prompting evaluation"""
    
    # Initialize LLM
    print("🚀 Initializing ChatOpenAI...")
    try:
        llm = initialize_llm()
        print("✅ Initialization successful!\n")
    except Exception as e:
        print(f"❌ LLM initialization error: {e}")
        return
    
    # Run evaluation
    evaluate_improved_methods(llm, data_limit=15)

if __name__ == "__main__":
    main() 