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
        model="claude-3-7-sonnet-20250219",  # Working Claude model
        temperature=0,
        api_key=os.environ.get("SHUBI_API_KEY"), 
        base_url=os.environ.get("SHUBI_URL")
    )
    return llm

# ==================== BASIC PROMPTING METHODS ====================

def zero_shot_prompting(llm, title):
    """Enhanced zero-shot prompting with comprehensive guidelines"""
    prompt = f"""You are an expert content analyst specializing in clickbait detection for digital media.

CLASSIFICATION TASK: Determine if this headline is clickbait (1) or legitimate news (0).

CLICKBAIT CHARACTERISTICS (Label: 1):
- Emotional manipulation: "SHOCKING", "AMAZING", "UNBELIEVABLE", "WOW"
- Curiosity gaps: "You won't believe what happened", "This will surprise you"
- Vague promises: "This simple trick", "One secret that", "The reason will shock you"
- Personal hooks: "You need to see this", "This will change your life"
- Exaggeration: "Everyone is talking about", "Nobody expected this"
- Withholding information: Creating questions without providing answers

NO-CLICKBAIT CHARACTERISTICS (Label: 0):
- Factual reporting: "Company reports earnings", "Study finds correlation"
- Specific information: "New law reduces taxes by 2%", "Earthquake magnitude 6.5"
- Neutral language: Professional, objective tone without emotional manipulation
- Complete information: Headlines that tell the story without tricks

HEADLINE TO ANALYZE: "{title}"

EVALUATION PROCESS:
1. Check for emotional manipulation language
2. Assess information completeness vs. curiosity gaps
3. Evaluate intent: inform vs. attract clicks through manipulation

ANSWER: [0 for no-clickbait] or [1 for clickbait]"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()
        
        # Enhanced parsing
        if "answer:" in response_text.lower():
            lines = response_text.split('\n')
            for line in lines:
                if "answer:" in line.lower():
                    if '0' in line:
                        return 0
                    elif '1' in line:
                        return 1
        
        return parse_prompt_response(response_text, "zero_shot")
    except Exception:
        return -1

def few_shot_prompting(llm, title):
    """Enhanced few-shot prompting with balanced, diverse examples"""
    prompt = f"""You are a clickbait detection expert. Learn from these carefully selected examples:

=== TRAINING EXAMPLES ===

Headline: "Federal Reserve raises interest rates by 0.5 percentage points"
Label: 0 (No-clickbait)
Analysis: Specific economic news with precise data, neutral reporting tone

Headline: "You'll NEVER believe what this dog did when his owner came home!"
Label: 1 (Clickbait)
Analysis: Emotional "NEVER believe", complete information withholding, curiosity hook

Headline: "Apple announces quarterly earnings: Revenue up 8% year-over-year"
Label: 0 (No-clickbait)
Analysis: Corporate news with specific financial data, straightforward reporting

Headline: "This ONE simple trick doctors don't want you to know about weight loss"
Label: 1 (Clickbait)
Analysis: "ONE simple trick" formula, conspiracy angle, vague health claims

Headline: "Study: Mediterranean diet reduces heart disease risk by 30%"
Label: 0 (No-clickbait)
Analysis: Research-based headline with specific statistical findings

Headline: "Local mom discovers WEIRD secret that celebrities use to stay young"
Label: 1 (Clickbait)
Analysis: "WEIRD secret" mystery, celebrity angle, vague promise

Headline: "Breaking: 7.2 magnitude earthquake strikes off Japan coast"
Label: 0 (No-clickbait)
Analysis: Breaking news with specific location and magnitude data

Headline: "These 7 photos will restore your faith in humanity (number 4 is incredible!)"
Label: 1 (Clickbait)
Analysis: Listicle format, emotional manipulation, curiosity hook about "number 4"

=== YOUR TASK ===

Based on the pattern recognition from these examples, classify:

Headline: "{title}"

Consider:
- Language tone (neutral vs emotional manipulation)
- Information completeness (specific facts vs curiosity gaps)
- Intent (inform vs attract clicks through tricks)

ANSWER: [0 for no-clickbait] or [1 for clickbait]"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()
        
        # Enhanced parsing
        if "answer:" in response_text.lower():
            lines = response_text.split('\n')
            for line in lines:
                if "answer:" in line.lower():
                    if '0' in line:
                        return 0
                    elif '1' in line:
                        return 1
        
        return parse_prompt_response(response_text, "few_shot")
    except Exception:
        return -1

def chain_of_thought_prompting(llm, title):
    """Advanced Chain-of-thought prompting with systematic analysis"""
    prompt = f"""Perform a comprehensive analysis to determine if this headline is clickbait. Use systematic reasoning:

HEADLINE: "{title}"

=== SYSTEMATIC ANALYSIS ===

STEP 1: LANGUAGE ANALYSIS
Examine for clickbait language patterns:
- Emotional manipulation words: "SHOCKING", "AMAZING", "INCREDIBLE", "UNBELIEVABLE"
- Curiosity gap phrases: "You won't believe", "What happens next", "The reason will shock you"
- Vague promise words: "simple trick", "secret that", "one weird tip"
- Personal hooks: "You need to see", "This will change your life"
- Exaggerated claims: "everyone", "nobody", "always", "never"

STEP 2: INFORMATION COMPLETENESS
Evaluate information delivery:
- Does it provide specific facts and data?
- Does it answer the "who, what, when, where, why"?
- Does it withhold key information to create curiosity?
- Is the headline complete or does it require clicking to understand?

STEP 3: INTENT ASSESSMENT
Determine the primary purpose:
- Is it designed to inform readers with facts?
- Is it designed to manipulate emotions for clicks?
- Does it use professional journalism language?
- Does it feel like entertainment/sensational content?

STEP 4: JOURNALISTIC STANDARDS
Check against news standards:
- Would this appear in a reputable newspaper?
- Does it maintain objectivity and neutrality?
- Is it sensationalized beyond the actual story?
- Does it respect reader intelligence?

STEP 5: FINAL REASONING
Synthesizing all analysis points:
- Weigh the evidence from each step
- Consider overall impression and intent
- Make evidence-based classification

=== CONCLUSION ===
Based on this systematic analysis, this headline is:
FINAL ANSWER: [0 for no-clickbait] or [1 for clickbait]

Rationale: [Brief explanation of the key factors that led to this classification]"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()
        
        # Enhanced parsing for CoT
        if "final answer:" in response_text.lower():
            lines = response_text.split('\n')
            for line in lines:
                if "final answer:" in line.lower():
                    if '0' in line:
                        return 0
                    elif '1' in line:
                        return 1
        
        # Fallback to ANSWER: format
        if 'ANSWER:' in response_text.upper():
            lines = response_text.split('\n')
            for line in reversed(lines):
                if 'ANSWER:' in line.upper():
                    if '0' in line:
                        return 0
                    elif '1' in line:
                        return 1
                        
        # Last resort parsing
        import re
        numbers = re.findall(r'\b[01]\b', response_text)
        if numbers:
            return int(numbers[-1])
            
        return -1
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

def load_evaluation_data(file_path, limit=50):
    """Load data from jsonl file for evaluation"""
    data = []
    try:
        with jsonlines.open(file_path) as reader:
            for i, obj in enumerate(reader):
                if i >= limit:
                    break
                # Extract title from text field 
                text = obj.get('text', '')
                title = text.split(' [SEP] ')[0] if ' [SEP] ' in text else text
                data.append({
                    'title': title, 
                    'label': obj['label']
                })
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
    """Calculate evaluation metrics including Macro F1"""
    # Filter out failed predictions (-1)
    valid_pairs = [(pred, true) for pred, true in zip(predictions, true_labels) if pred != -1]
    
    if not valid_pairs:
        print(f"❌ No valid predictions for {method_name}")
        return None
    
    valid_predictions, valid_true_labels = zip(*valid_pairs)
    
    # Calculate metrics
    accuracy = accuracy_score(valid_true_labels, valid_predictions)
    
    # Calculate both Macro and Weighted metrics
    precision_macro = precision_score(valid_true_labels, valid_predictions, average='macro', zero_division=0)
    recall_macro = recall_score(valid_true_labels, valid_predictions, average='macro', zero_division=0)
    f1_macro = f1_score(valid_true_labels, valid_predictions, average='macro', zero_division=0)
    
    precision_weighted = precision_score(valid_true_labels, valid_predictions, average='weighted', zero_division=0)
    recall_weighted = recall_score(valid_true_labels, valid_predictions, average='weighted', zero_division=0)
    f1_weighted = f1_score(valid_true_labels, valid_predictions, average='weighted', zero_division=0)
    
    # Print detailed classification report
    from sklearn.metrics import classification_report, confusion_matrix
    print(f"\n📊 Classification Report for {method_name}:")
    print(classification_report(valid_true_labels, valid_predictions, 
                              target_names=['No-clickbait', 'Clickbait'], digits=3))
    
    print(f"📈 Confusion Matrix:")
    print(confusion_matrix(valid_true_labels, valid_predictions))
    
    # Compare with targets
    print(f"\n🎯 TARGET vs ACTUAL COMPARISON:")
    print(f"   Accuracy:   {accuracy:.3f} (target: 0.74-0.77) {'✅' if 0.74 <= accuracy <= 0.77 else '⚠️'}")
    print(f"   Macro F1:   {f1_macro:.3f} (target: 0.64-0.66) {'✅' if 0.64 <= f1_macro <= 0.66 else '⚠️'}")
    print(f"   Weighted F1: {f1_weighted:.3f}")
    
    return {
        "method": method_name,
        "accuracy": round(accuracy, 3),
        "precision_macro": round(precision_macro, 3),
        "recall_macro": round(recall_macro, 3),
        "f1_macro": round(f1_macro, 3),
        "precision_weighted": round(precision_weighted, 3),
        "recall_weighted": round(recall_weighted, 3),
        "f1_weighted": round(f1_weighted, 3),
        "valid_samples": len(valid_predictions),
        "total_samples": len(predictions)
    }

def run_evaluation_comparison(llm, data_limit=15):
    """Run comparison of all prompting methods"""
    print("🚀 EVALUATING ALL PROMPTING METHODS")
    print("=" * 60)
    
    # Load test data
    data_path = "../../shared/data/test/data.jsonl"
    test_data = load_evaluation_data(data_path, limit=data_limit)
    
    if not test_data:
        print("❌ Unable to load test data")
        return
    
    print(f"✅ Loaded {len(test_data)} test samples\n")
    
    # Test 3 methods chính để có kết quả nhanh
    methods = [
        "zero_shot", "few_shot", "cot"
    ]
    
    results = []
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"🧪 TESTING: {method.upper()}")
        print(f"{'='*50}")
        
        try:
            print(f"🔄 Starting {method} evaluation...")
            predictions, true_labels = evaluate_prompting_method(
                llm, test_data, method, delay=1.0
            )
            metrics = calculate_metrics(predictions, true_labels, method)
            if metrics:
                results.append(metrics)
                print(f"✅ {method} completed successfully!")
            else:
                print(f"⚠️ {method} failed to generate valid metrics")
                
        except KeyboardInterrupt:
            print("\n⚠️ Stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in {method} evaluation: {e}")
            print(f"🔄 Continuing with next method...")
            continue
    
    # Display comparison results
    if results:
        print("\n" + "=" * 70)
        print("           📈 FINAL EVALUATION COMPARISON")
        print("=" * 70)
        
        df = pd.DataFrame(results)
        
        # Sort by F1-Score
        df = df.sort_values('f1_macro', ascending=False) # Changed to f1_macro for consistency
        
        print(df.to_string(index=False, float_format='%.3f'))
        
        # Find best method
        best_method = df.iloc[0]
        print(f"\n🏆 BEST METHOD: {best_method['method']}")
        print(f"   📊 F1-Score: {best_method['f1_macro']:.3f}") # Changed to f1_macro
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
            print(f"   {row['method']}: {row['f1_macro']:.3f} F1 ({success_rate:.1f}% success rate)") # Changed to f1_macro
        
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
    
    # Run evaluation - test thử với 50 mẫu trước
    run_evaluation_comparison(llm, data_limit=50)  # Test với 50 samples để đảm bảo hoạt động

if __name__ == "__main__":
    main() 