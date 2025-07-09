#!/usr/bin/env python3
"""
Test script riêng cho Chain-of-Thought prompting
"""

import os
import time
import jsonlines
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load environment variables
load_dotenv()

def initialize_llm():
    """Initialize Claude model"""
    return ChatOpenAI(
        model="claude-3-7-sonnet-20250219",
        temperature=0.1,
        api_key=os.environ.get("SHUBI_API_KEY"),
        base_url=os.environ.get("SHUBI_URL")
    )

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
        print(f"🔍 Analyzing: {title[:60]}...")
        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()
        
        print(f"📝 CoT Response:\n{response_text[:200]}...")
        
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
            
        print(f"⚠️ Could not parse: {response_text[-100:]}")
        return -1
        
    except Exception as e:
        print(f"❌ Error in CoT: {e}")
        return -1

def load_test_data(limit=10):
    """Load small test dataset"""
    data = []
    try:
        with jsonlines.open("../../shared/data/test/data.jsonl") as reader:
            for i, obj in enumerate(reader):
                if i >= limit:
                    break
                text = obj.get('text', '')
                title = text.split(' [SEP] ')[0] if ' [SEP] ' in text else text
                data.append({
                    'title': title,
                    'label': obj['label']
                })
        return data
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return []

def test_cot_method():
    """Test Chain-of-Thought method với debug info"""
    print("🧠 TESTING CHAIN-OF-THOUGHT PROMPTING")
    print("=" * 50)
    
    # Check environment
    if not os.environ.get("SHUBI_API_KEY"):
        print("❌ SHUBI_API_KEY not found!")
        return
        
    # Initialize LLM
    try:
        llm = initialize_llm()
        print("✅ LLM initialized successfully")
    except Exception as e:
        print(f"❌ LLM initialization failed: {e}")
        return
    
        # Load test data
    test_data = load_test_data(limit=50)  # Test với 50 mẫu để đồng bộ với 2 methods khác
    if not test_data:
        print("❌ No test data loaded")
        return
        
    print(f"✅ Loaded {len(test_data)} test samples\n")
    
    # Test each sample
    predictions = []
    true_labels = []
    
    for i, item in enumerate(test_data):
        print(f"\n--- SAMPLE {i+1}/{len(test_data)} ---")
        title = item['title']
        true_label = item['label']
        
        try:
            prediction = chain_of_thought_prompting(llm, title)
            predictions.append(prediction)
            true_labels.append(true_label)
            
            if prediction != -1:
                label_map = {0: "No-clickbait", 1: "Clickbait"}
                pred_name = label_map.get(prediction, "Unknown")
                true_name = label_map.get(true_label, "Unknown")
                correct = "✅" if prediction == true_label else "❌"
                print(f"🎯 Prediction: {prediction} ({pred_name}) | True: {true_label} ({true_name}) {correct}")
            else:
                print(f"❌ Failed to get prediction")
                
        except Exception as e:
            print(f"❌ Error processing sample: {e}")
            predictions.append(-1)
            true_labels.append(true_label)
            
        # Rate limiting
        time.sleep(1.0)
    
    # Calculate metrics
    print(f"\n{'='*50}")
    print("📊 CHAIN-OF-THOUGHT RESULTS")
    print(f"{'='*50}")
    
    valid_pairs = [(p, t) for p, t in zip(predictions, true_labels) if p != -1]
    
    if valid_pairs:
        valid_predictions, valid_true_labels = zip(*valid_pairs)
        
        accuracy = accuracy_score(valid_true_labels, valid_predictions)
        f1_macro = f1_score(valid_true_labels, valid_predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(valid_true_labels, valid_predictions, average='weighted', zero_division=0)
        
        print(f"📈 Accuracy: {accuracy:.3f}")
        print(f"📈 Macro F1: {f1_macro:.3f}")
        print(f"📈 Weighted F1: {f1_weighted:.3f}")
        print(f"📈 Valid samples: {len(valid_predictions)}/{len(predictions)}")
        
        print(f"\n📊 Classification Report:")
        print(classification_report(valid_true_labels, valid_predictions, 
                                  target_names=['No-clickbait', 'Clickbait'], digits=3))
        
        print(f"\n🎯 TARGET COMPARISON:")
        print(f"   Accuracy:   {accuracy:.3f} (target: 0.74-0.77) {'✅' if 0.74 <= accuracy <= 0.77 else '⚠️'}")
        print(f"   Macro F1:   {f1_macro:.3f} (target: 0.64-0.66) {'✅' if 0.64 <= f1_macro <= 0.66 else '⚠️'}")
        
    else:
        print("❌ No valid predictions generated")

if __name__ == "__main__":
    test_cot_method() 