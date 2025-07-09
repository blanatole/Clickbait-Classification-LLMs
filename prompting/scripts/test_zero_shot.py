#!/usr/bin/env python3
"""
Zero-shot Prompting cho Clickbait Classification
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
        print(f"🔍 Analyzing: {title[:60]}...")
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
        
        # Fallback parsing
        import re
        numbers = re.findall(r'\b[01]\b', response_text)
        if numbers:
            return int(numbers[-1])
            
        print(f"⚠️ Could not parse: {response_text[:100]}...")
        return -1
        
    except Exception as e:
        print(f"❌ Error in Zero-shot: {e}")
        return -1

def load_test_data(limit=50):
    """Load test dataset"""
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

def test_zero_shot():
    """Test Zero-shot method"""
    print("🎯 TESTING ENHANCED ZERO-SHOT PROMPTING")
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
    test_data = load_test_data(limit=50)
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
            prediction = zero_shot_prompting(llm, title)
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
    print("📊 ZERO-SHOT RESULTS")
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
        
        # Save results
        import pandas as pd
        result = {
            "method": "zero_shot",
            "accuracy": round(accuracy, 3),
            "f1_macro": round(f1_macro, 3),
            "f1_weighted": round(f1_weighted, 3),
            "valid_samples": len(valid_predictions),
            "total_samples": len(predictions)
        }
        
        df = pd.DataFrame([result])
        df.to_csv("zero_shot_results.csv", index=False)
        print(f"\n💾 Results saved to: zero_shot_results.csv")
        
    else:
        print("❌ No valid predictions generated")

if __name__ == "__main__":
    test_zero_shot() 