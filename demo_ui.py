#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Demo UI cho Clickbait Classification
Sử dụng gradio để tạo web interface đơn giản
"""

import os
import sys
import gradio as gr
import torch
from typing import Tuple, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.inference import ClickbaitClassifier

# Global variables
classifier = None
model_loaded = False

def load_model(model_path: str = "outputs/tinybert_rtx3050_demo"):
    """Load the trained model"""
    global classifier, model_loaded
    
    try:
        if os.path.exists(model_path):
            classifier = ClickbaitClassifier(model_path)
            model_loaded = True
            return f"✅ Model loaded successfully from {model_path}"
        else:
            return f"❌ Model not found at {model_path}. Please train a model first."
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"

def predict_clickbait(text: str) -> Tuple[str, str, str]:
    """Predict if text is clickbait"""
    global classifier, model_loaded
    
    if not model_loaded or classifier is None:
        return "❌ Model not loaded", "", ""
    
    if not text.strip():
        return "Please enter some text", "", ""
    
    try:
        result = classifier.predict(text, return_probabilities=True)
        
        # Main prediction
        prediction = result['predicted_class'].upper()
        confidence = f"{result['confidence']:.3f}"
        
        # Detailed probabilities
        prob_no_clickbait = result['probabilities']['no-clickbait']
        prob_clickbait = result['probabilities']['clickbait']
        
        probabilities = f"No-clickbait: {prob_no_clickbait:.3f}\nClickbait: {prob_clickbait:.3f}"
        
        # Visual indicator
        if result['predicted_class'] == 'clickbait':
            if result['confidence'] > 0.8:
                indicator = "🔴 Strong clickbait signal"
            elif result['confidence'] > 0.6:
                indicator = "🟡 Moderate clickbait signal"
            else:
                indicator = "🟢 Weak clickbait signal"
        else:
            if result['confidence'] > 0.8:
                indicator = "✅ Clearly not clickbait"
            elif result['confidence'] > 0.6:
                indicator = "🟡 Probably not clickbait"
            else:
                indicator = "🔴 Uncertain classification"
        
        return f"{prediction} (Confidence: {confidence})", probabilities, indicator
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "", ""

def batch_predict(text_list: str) -> str:
    """Predict multiple texts (one per line)"""
    global classifier, model_loaded
    
    if not model_loaded or classifier is None:
        return "❌ Model not loaded"
    
    texts = [line.strip() for line in text_list.split('\n') if line.strip()]
    
    if not texts:
        return "Please enter some texts (one per line)"
    
    try:
        results = classifier.predict_batch(texts)
        
        output = []
        for i, result in enumerate(results):
            text = result['text'][:60] + "..." if len(result['text']) > 60 else result['text']
            pred = result['predicted_class']
            conf = result['confidence']
            output.append(f"{i+1}. {pred.upper()} ({conf:.3f}) | {text}")
        
        # Summary
        clickbait_count = sum(1 for r in results if r['predicted_class'] == 'clickbait')
        total = len(results)
        
        summary = f"\n--- SUMMARY ---\n"
        summary += f"Total: {total}, Clickbait: {clickbait_count} ({clickbait_count/total*100:.1f}%), "
        summary += f"No-clickbait: {total-clickbait_count} ({(total-clickbait_count)/total*100:.1f}%)"
        
        return '\n'.join(output) + summary
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

def create_demo_interface():
    """Create Gradio interface"""
    
    # Custom CSS
    css = """
    .gradio-container {
        max-width: 800px !important;
        margin: auto !important;
    }
    """
    
    with gr.Blocks(css=css, title="Clickbait Classification Demo") as demo:
        
        gr.Markdown("""
        # 🎯 Clickbait Classification Demo
        
        **Phát hiện tiêu đề clickbait sử dụng AI**
        
        Được phát triển cho đồ án KLTN - Tối ưu cho RTX 3050
        """)
        
        # Model loading section
        with gr.Row():
            model_path_input = gr.Textbox(
                value="outputs/tinybert_rtx3050_demo",
                label="Model Path",
                placeholder="Enter path to trained model"
            )
            load_btn = gr.Button("Load Model", variant="primary")
        
        model_status = gr.Textbox(label="Model Status", interactive=False)
        
        # Single prediction section
        gr.Markdown("## 🔍 Single Text Analysis")
        
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Enter text to analyze",
                    placeholder="Enter a headline or text to check if it's clickbait...",
                    lines=3
                )
                predict_btn = gr.Button("Analyze Text", variant="secondary")
            
            with gr.Column(scale=1):
                prediction_output = gr.Textbox(label="Prediction", interactive=False)
                probabilities_output = gr.Textbox(label="Probabilities", interactive=False)
                indicator_output = gr.Textbox(label="Analysis", interactive=False)
        
        # Batch prediction section
        gr.Markdown("## 📊 Batch Analysis")
        
        with gr.Row():
            with gr.Column():
                batch_input = gr.Textbox(
                    label="Enter multiple texts (one per line)",
                    placeholder="Enter multiple headlines or texts, one per line...",
                    lines=8
                )
                batch_btn = gr.Button("Analyze Batch", variant="secondary")
            
            with gr.Column():
                batch_output = gr.Textbox(
                    label="Batch Results",
                    lines=12,
                    interactive=False
                )
        
        # Example section
        gr.Markdown("## 📝 Examples")
        
        example_texts = [
            "You Won't Believe What Happened Next!",
            "Study Shows Correlation Between Exercise and Mental Health",
            "10 Shocking Facts That Will Change Your Life Forever",
            "Federal Reserve Raises Interest Rates by 0.25%",
            "This Simple Trick Will Save You Thousands of Dollars"
        ]
        
        gr.Examples(
            examples=[[text] for text in example_texts],
            inputs=[text_input],
            outputs=[prediction_output, probabilities_output, indicator_output],
            fn=predict_clickbait,
            cache_examples=False
        )
        
        # Event handlers
        load_btn.click(
            fn=load_model,
            inputs=[model_path_input],
            outputs=[model_status]
        )
        
        predict_btn.click(
            fn=predict_clickbait,
            inputs=[text_input],
            outputs=[prediction_output, probabilities_output, indicator_output]
        )
        
        batch_btn.click(
            fn=batch_predict,
            inputs=[batch_input],
            outputs=[batch_output]
        )
        
        # Auto-load model on startup
        demo.load(
            fn=lambda: load_model("outputs/tinybert_rtx3050_demo"),
            outputs=[model_status]
        )
        
        gr.Markdown("""
        ---
        **Note:** 
        - 🔴 Higher confidence means stronger signal
        - 🟡 Medium confidence suggests uncertainty  
        - ✅ Lower confidence but correct classification
        - This is a demo model trained on limited data
        """)
    
    return demo

def main():
    """Main function to run the demo"""
    print("🚀 Starting Clickbait Classification Demo...")
    
    # Check if we have a trained model
    default_model_path = "outputs/tinybert_rtx3050_demo"
    if not os.path.exists(default_model_path):
        print(f"⚠️ Default model not found at {default_model_path}")
        print("Please train a model first using:")
        print("python src/train_demo.py --model tinybert --quick_test")
        print("\nStarting demo anyway (you can load a model manually)...")
    
    # Create and launch interface
    demo = create_demo_interface()
    
    print("🌐 Demo will be available at: http://localhost:7860")
    print("Press Ctrl+C to stop the demo")
    
    demo.launch(
        server_name="0.0.0.0",  # Allow access from other devices
        server_port=7860,
        share=False,  # Set to True to create public link
        debug=False
    )

if __name__ == "__main__":
    main() 