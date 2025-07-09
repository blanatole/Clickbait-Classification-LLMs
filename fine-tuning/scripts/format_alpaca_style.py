#!/usr/bin/env python3
"""
Format clickbait dataset to Alpaca-style instruction format
Instruction: Classify & explain
Input: Headline: [text]
Response: 1. Analysis... 2. Label.
"""

import json
import os
from pathlib import Path

def format_to_alpaca_style(data_item):
    """Convert clickbait data to Alpaca-style format"""
    
    # Extract text from original format
    text = data_item['text']
    if '[SEP]' in text:
        headline = text.split('[SEP]')[0].strip().strip('"')
    else:
        headline = text.strip()
    
    label = int(data_item['label'])
    
    # Create instruction
    instruction = "Classify the following news headline as clickbait (1) or not clickbait (0), and provide a detailed analysis explaining your reasoning."
    
    # Create input
    input_text = f"Headline: {headline}"
    
    # Create response with analysis and label
    if label == 1:
        response = f"""1. Analysis: This headline exhibits clickbait characteristics such as emotional language, vague descriptions, curiosity gaps, or sensational claims that are designed to attract clicks rather than inform readers directly.

2. Label: 1 (Clickbait)"""
    else:
        response = f"""1. Analysis: This headline provides clear, factual information without emotional manipulation, vague descriptions, or curiosity gaps. It directly informs readers about the content.

2. Label: 0 (No-clickbait)"""
    
    return {
        "instruction": instruction,
        "input": input_text,
        "output": response,
        "label": label  # Keep original label for reference
    }

def convert_dataset_to_alpaca(input_file, output_file):
    """Convert JSONL dataset to Alpaca format"""
    print(f"Converting {input_file} to Alpaca format...")
    
    converted_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                alpaca_item = format_to_alpaca_style(item)
                converted_data.append(alpaca_item)
                
                if line_num % 1000 == 0:
                    print(f"  Processed {line_num} items...")
                    
            except Exception as e:
                print(f"  Error processing line {line_num}: {e}")
                continue
    
    # Save converted data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(converted_data)} items to {output_file}")
    return len(converted_data)

def main():
    """Convert all dataset splits to Alpaca format"""
    print("🔄 CONVERTING CLICKBAIT DATASET TO ALPACA FORMAT")
    print("=" * 60)
    
    # Define input and output paths
    datasets = {
        "train": {
            "input": "shared/data/train/data.jsonl",
            "output": "fine-tuning/data/alpaca/train.jsonl"
        },
        "validation": {
            "input": "shared/data/val/data.jsonl", 
            "output": "fine-tuning/data/alpaca/val.jsonl"
        },
        "test": {
            "input": "shared/data/test/data.jsonl",
            "output": "fine-tuning/data/alpaca/test.jsonl"
        }
    }
    
    total_converted = 0
    
    for split, paths in datasets.items():
        if os.path.exists(paths["input"]):
            count = convert_dataset_to_alpaca(paths["input"], paths["output"])
            total_converted += count
            print(f"  {split}: {count} samples")
        else:
            print(f"  ⚠️ {split}: File not found - {paths['input']}")
    
    print(f"\n✅ Total converted: {total_converted} samples")
    print(f"📁 Alpaca-format data saved to: fine-tuning/data/alpaca/")
    
    # Show example
    example_file = "fine-tuning/data/alpaca/train.jsonl"
    if os.path.exists(example_file):
        print(f"\n📋 EXAMPLE ALPACA FORMAT:")
        print("=" * 60)
        with open(example_file, 'r', encoding='utf-8') as f:
            example = json.loads(f.readline())
            print(f"Instruction: {example['instruction']}")
            print(f"\nInput: {example['input']}")
            print(f"\nOutput: {example['output']}")

if __name__ == "__main__":
    main() 