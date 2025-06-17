#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Preprocessor cho Clickbait Classification
Chuyển đổi từ format hiện tại sang format yêu cầu trong README
"""

import json
import os
from typing import List, Dict, Any
import random
from tqdm import tqdm

class ClickbaitDataPreprocessor:
    """Preprocessor để chuẩn bị dữ liệu cho training"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        
    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Load JSONL file"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return []
        return data
    
    def save_jsonl(self, data: List[Dict[str, Any]], file_path: str):
        """Save data to JSONL file"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def extract_text_features(self, item: Dict[str, Any]) -> str:
        """Extract and combine relevant text features"""
        text_parts = []
        
        # PostText (nội dung bài đăng)
        post_text = item.get('postText', [])
        if post_text and len(post_text) > 0:
            text_parts.append(post_text[0])
        
        # TargetTitle (tiêu đề bài viết) - quan trọng nhất
        target_title = item.get('targetTitle', '')
        if target_title:
            text_parts.append(target_title)
        
        # TargetDescription (mô tả)
        target_desc = item.get('targetDescription', '')
        if target_desc and len(target_desc) < 200:  # Chỉ lấy mô tả ngắn
            text_parts.append(target_desc[:200])
        
        # Combine with separator
        combined_text = ' [SEP] '.join(text_parts)
        
        # Clean text
        combined_text = combined_text.replace('\n', ' ').replace('\r', ' ')
        combined_text = ' '.join(combined_text.split())  # Remove extra spaces
        
        return combined_text
    
    def convert_dataset(self, dataset_name: str = "all"):
        """Convert dataset to required format"""
        print(f"🔄 Converting {dataset_name} dataset...")
        
        datasets = ['train', 'val', 'test'] if dataset_name == "all" else [dataset_name]
        
        for ds_name in datasets:
            print(f"\n📂 Processing {ds_name} set...")
            
            # Load instances and truth
            instances_path = f"{self.data_dir}/{ds_name}/instances.jsonl"
            truth_path = f"{self.data_dir}/{ds_name}/truth.jsonl"
            
            instances = self.load_jsonl(instances_path)
            truth_data = self.load_jsonl(truth_path)
            
            if not instances or not truth_data:
                print(f"❌ Failed to load {ds_name} data")
                continue
            
            # Create truth mapping
            truth_map = {item['id']: item for item in truth_data}
            
            # Convert format
            converted_data = []
            skipped = 0
            
            for item in tqdm(instances, desc=f"Converting {ds_name}"):
                item_id = item['id']
                
                if item_id not in truth_map:
                    skipped += 1
                    continue
                
                # Extract text
                text = self.extract_text_features(item)
                if len(text.strip()) < 10:  # Skip too short texts
                    skipped += 1
                    continue
                
                # Get label
                truth_item = truth_map[item_id]
                label = 1 if truth_item['truthClass'] == 'clickbait' else 0
                
                # Create new format
                converted_item = {
                    "id": item_id,
                    "text": text,
                    "label": label,
                    "truth_mean": truth_item.get('truthMean', 0.0),
                    "truth_class": truth_item['truthClass']
                }
                
                converted_data.append(converted_item)
            
            # Save converted data
            output_path = f"{self.data_dir}/{ds_name}/data.jsonl"
            self.save_jsonl(converted_data, output_path)
            
            print(f"✅ {ds_name}: {len(converted_data)} samples converted, {skipped} skipped")
            print(f"   Saved to: {output_path}")
            
            # Print sample
            if converted_data:
                sample = converted_data[0]
                print(f"   Sample: {sample['text'][:100]}...")
                print(f"   Label: {sample['label']} ({sample['truth_class']})")
    
    def create_demo_datasets(self, demo_size: int = 100):
        """Create small demo datasets for RTX 3050 testing"""
        print(f"\n🎯 Creating demo datasets ({demo_size} samples each)...")
        
        for ds_name in ['train', 'val', 'test']:
            # Load full dataset
            data_path = f"{self.data_dir}/{ds_name}/data.jsonl"
            if not os.path.exists(data_path):
                print(f"❌ {data_path} not found. Run convert_dataset first.")
                continue
                
            full_data = self.load_jsonl(data_path)
            
            if len(full_data) < demo_size:
                demo_data = full_data
            else:
                # Stratified sampling to maintain class balance
                clickbait_samples = [item for item in full_data if item['label'] == 1]
                no_clickbait_samples = [item for item in full_data if item['label'] == 0]
                
                # Calculate proportional sizes
                total_clickbait = len(clickbait_samples)
                total_no_clickbait = len(no_clickbait_samples)
                total_samples = total_clickbait + total_no_clickbait
                
                clickbait_demo_size = int(demo_size * total_clickbait / total_samples)
                no_clickbait_demo_size = demo_size - clickbait_demo_size
                
                # Random sampling
                random.seed(42)
                demo_clickbait = random.sample(clickbait_samples, 
                                             min(clickbait_demo_size, total_clickbait))
                demo_no_clickbait = random.sample(no_clickbait_samples, 
                                                min(no_clickbait_demo_size, total_no_clickbait))
                
                demo_data = demo_clickbait + demo_no_clickbait
                random.shuffle(demo_data)
            
            # Save demo dataset
            demo_path = f"{self.data_dir}/{ds_name}/data_demo.jsonl"
            self.save_jsonl(demo_data, demo_path)
            
            # Statistics
            clickbait_count = sum(1 for item in demo_data if item['label'] == 1)
            no_clickbait_count = len(demo_data) - clickbait_count
            
            print(f"✅ {ds_name} demo: {len(demo_data)} samples")
            print(f"   - Clickbait: {clickbait_count} ({clickbait_count/len(demo_data)*100:.1f}%)")
            print(f"   - No-clickbait: {no_clickbait_count} ({no_clickbait_count/len(demo_data)*100:.1f}%)")
            print(f"   - Saved to: {demo_path}")
    
    def get_dataset_stats(self):
        """Print dataset statistics"""
        print("\n📊 DATASET STATISTICS")
        print("=" * 50)
        
        for ds_name in ['train', 'val', 'test']:
            # Full dataset
            full_path = f"{self.data_dir}/{ds_name}/data.jsonl"
            demo_path = f"{self.data_dir}/{ds_name}/data_demo.jsonl"
            
            for data_type, path in [("Full", full_path), ("Demo", demo_path)]:
                if os.path.exists(path):
                    data = self.load_jsonl(path)
                    if data:
                        clickbait = sum(1 for item in data if item['label'] == 1)
                        no_clickbait = len(data) - clickbait
                        
                        print(f"\n{ds_name.upper()} {data_type}:")
                        print(f"  - Total: {len(data):,}")
                        print(f"  - Clickbait: {clickbait:,} ({clickbait/len(data)*100:.1f}%)")
                        print(f"  - No-clickbait: {no_clickbait:,} ({no_clickbait/len(data)*100:.1f}%)")
                        
                        # Text length stats
                        text_lengths = [len(item['text']) for item in data]
                        avg_len = sum(text_lengths) / len(text_lengths)
                        max_len = max(text_lengths)
                        min_len = min(text_lengths)
                        
                        print(f"  - Avg text length: {avg_len:.1f} chars")
                        print(f"  - Text length range: {min_len} - {max_len} chars")

def main():
    """Main function"""
    print("🚀 CLICKBAIT DATA PREPROCESSOR")
    print("Converting data to required format...")
    
    preprocessor = ClickbaitDataPreprocessor()
    
    # Convert all datasets
    preprocessor.convert_dataset("all")
    
    # Create demo datasets for RTX 3050
    preprocessor.create_demo_datasets(demo_size=100)
    
    # Show statistics
    preprocessor.get_dataset_stats()
    
    print("\n✅ Data preprocessing completed!")
    print("📁 Files created:")
    print("   - data/{train,val,test}/data.jsonl (full datasets)")
    print("   - data/{train,val,test}/data_demo.jsonl (demo datasets)")

if __name__ == "__main__":
    main() 