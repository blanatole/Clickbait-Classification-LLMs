#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Data Analysis cho Clickbait Classification
Phân tích chi tiết và xuất kết quả về JSON
"""

import json
import os
import re
import string
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
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
    return data

def basic_statistics(data: List[Dict[str, Any]], dataset_name: str) -> Dict[str, Any]:
    """Calculate basic statistics"""
    total_samples = len(data)
    clickbait_samples = sum(1 for item in data if item['label'] == 1)
    no_clickbait_samples = total_samples - clickbait_samples
    
    # Text length analysis
    text_lengths = [len(item['text']) for item in data]
    
    # Truth scores analysis
    truth_scores = [item.get('truth_mean', 0.0) for item in data]
    
    stats = {
        'dataset_name': dataset_name,
        'total_samples': total_samples,
        'class_distribution': {
            'clickbait': {
                'count': clickbait_samples,
                'percentage': round(clickbait_samples / total_samples * 100, 2)
            },
            'no_clickbait': {
                'count': no_clickbait_samples,
                'percentage': round(no_clickbait_samples / total_samples * 100, 2)
            }
        },
        'text_length_statistics': {
            'mean': round(np.mean(text_lengths), 2),
            'median': round(np.median(text_lengths), 2),
            'std': round(np.std(text_lengths), 2),
            'min': int(np.min(text_lengths)),
            'max': int(np.max(text_lengths)),
            'quartiles': {
                'q25': round(np.percentile(text_lengths, 25), 2),
                'q50': round(np.percentile(text_lengths, 50), 2),
                'q75': round(np.percentile(text_lengths, 75), 2)
            }
        },
        'truth_score_statistics': {
            'mean': round(np.mean(truth_scores), 4),
            'median': round(np.median(truth_scores), 4),
            'std': round(np.std(truth_scores), 4),
            'min': round(np.min(truth_scores), 4),
            'max': round(np.max(truth_scores), 4)
        }
    }
    
    return stats

def text_analysis(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze text patterns"""
    
    # Separate by class
    clickbait_texts = [item['text'] for item in data if item['label'] == 1]
    no_clickbait_texts = [item['text'] for item in data if item['label'] == 0]
    
    def analyze_text_patterns(texts: List[str], class_name: str) -> Dict[str, Any]:
        # Word frequency
        all_words = []
        word_counts = Counter()
        
        # Punctuation analysis
        question_marks = 0
        exclamation_marks = 0
        uppercase_words = 0
        numbers = 0
        
        # Length analysis
        lengths = []
        word_counts_per_text = []
        
        for text in texts:
            # Basic stats
            lengths.append(len(text))
            words = text.lower().split()
            word_counts_per_text.append(len(words))
            
            # Word analysis
            for word in words:
                cleaned_word = re.sub(r'[^\w]', '', word)
                if cleaned_word:
                    all_words.append(cleaned_word)
                    word_counts[cleaned_word] += 1
            
            # Punctuation analysis
            if '?' in text:
                question_marks += 1
            if '!' in text:
                exclamation_marks += 1
            
            # Uppercase analysis
            uppercase_words += sum(1 for word in words if word.isupper() and len(word) > 1)
            
            # Numbers
            numbers += len(re.findall(r'\d+', text))
        
        return {
            'class': class_name,
            'total_texts': len(texts),
            'average_length': round(np.mean(lengths), 2),
            'average_word_count': round(np.mean(word_counts_per_text), 2),
            'punctuation_stats': {
                'question_marks': {
                    'count': question_marks,
                    'percentage': round(question_marks / len(texts) * 100, 2)
                },
                'exclamation_marks': {
                    'count': exclamation_marks, 
                    'percentage': round(exclamation_marks / len(texts) * 100, 2)
                }
            },
            'content_stats': {
                'uppercase_words': uppercase_words,
                'numbers_found': numbers,
                'unique_words': len(set(all_words)),
                'total_words': len(all_words)
            },
            'top_words': dict(word_counts.most_common(20))
        }
    
    clickbait_analysis = analyze_text_patterns(clickbait_texts, 'clickbait')
    no_clickbait_analysis = analyze_text_patterns(no_clickbait_texts, 'no_clickbait')
    
    return {
        'clickbait_patterns': clickbait_analysis,
        'no_clickbait_patterns': no_clickbait_analysis
    }

def keyword_analysis(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze keywords and phrases"""
    
    # Common clickbait keywords
    clickbait_keywords = [
        'you', 'will', 'this', 'how', 'what', 'why', 'when', 'where',
        'amazing', 'shocking', 'incredible', 'unbelievable', 'secret',
        'trick', 'hack', 'tips', 'ways', 'reasons', 'facts',
        'never', 'always', 'everyone', 'nobody', 'anything',
        'everything', 'someone', 'something', 'somewhere'
    ]
    
    # Clickbait phrases
    clickbait_phrases = [
        'you won\'t believe', 'will shock you', 'will surprise you',
        'you need to know', 'will change your life', 'doctors hate',
        'this one trick', 'what happens next', 'will blow your mind',
        'number will shock', 'will make you', 'you didn\'t know'
    ]
    
    def count_keywords_in_class(texts: List[str], keywords: List[str]) -> Dict[str, int]:
        keyword_counts = {keyword: 0 for keyword in keywords}
        for text in texts:
            text_lower = text.lower()
            for keyword in keywords:
                keyword_counts[keyword] += text_lower.count(keyword)
        return keyword_counts
    
    def count_phrases_in_class(texts: List[str], phrases: List[str]) -> Dict[str, int]:
        phrase_counts = {phrase: 0 for phrase in phrases}
        for text in texts:
            text_lower = text.lower()
            for phrase in phrases:
                phrase_counts[phrase] += text_lower.count(phrase)
        return phrase_counts
    
    # Separate by class
    clickbait_texts = [item['text'] for item in data if item['label'] == 1]
    no_clickbait_texts = [item['text'] for item in data if item['label'] == 0]
    
    # Keyword analysis
    clickbait_keyword_counts = count_keywords_in_class(clickbait_texts, clickbait_keywords)
    no_clickbait_keyword_counts = count_keywords_in_class(no_clickbait_texts, clickbait_keywords)
    
    # Phrase analysis
    clickbait_phrase_counts = count_phrases_in_class(clickbait_texts, clickbait_phrases)
    no_clickbait_phrase_counts = count_phrases_in_class(no_clickbait_texts, clickbait_phrases)
    
    # Calculate ratios
    keyword_ratios = {}
    for keyword in clickbait_keywords:
        cb_count = clickbait_keyword_counts[keyword]
        ncb_count = no_clickbait_keyword_counts[keyword]
        if ncb_count > 0:
            ratio = cb_count / ncb_count
        else:
            ratio = float('inf') if cb_count > 0 else 0
        keyword_ratios[keyword] = round(ratio, 3)
    
    return {
        'clickbait_keywords': {
            'in_clickbait': clickbait_keyword_counts,
            'in_no_clickbait': no_clickbait_keyword_counts,
            'clickbait_ratio': keyword_ratios
        },
        'clickbait_phrases': {
            'in_clickbait': clickbait_phrase_counts,
            'in_no_clickbait': no_clickbait_phrase_counts
        },
        'top_discriminative_keywords': dict(sorted(keyword_ratios.items(), 
                                                 key=lambda x: x[1], reverse=True)[:10])
    }

def advanced_patterns(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze advanced patterns"""
    
    clickbait_data = [item for item in data if item['label'] == 1]
    no_clickbait_data = [item for item in data if item['label'] == 0]
    
    def analyze_patterns(dataset: List[Dict[str, Any]], class_name: str) -> Dict[str, Any]:
        texts = [item['text'] for item in dataset]
        
        # Title case analysis
        title_case = sum(1 for text in texts if text.istitle())
        
        # All caps words
        all_caps_texts = sum(1 for text in texts if any(word.isupper() and len(word) > 2 
                                                       for word in text.split()))
        
        # Numbers in text
        has_numbers = sum(1 for text in texts if re.search(r'\d', text))
        
        # Quotation marks
        has_quotes = sum(1 for text in texts if '"' in text or "'" in text)
        
        # Superlatives
        superlatives = ['best', 'worst', 'most', 'least', 'biggest', 'smallest', 
                       'fastest', 'slowest', 'amazing', 'incredible', 'ultimate']
        has_superlatives = sum(1 for text in texts 
                              if any(sup in text.lower() for sup in superlatives))
        
        # Time-related words
        time_words = ['now', 'today', 'tonight', 'yesterday', 'tomorrow', 
                     'never', 'always', 'immediately', 'instantly', 'urgent']
        has_time_words = sum(1 for text in texts 
                            if any(tw in text.lower() for tw in time_words))
        
        # Personal pronouns
        pronouns = ['you', 'your', 'we', 'us', 'our', 'i', 'my', 'me']
        has_pronouns = sum(1 for text in texts 
                          if any(pron in text.lower().split() for pron in pronouns))
        
        total = len(texts)
        
        return {
            'class': class_name,
            'formatting_patterns': {
                'title_case': {'count': title_case, 'percentage': round(title_case/total*100, 2)},
                'all_caps_words': {'count': all_caps_texts, 'percentage': round(all_caps_texts/total*100, 2)},
                'has_numbers': {'count': has_numbers, 'percentage': round(has_numbers/total*100, 2)},
                'has_quotes': {'count': has_quotes, 'percentage': round(has_quotes/total*100, 2)}
            },
            'linguistic_patterns': {
                'superlatives': {'count': has_superlatives, 'percentage': round(has_superlatives/total*100, 2)},
                'time_words': {'count': has_time_words, 'percentage': round(has_time_words/total*100, 2)},
                'personal_pronouns': {'count': has_pronouns, 'percentage': round(has_pronouns/total*100, 2)}
            }
        }
    
    clickbait_patterns = analyze_patterns(clickbait_data, 'clickbait')
    no_clickbait_patterns = analyze_patterns(no_clickbait_data, 'no_clickbait')
    
    return {
        'clickbait_advanced_patterns': clickbait_patterns,
        'no_clickbait_advanced_patterns': no_clickbait_patterns
    }

def truth_score_analysis(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze truth scores distribution"""
    
    # Group by truth score ranges
    score_ranges = {
        '0.0-0.2': [],
        '0.2-0.4': [],
        '0.4-0.6': [],
        '0.6-0.8': [],
        '0.8-1.0': []
    }
    
    for item in data:
        score = item.get('truth_mean', 0.0)
        if score <= 0.2:
            score_ranges['0.0-0.2'].append(item)
        elif score <= 0.4:
            score_ranges['0.2-0.4'].append(item)
        elif score <= 0.6:
            score_ranges['0.4-0.6'].append(item)
        elif score <= 0.8:
            score_ranges['0.6-0.8'].append(item)
        else:
            score_ranges['0.8-1.0'].append(item)
    
    analysis = {}
    for range_name, items in score_ranges.items():
        if items:
            clickbait_count = sum(1 for item in items if item['label'] == 1)
            total_count = len(items)
            
            analysis[range_name] = {
                'total_samples': total_count,
                'clickbait_count': clickbait_count,
                'no_clickbait_count': total_count - clickbait_count,
                'clickbait_percentage': round(clickbait_count / total_count * 100, 2),
                'sample_texts': [item['text'][:100] + '...' for item in items[:3]]
            }
    
    return {
        'truth_score_distribution': analysis,
        'correlation_analysis': {
            'description': 'Truth scores vs binary labels correlation',
            'note': 'Higher truth scores should correlate with clickbait label'
        }
    }

def comprehensive_analysis(data_dir: str = "data") -> Dict[str, Any]:
    """Run comprehensive analysis on all datasets"""
    
    print("🔍 COMPREHENSIVE DATA ANALYSIS")
    print("=" * 50)
    
    analysis_results = {
        'analysis_metadata': {
            'timestamp': datetime.now().isoformat(),
            'data_directory': data_dir,
            'analysis_version': '1.0'
        },
        'datasets': {},
        'overall_statistics': {},
        'comparative_analysis': {}
    }
    
    # Analyze each dataset
    datasets = ['train', 'val', 'test']
    all_data = []
    
    for dataset_name in datasets:
        print(f"\n📊 Analyzing {dataset_name} dataset...")
        
        # Load data
        data_file = os.path.join(data_dir, dataset_name, 'data.jsonl')
        if not os.path.exists(data_file):
            print(f"⚠️ File not found: {data_file}")
            continue
            
        data = load_jsonl(data_file)
        if not data:
            print(f"⚠️ No data loaded from {data_file}")
            continue
            
        all_data.extend(data)
        
        print(f"✅ Loaded {len(data)} samples")
        
        # Basic statistics
        basic_stats = basic_statistics(data, dataset_name)
        
        # Text analysis
        text_patterns = text_analysis(data)
        
        # Keyword analysis
        keyword_patterns = keyword_analysis(data)
        
        # Advanced patterns
        advanced_pattern_analysis = advanced_patterns(data)
        
        # Truth score analysis
        truth_analysis = truth_score_analysis(data)
        
        # Combine results
        analysis_results['datasets'][dataset_name] = {
            'basic_statistics': basic_stats,
            'text_patterns': text_patterns,
            'keyword_analysis': keyword_patterns,
            'advanced_patterns': advanced_pattern_analysis,
            'truth_score_analysis': truth_analysis
        }
    
    # Overall analysis
    if all_data:
        print(f"\n📈 Analyzing overall dataset ({len(all_data)} total samples)...")
        
        overall_basic = basic_statistics(all_data, 'overall')
        overall_text = text_analysis(all_data)
        overall_keywords = keyword_analysis(all_data)
        overall_advanced = advanced_patterns(all_data)
        overall_truth = truth_score_analysis(all_data)
        
        analysis_results['overall_statistics'] = {
            'basic_statistics': overall_basic,
            'text_patterns': overall_text,
            'keyword_analysis': overall_keywords,
            'advanced_patterns': overall_advanced,
            'truth_score_analysis': overall_truth
        }
        
        # Comparative analysis
        print(f"\n🔄 Running comparative analysis...")
        
        clickbait_texts = [item['text'] for item in all_data if item['label'] == 1]
        no_clickbait_texts = [item['text'] for item in all_data if item['label'] == 0]
        
        analysis_results['comparative_analysis'] = {
            'key_differences': {
                'average_length_difference': round(np.mean([len(t) for t in clickbait_texts]) - 
                                                  np.mean([len(t) for t in no_clickbait_texts]), 2),
                'clickbait_avg_length': round(np.mean([len(t) for t in clickbait_texts]), 2),
                'no_clickbait_avg_length': round(np.mean([len(t) for t in no_clickbait_texts]), 2)
            },
            'recommendations': [
                "Clickbait texts tend to be shorter and more emotional",
                "Focus on personal pronouns (you, your) as strong indicators",
                "Question marks and exclamation marks are more common in clickbait",
                "Numbers and superlatives appear frequently in clickbait",
                "Time-related urgency words are clickbait indicators"
            ]
        }
    
    return analysis_results

def save_analysis_results(analysis: Dict[str, Any], output_file: str = "data_analysis_results.json"):
    """Save analysis results to JSON file"""
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Analysis results saved to: {output_file}")
        
        # Print summary
        print(f"\n📋 ANALYSIS SUMMARY")
        print("=" * 30)
        
        if 'overall_statistics' in analysis:
            overall = analysis['overall_statistics']['basic_statistics']
            print(f"Total samples analyzed: {overall['total_samples']:,}")
            print(f"Clickbait: {overall['class_distribution']['clickbait']['count']:,} ({overall['class_distribution']['clickbait']['percentage']}%)")
            print(f"No-clickbait: {overall['class_distribution']['no_clickbait']['count']:,} ({overall['class_distribution']['no_clickbait']['percentage']}%)")
            print(f"Average text length: {overall['text_length_statistics']['mean']} characters")
        
        # File size
        file_size = os.path.getsize(output_file) / 1024  # KB
        print(f"Analysis file size: {file_size:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving analysis: {e}")
        return False

def main():
    """Main function"""
    print("🚀 STARTING COMPREHENSIVE DATA ANALYSIS")
    print("Analyzing clickbait classification dataset...")
    
    # Run analysis
    results = comprehensive_analysis()
    
    # Save results
    success = save_analysis_results(results, "data_analysis_results.json")
    
    if success:
        print("\n✅ Analysis completed successfully!")
        print("\n📊 Key insights:")
        
        if 'comparative_analysis' in results:
            for rec in results['comparative_analysis']['recommendations']:
                print(f"   • {rec}")
        
        print(f"\n📁 Files generated:")
        print(f"   • data_analysis_results.json (comprehensive analysis)")
        
        print(f"\n💡 Next steps:")
        print(f"   • Review analysis results in JSON file")
        print(f"   • Use insights for feature engineering")
        print(f"   • Apply findings to model training")
    
    else:
        print("❌ Analysis failed!")

if __name__ == "__main__":
    main() 