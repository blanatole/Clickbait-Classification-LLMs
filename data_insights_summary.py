#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Insights Summary - Tóm tắt phân tích dữ liệu clickbait
"""

import json
import os

def load_analysis_results(file_path="data_analysis_results.json"):
    """Load analysis results from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading analysis results: {e}")
        return None

def print_key_insights(analysis):
    """Print key insights from the analysis"""
    
    print("🔍 PHÂN TÍCH DỮ LIỆU CLICKBAIT - KẾT QUẢ CHI TIẾT")
    print("=" * 60)
    
    # Overall statistics
    if 'overall_statistics' in analysis:
        overall = analysis['overall_statistics']['basic_statistics']
        print(f"\n📊 THỐNG KÊ TỔNG QUAN:")
        print(f"   • Tổng số mẫu: {overall['total_samples']:,}")
        print(f"   • Clickbait: {overall['class_distribution']['clickbait']['count']:,} ({overall['class_distribution']['clickbait']['percentage']}%)")
        print(f"   • No-clickbait: {overall['class_distribution']['no_clickbait']['count']:,} ({overall['class_distribution']['no_clickbait']['percentage']}%)")
        print(f"   • Độ dài text trung bình: {overall['text_length_statistics']['mean']} ký tự")
        print(f"   • Độ dài text ngắn nhất: {overall['text_length_statistics']['min']} ký tự")
        print(f"   • Độ dài text dài nhất: {overall['text_length_statistics']['max']} ký tự")
    
    # Text patterns comparison
    if 'overall_statistics' in analysis:
        text_patterns = analysis['overall_statistics']['text_patterns']
        
        print(f"\n📝 SO SÁNH PATTERNS TEXT:")
        
        clickbait = text_patterns['clickbait_patterns']
        no_clickbait = text_patterns['no_clickbait_patterns']
        
        print(f"   CLICKBAIT:")
        print(f"   • Độ dài trung bình: {clickbait['average_length']:.1f} ký tự")
        print(f"   • Số từ trung bình: {clickbait['average_word_count']:.1f} từ")
        print(f"   • Có dấu hỏi: {clickbait['punctuation_stats']['question_marks']['percentage']:.1f}%")
        print(f"   • Có dấu cảm: {clickbait['punctuation_stats']['exclamation_marks']['percentage']:.1f}%")
        
        print(f"   NO-CLICKBAIT:")
        print(f"   • Độ dài trung bình: {no_clickbait['average_length']:.1f} ký tự")
        print(f"   • Số từ trung bình: {no_clickbait['average_word_count']:.1f} từ")
        print(f"   • Có dấu hỏi: {no_clickbait['punctuation_stats']['question_marks']['percentage']:.1f}%")
        print(f"   • Có dấu cảm: {no_clickbait['punctuation_stats']['exclamation_marks']['percentage']:.1f}%")
    
    # Top discriminative keywords
    if 'overall_statistics' in analysis:
        keywords = analysis['overall_statistics']['keyword_analysis']['top_discriminative_keywords']
        
        print(f"\n🔑 TỪ KHÓA PHÂN BIỆT CLICKBAIT:")
        for i, (keyword, ratio) in enumerate(list(keywords.items())[:10], 1):
            if ratio != float('inf'):
                print(f"   {i:2d}. '{keyword}': tỷ lệ {ratio:.2f}x (xuất hiện nhiều hơn trong clickbait)")
    
    # Advanced patterns
    if 'overall_statistics' in analysis:
        advanced = analysis['overall_statistics']['advanced_patterns']
        
        print(f"\n🎯 PATTERNS NÂNG CAO:")
        
        cb_patterns = advanced['clickbait_advanced_patterns']
        ncb_patterns = advanced['no_clickbait_advanced_patterns']
        
        print(f"   CLICKBAIT:")
        print(f"   • Có số: {cb_patterns['formatting_patterns']['has_numbers']['percentage']:.1f}%")
        print(f"   • Có từ so sánh: {cb_patterns['linguistic_patterns']['superlatives']['percentage']:.1f}%")
        print(f"   • Có từ thời gian: {cb_patterns['linguistic_patterns']['time_words']['percentage']:.1f}%")
        print(f"   • Có đại từ nhân xưng: {cb_patterns['linguistic_patterns']['personal_pronouns']['percentage']:.1f}%")
        
        print(f"   NO-CLICKBAIT:")
        print(f"   • Có số: {ncb_patterns['formatting_patterns']['has_numbers']['percentage']:.1f}%")
        print(f"   • Có từ so sánh: {ncb_patterns['linguistic_patterns']['superlatives']['percentage']:.1f}%")
        print(f"   • Có từ thời gian: {ncb_patterns['linguistic_patterns']['time_words']['percentage']:.1f}%")
        print(f"   • Có đại từ nhân xưng: {ncb_patterns['linguistic_patterns']['personal_pronouns']['percentage']:.1f}%")
    
    # Truth score analysis
    if 'overall_statistics' in analysis:
        truth_analysis = analysis['overall_statistics']['truth_score_analysis']['truth_score_distribution']
        
        print(f"\n📈 PHÂN TÍCH TRUTH SCORES:")
        for score_range, data in truth_analysis.items():
            print(f"   {score_range}: {data['total_samples']} mẫu, {data['clickbait_percentage']:.1f}% clickbait")
    
    # Comparative insights
    if 'comparative_analysis' in analysis:
        comp = analysis['comparative_analysis']
        
        print(f"\n🔄 INSIGHTS SO SÁNH:")
        print(f"   • Clickbait ngắn hơn {abs(comp['key_differences']['average_length_difference']):.1f} ký tự")
        print(f"   • Clickbait: {comp['key_differences']['clickbait_avg_length']:.1f} ký tự")
        print(f"   • No-clickbait: {comp['key_differences']['no_clickbait_avg_length']:.1f} ký tự")
    
    # Top words in each class
    if 'overall_statistics' in analysis:
        text_patterns = analysis['overall_statistics']['text_patterns']
        
        print(f"\n📊 TOP TỪ XUẤT HIỆN NHIỀU NHẤT:")
        
        print(f"   CLICKBAIT:")
        cb_words = text_patterns['clickbait_patterns']['top_words']
        for i, (word, count) in enumerate(list(cb_words.items())[:10], 1):
            if word != 'sep':  # Skip separator
                print(f"   {i:2d}. '{word}': {count:,} lần")
        
        print(f"   NO-CLICKBAIT:")
        ncb_words = text_patterns['no_clickbait_patterns']['top_words']
        for i, (word, count) in enumerate(list(ncb_words.items())[:10], 1):
            if word != 'sep':  # Skip separator
                print(f"   {i:2d}. '{word}': {count:,} lần")

def print_recommendations(analysis):
    """Print actionable recommendations"""
    
    print(f"\n💡 KHUYẾN NGHỊ CHO TRAINING MODEL:")
    print("=" * 40)
    
    if 'comparative_analysis' in analysis:
        recommendations = analysis['comparative_analysis']['recommendations']
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    print(f"\n🎯 FEATURE ENGINEERING SUGGESTIONS:")
    print(f"   • Text length (clickbait thường ngắn hơn)")
    print(f"   • Punctuation ratio (dấu hỏi, dấu cảm)")
    print(f"   • Personal pronouns count ('you', 'your')")
    print(f"   • Question words frequency ('how', 'what', 'why')")
    print(f"   • Numbers presence")
    print(f"   • Superlative words")
    print(f"   • Time urgency words")
    print(f"   • TF-IDF scores cho top discriminative words")
    
    print(f"\n🔧 MODEL TRAINING TIPS:")
    print(f"   • Sử dụng attention mechanism cho personal pronouns")
    print(f"   • Weight higher cho question marks và exclamation marks")
    print(f"   • Feature engineering cho text length ratio")
    print(f"   • Ensemble method với text + metadata features")

def save_insights_summary(analysis, output_file="data_insights_summary.json"):
    """Save key insights to a separate JSON file"""
    
    insights_summary = {
        "timestamp": analysis.get('analysis_metadata', {}).get('timestamp', ''),
        "dataset_overview": {},
        "key_differences": {},
        "top_indicators": {},
        "recommendations": []
    }
    
    # Dataset overview
    if 'overall_statistics' in analysis:
        overall = analysis['overall_statistics']['basic_statistics']
        insights_summary["dataset_overview"] = {
            "total_samples": overall['total_samples'],
            "clickbait_percentage": overall['class_distribution']['clickbait']['percentage'],
            "average_text_length": overall['text_length_statistics']['mean'],
            "text_length_range": {
                "min": overall['text_length_statistics']['min'],
                "max": overall['text_length_statistics']['max']
            }
        }
    
    # Key differences
    if 'overall_statistics' in analysis:
        text_patterns = analysis['overall_statistics']['text_patterns']
        cb_patterns = text_patterns['clickbait_patterns']
        ncb_patterns = text_patterns['no_clickbait_patterns']
        
        insights_summary["key_differences"] = {
            "length_difference": cb_patterns['average_length'] - ncb_patterns['average_length'],
            "question_marks_difference": cb_patterns['punctuation_stats']['question_marks']['percentage'] - 
                                       ncb_patterns['punctuation_stats']['question_marks']['percentage'],
            "exclamation_marks_difference": cb_patterns['punctuation_stats']['exclamation_marks']['percentage'] - 
                                          ncb_patterns['punctuation_stats']['exclamation_marks']['percentage']
        }
    
    # Top indicators
    if 'overall_statistics' in analysis:
        keywords = analysis['overall_statistics']['keyword_analysis']['top_discriminative_keywords']
        insights_summary["top_indicators"] = dict(list(keywords.items())[:10])
    
    # Recommendations
    if 'comparative_analysis' in analysis:
        insights_summary["recommendations"] = analysis['comparative_analysis']['recommendations']
    
    # Save to file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(insights_summary, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Insights summary saved to: {output_file}")
        return True
    except Exception as e:
        print(f"❌ Error saving insights summary: {e}")
        return False

def main():
    """Main function"""
    
    # Load analysis results
    analysis = load_analysis_results()
    
    if analysis is None:
        print("❌ Could not load analysis results!")
        return
    
    # Print insights
    print_key_insights(analysis)
    print_recommendations(analysis)
    
    # Save summary
    save_insights_summary(analysis)
    
    print(f"\n✅ HOÀN THÀNH PHÂN TÍCH DỮ LIỆU!")
    print(f"📁 Files created:")
    print(f"   • data_analysis_results.json (full analysis)")
    print(f"   • data_insights_summary.json (key insights)")

if __name__ == "__main__":
    main() 