#!/usr/bin/env python3
"""
Benchmark Results Analysis and Comparison
Generate comprehensive comparison tables and visualizations for all trained models
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import argparse


def load_all_results(output_dir="outputs"):
    """Load all training results from output directories"""
    results = {}
    
    # Search for results.json files in subdirectories
    for subdir in Path(output_dir).iterdir():
        if subdir.is_dir():
            results_file = subdir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                        model_key = data.get('model_key', subdir.name)
                        results[model_key] = data
                        print(f"✅ Loaded results for {model_key}")
                except Exception as e:
                    print(f"❌ Error loading {results_file}: {e}")
    
    return results


def create_comparison_table(results):
    """Create a comprehensive comparison table"""
    comparison_data = []
    
    for model_key, data in results.items():
        config = data.get('training_config', {})
        metrics = data.get('test_metrics', {})
        
        row = {
            'Model': model_key,
            'Model Name': data.get('model_name', 'Unknown'),
            'Parameters': get_parameter_count(model_key),
            'Training Method': get_training_method(model_key, config),
            'Batch Size': config.get('batch_size', 'N/A'),
            'Learning Rate': f"{config.get('learning_rate', 0):.1e}",
            'Epochs': config.get('epochs', 'N/A'),
            'Max Length': config.get('max_length', 'N/A'),
            'F1 Score': f"{metrics.get('f1_weighted', 0):.4f}",
            'F1 Binary': f"{metrics.get('f1_binary', 0):.4f}",
            'F1 Macro': f"{metrics.get('f1_macro', 0):.4f}",
            'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
            'Training Time': data.get('training_time_formatted', 'N/A'),
            'Training Loss': f"{data.get('train_loss', 0):.4f}",
            'Timestamp': data.get('timestamp', 'N/A')
        }
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by F1 Score (descending)
    df = df.sort_values('F1 Score', ascending=False)
    
    return df


def get_parameter_count(model_key):
    """Get approximate parameter count for models"""
    param_counts = {
        'bert-base-uncased': '110M',
        'deberta-v3-base': '184M', 
        'bert-large-uncased': '340M',
        'mistral-7b-v0.2': '7B',
        'mistral-7b-instruct': '7B',
        'llama2-7b': '7B',
        'llama3-8b': '8B',
        'llama2-13b': '13B'
    }
    return param_counts.get(model_key, 'Unknown')


def get_training_method(model_key, config):
    """Determine training method"""
    if 'lora_r' in config:
        quant = config.get('quantization', '4bit')
        return f"LoRA + {quant}"
    elif config.get('fp16', False):
        return "Full Fine-tune (FP16)"
    else:
        return "Full Fine-tune"


def create_visualizations(df, output_dir):
    """Create various visualization charts"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. F1 Score Comparison
    plt.figure(figsize=(12, 8))
    df_sorted = df.sort_values('F1 Score')
    
    colors = ['#ff7f0e' if 'lora' in model.lower() else '#1f77b4' 
              for model in df_sorted['Model']]
    
    bars = plt.barh(df_sorted['Model'], df_sorted['F1 Score'].astype(float), color=colors)
    plt.xlabel('F1 Score (Weighted)', fontsize=12)
    plt.title('Model Performance Comparison - F1 Score', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Full Fine-tune'),
        Patch(facecolor='#ff7f0e', label='LoRA')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Training Time vs Performance
    plt.figure(figsize=(10, 8))
    
    # Convert training time to minutes
    times = []
    for time_str in df['Training Time']:
        if 'h' in time_str and 'm' in time_str:
            parts = time_str.replace('h', '').replace('m', '').replace('s', '').split()
            hours = float(parts[0]) if len(parts) >= 1 else 0
            minutes = float(parts[1]) if len(parts) >= 2 else 0
            total_minutes = hours * 60 + minutes
        else:
            total_minutes = 0
        times.append(total_minutes)
    
    df['Training Time (Minutes)'] = times
    
    # Create scatter plot
    colors = ['orange' if 'lora' in model.lower() else 'blue' 
              for model in df['Model']]
    
    scatter = plt.scatter(df['Training Time (Minutes)'], df['F1 Score'].astype(float), 
                         c=colors, s=100, alpha=0.7)
    
    # Add model labels
    for i, model in enumerate(df['Model']):
        plt.annotate(model, (df['Training Time (Minutes)'].iloc[i], 
                           df['F1 Score'].astype(float).iloc[i]),
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=9, alpha=0.8)
    
    plt.xlabel('Training Time (Minutes)', fontsize=12)
    plt.ylabel('F1 Score (Weighted)', fontsize=12)
    plt.title('Training Time vs Performance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/time_vs_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Metrics Comparison Radar Chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Select top 5 models by F1 score
    top_models = df.head(5)
    
    metrics = ['F1 Score', 'Accuracy', 'F1 Binary', 'F1 Macro']
    angles = [n / float(len(metrics)) * 2 * 3.14159 for n in range(len(metrics))]
    angles += angles[:1]  # Complete the circle
    
    colors = plt.cm.Set3(range(len(top_models)))
    
    for i, (_, row) in enumerate(top_models.iterrows()):
        values = [float(row[metric]) for metric in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('Top 5 Models - Multi-Metric Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualizations saved to {output_dir}/")


def generate_summary_report(df, results, output_dir):
    """Generate a comprehensive summary report"""
    
    report = []
    report.append("# 🎯 Clickbait Classification - RTX A5000 Benchmark Results\n")
    report.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Total Models Trained:** {len(df)}\n\n")
    
    # Best performing models
    report.append("## 🏆 Top Performing Models\n")
    top_3 = df.head(3)
    for i, (_, row) in enumerate(top_3.iterrows(), 1):
        report.append(f"**{i}. {row['Model']}**\n")
        report.append(f"   - F1 Score: {row['F1 Score']}\n")
        report.append(f"   - Accuracy: {row['Accuracy']}\n")
        report.append(f"   - Training Time: {row['Training Time']}\n")
        report.append(f"   - Parameters: {row['Parameters']}\n\n")
    
    # Training method comparison
    report.append("## 📊 Training Method Analysis\n")
    
    lora_models = df[df['Training Method'].str.contains('LoRA')]
    full_ft_models = df[df['Training Method'].str.contains('Full')]
    
    if len(lora_models) > 0:
        report.append(f"**LoRA Models ({len(lora_models)}):**\n")
        report.append(f"- Average F1 Score: {lora_models['F1 Score'].astype(float).mean():.4f}\n")
        report.append(f"- Best F1 Score: {lora_models['F1 Score'].astype(float).max():.4f}\n\n")
    
    if len(full_ft_models) > 0:
        report.append(f"**Full Fine-tune Models ({len(full_ft_models)}):**\n")
        report.append(f"- Average F1 Score: {full_ft_models['F1 Score'].astype(float).mean():.4f}\n")
        report.append(f"- Best F1 Score: {full_ft_models['F1 Score'].astype(float).max():.4f}\n\n")
    
    # Parameter efficiency
    report.append("## ⚡ Efficiency Analysis\n")
    
    # Best performance per parameter size
    param_groups = {
        'Small (≤200M)': df[df['Parameters'].isin(['110M', '184M'])],
        'Medium (300M-1B)': df[df['Parameters'].isin(['340M'])],
        'Large (7B-8B)': df[df['Parameters'].isin(['7B', '8B'])],
        'Extra Large (≥13B)': df[df['Parameters'].isin(['13B'])]
    }
    
    for group_name, group_df in param_groups.items():
        if len(group_df) > 0:
            best_model = group_df.loc[group_df['F1 Score'].astype(float).idxmax()]
            report.append(f"**{group_name}:** {best_model['Model']} (F1: {best_model['F1 Score']})\n")
    
    report.append("\n")
    
    # Detailed results table
    report.append("## 📋 Detailed Results Table\n")
    report.append("| Model | Parameters | Method | F1 Score | Accuracy | Training Time |\n")
    report.append("|-------|------------|--------|----------|----------|---------------|\n")
    
    for _, row in df.iterrows():
        report.append(f"| {row['Model']} | {row['Parameters']} | {row['Training Method']} | "
                     f"{row['F1 Score']} | {row['Accuracy']} | {row['Training Time']} |\n")
    
    # Save report
    report_path = f"{output_dir}/benchmark_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"📄 Summary report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark results and comparisons")
    parser.add_argument("--output_dir", default="outputs", help="Output directory with results")
    parser.add_argument("--save_csv", action="store_true", help="Save results as CSV")
    
    args = parser.parse_args()
    
    print("📊 BENCHMARK RESULTS ANALYSIS")
    print("="*50)
    
    # Load all results
    results = load_all_results(args.output_dir)
    
    if not results:
        print("❌ No training results found!")
        print(f"Make sure there are trained models in {args.output_dir}/")
        return
    
    print(f"📈 Found results for {len(results)} models")
    
    # Create comparison table
    df = create_comparison_table(results)
    
    # Display results
    print("\n🏆 MODEL PERFORMANCE RANKING:")
    print("="*80)
    
    display_columns = ['Model', 'Parameters', 'Training Method', 'F1 Score', 'Accuracy', 'Training Time']
    print(df[display_columns].to_string(index=False))
    
    # Save CSV if requested
    if args.save_csv:
        csv_path = f"{args.output_dir}/benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Results saved to {csv_path}")
    
    # Create visualizations
    print(f"\n🎨 Generating visualizations...")
    create_visualizations(df, args.output_dir)
    
    # Generate summary report
    print(f"📝 Generating summary report...")
    generate_summary_report(df, results, args.output_dir)
    
    # Generate summary JSON
    summary = {
        'total_models': len(results),
        'best_model': {
            'name': df.iloc[0]['Model'],
            'f1_score': df.iloc[0]['F1 Score'],
            'accuracy': df.iloc[0]['Accuracy']
        },
        'models': df.to_dict('records'),
        'generated_at': datetime.now().isoformat()
    }
    
    summary_path = f"{args.output_dir}/benchmark_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print key insights
    print(f"\n💡 KEY INSIGHTS:")
    print("="*50)
    
    best_model = df.iloc[0]
    print(f"🥇 Best Overall: {best_model['Model']} (F1: {best_model['F1 Score']})")
    
    # Best in each category
    lora_models = df[df['Training Method'].str.contains('LoRA')]
    if len(lora_models) > 0:
        best_lora = lora_models.iloc[0]
        print(f"🥇 Best LoRA: {best_lora['Model']} (F1: {best_lora['F1 Score']})")
    
    full_ft = df[df['Training Method'].str.contains('Full')]
    if len(full_ft) > 0:
        best_full = full_ft.iloc[0]
        print(f"🥇 Best Full Fine-tune: {best_full['Model']} (F1: {best_full['F1 Score']})")
    
    print(f"\n✅ Analysis complete! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main() 