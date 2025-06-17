#!/usr/bin/env python3
"""
Run all clickbait classification experiments on RTX A5000
Master script to train all models and generate comparison results
"""

import subprocess
import time
import json
import os
from datetime import datetime
from pathlib import Path


def run_command(command, description):
    """Run a command and track execution time"""
    print(f"\n{'='*80}")
    print(f"🚀 {description}")
    print(f"Command: {command}")
    print(f"{'='*80}")
    
    start_time = time.time()
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        execution_time = time.time() - start_time
        
        print(f"✅ {description} completed in {execution_time//60:.0f}m {execution_time%60:.0f}s")
        return True, execution_time, result.stdout
        
    except subprocess.CalledProcessError as e:
        execution_time = time.time() - start_time
        print(f"❌ {description} failed after {execution_time//60:.0f}m {execution_time%60:.0f}s")
        print(f"Error: {e.stderr}")
        return False, execution_time, e.stderr


def check_gpu_memory():
    """Check available GPU memory"""
    try:
        result = subprocess.run(
            "nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits",
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                total, used, free = map(int, line.split(', '))
                print(f"GPU {i}: {total}MB total, {used}MB used, {free}MB free")
                
                if free < 8000:  # Less than 8GB free
                    print(f"⚠️  Warning: GPU {i} has less than 8GB free memory")
                    return False
        return True
    except Exception as e:
        print(f"Could not check GPU memory: {e}")
        return True


def main():
    experiment_start = time.time()
    
    print("🎯 CLICKBAIT CLASSIFICATION - RTX A5000 FULL EXPERIMENT SUITE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check GPU memory
    print("🔧 Checking GPU status...")
    if not check_gpu_memory():
        print("⚠️  Warning: Low GPU memory detected. Consider closing other processes.")
    
    # Create output directory
    output_dir = "outputs"
    Path(output_dir).mkdir(exist_ok=True)
    
    # Experiment tracking
    experiment_log = {
        "start_time": datetime.now().isoformat(),
        "experiments": {},
        "summary": {}
    }
    
    # Define experiments
    experiments = [
        {
            "name": "BERT Family Models",
            "command": "python scripts/train_bert_family.py --model all",
            "description": "Train BERT-base, DeBERTa-v3-base, BERT-large",
            "expected_time_minutes": 180,  # ~3 hours
            "models": ["bert-base-uncased", "deberta-v3-base", "bert-large-uncased"]
        },
        {
            "name": "Mistral Models",
            "command": "python scripts/train_llm_lora.py --model mistral-7b-v0.2",
            "description": "Train Mistral-7B-v0.2 with QLoRA",
            "expected_time_minutes": 120,  # ~2 hours
            "models": ["mistral-7b-v0.2"]
        },
        {
            "name": "Mistral Instruct",
            "command": "python scripts/train_llm_lora.py --model mistral-7b-instruct",
            "description": "Train Mistral-7B-Instruct with QLoRA",
            "expected_time_minutes": 90,   # ~1.5 hours (fewer epochs)
            "models": ["mistral-7b-instruct"]
        },
        {
            "name": "Llama-2-7B",
            "command": "python scripts/train_llm_lora.py --model llama2-7b",
            "description": "Train Llama-2-7B with QLoRA",
            "expected_time_minutes": 120,  # ~2 hours
            "models": ["llama2-7b"]
        },
        {
            "name": "Llama-3-8B",
            "command": "python scripts/train_llm_lora.py --model llama3-8b",
            "description": "Train Llama-3-8B with QLoRA",
            "expected_time_minutes": 150,  # ~2.5 hours
            "models": ["llama3-8b"]  
        },
        {
            "name": "Llama-2-13B",
            "command": "python scripts/train_llm_lora.py --model llama2-13b",
            "description": "Train Llama-2-13B with QLoRA (8-bit)",
            "expected_time_minutes": 180,  # ~3 hours
            "models": ["llama2-13b"]
        }
    ]
    
    # Run experiments
    successful_experiments = []
    failed_experiments = []
    
    for i, experiment in enumerate(experiments, 1):
        print(f"\n🎬 EXPERIMENT {i}/{len(experiments)}: {experiment['name']}")
        print(f"Expected duration: ~{experiment['expected_time_minutes']} minutes")
        print(f"Models: {', '.join(experiment['models'])}")
        
        # Check if user wants to continue
        response = input(f"\nProceed with {experiment['name']}? (y/n/skip): ").lower()
        
        if response == 'n':
            print("❌ Experiment suite cancelled by user")
            break
        elif response == 'skip':
            print(f"⏭️  Skipping {experiment['name']}")
            continue
        
        # Run experiment
        success, duration, output = run_command(
            experiment["command"], 
            experiment["description"]
        )
        
        # Log results
        experiment_log["experiments"][experiment["name"]] = {
            "command": experiment["command"],
            "success": success,
            "duration_seconds": duration,
            "duration_formatted": f"{duration//3600:.0f}h {(duration%3600)//60:.0f}m {duration%60:.0f}s",
            "models": experiment["models"],
            "timestamp": datetime.now().isoformat()
        }
        
        if success:
            successful_experiments.append(experiment["name"])
            print(f"✅ {experiment['name']} completed successfully")
        else:
            failed_experiments.append(experiment["name"])
            print(f"❌ {experiment['name']} failed")
        
        # Save progress
        with open(f"{output_dir}/experiment_log.json", "w") as f:
            json.dump(experiment_log, f, indent=2)
        
        # Clean GPU memory between experiments
        print("🧹 Cleaning GPU memory...")
        try:
            subprocess.run("python -c 'import torch; torch.cuda.empty_cache()'", 
                         shell=True, check=False)
        except:
            pass
        
        time.sleep(10)  # Brief pause between experiments
    
    # Generate final summary
    total_time = time.time() - experiment_start
    experiment_log["end_time"] = datetime.now().isoformat()
    experiment_log["total_duration_seconds"] = total_time
    experiment_log["total_duration_formatted"] = f"{total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s"
    experiment_log["summary"] = {
        "total_experiments": len(experiments),
        "successful_experiments": len(successful_experiments),
        "failed_experiments": len(failed_experiments),
        "success_rate": len(successful_experiments) / len(experiments) * 100,
        "successful_list": successful_experiments,
        "failed_list": failed_experiments
    }
    
    # Save final log
    with open(f"{output_dir}/experiment_log.json", "w") as f:
        json.dump(experiment_log, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("🏁 EXPERIMENT SUITE COMPLETED")
    print(f"{'='*80}")
    print(f"Total time: {experiment_log['total_duration_formatted']}")
    print(f"Successful: {len(successful_experiments)}/{len(experiments)} experiments")
    print(f"Success rate: {experiment_log['summary']['success_rate']:.1f}%")
    
    if successful_experiments:
        print(f"\n✅ Successful experiments:")
        for exp in successful_experiments:
            print(f"   - {exp}")
    
    if failed_experiments:
        print(f"\n❌ Failed experiments:")
        for exp in failed_experiments:
            print(f"   - {exp}")
    
    # Generate comparison report
    if successful_experiments:
        print(f"\n📊 Generating comparison report...")
        try:
            subprocess.run("python scripts/benchmark_results.py", shell=True, check=True)
            print("✅ Comparison report generated successfully")
        except:
            print("⚠️  Could not generate comparison report")
    
    print(f"\n📁 All results saved to: {output_dir}/")
    print(f"📋 Experiment log: {output_dir}/experiment_log.json")


if __name__ == "__main__":
    main() 