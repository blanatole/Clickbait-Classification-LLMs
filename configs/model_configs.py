#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Configurations cho Clickbait Classification
Cấu hình cho các GPU khác nhau từ RTX 3050 đến A5000/A6000
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ModelConfig:
    """Base configuration class"""
    model_name: str
    model_path: str
    max_length: int
    batch_size_train: int
    batch_size_eval: int
    learning_rate: float
    num_epochs: int
    warmup_steps: int
    weight_decay: float
    gradient_accumulation_steps: int
    fp16: bool
    dataloader_num_workers: int
    save_steps: int
    eval_steps: int
    logging_steps: int
    early_stopping_patience: int
    gpu_memory_required: str  # Ước tính VRAM cần thiết
    description: str

# =============================================================================
# RTX 3050 CONFIGS (8GB VRAM) - Demo và Test nhanh
# =============================================================================

RTX_3050_DISTILBERT = ModelConfig(
    model_name="distilbert_rtx3050",
    model_path="distilbert-base-uncased",
    max_length=128,
    batch_size_train=16,
    batch_size_eval=32,
    learning_rate=5e-5,
    num_epochs=3,
    warmup_steps=100,
    weight_decay=0.01,
    gradient_accumulation_steps=2,
    fp16=True,
    dataloader_num_workers=2,
    save_steps=100,
    eval_steps=50,
    logging_steps=25,
    early_stopping_patience=3,
    gpu_memory_required="4-6GB",
    description="DistilBERT nhẹ cho RTX 3050, demo nhanh"
)

RTX_3050_TINYBERT = ModelConfig(
    model_name="tinybert_rtx3050", 
    model_path="huawei-noah/TinyBERT_General_4L_312D",
    max_length=128,
    batch_size_train=24,
    batch_size_eval=48,
    learning_rate=8e-5,
    num_epochs=5,
    warmup_steps=100,
    weight_decay=0.01,
    gradient_accumulation_steps=1,
    fp16=True,
    dataloader_num_workers=2,
    save_steps=100,
    eval_steps=50,
    logging_steps=25,
    early_stopping_patience=5,
    gpu_memory_required="2-4GB",
    description="TinyBERT rất nhẹ, phù hợp demo quick test"
)

RTX_3050_ELECTRA_SMALL = ModelConfig(
    model_name="electra_small_rtx3050",
    model_path="google/electra-small-discriminator",
    max_length=128,
    batch_size_train=20,
    batch_size_eval=40,
    learning_rate=3e-4,
    num_epochs=4,
    warmup_steps=150,
    weight_decay=0.01,
    gradient_accumulation_steps=1,
    fp16=True,
    dataloader_num_workers=2,
    save_steps=100,
    eval_steps=50,
    logging_steps=25,
    early_stopping_patience=4,
    gpu_memory_required="3-5GB",
    description="ELECTRA-small nhanh và hiệu quả cho RTX 3050"
)

# =============================================================================
# A5000/A6000 CONFIGS (24GB+ VRAM) - Full Performance
# =============================================================================

A5000_DEBERTA_V3 = ModelConfig(
    model_name="deberta_v3_a5000",
    model_path="microsoft/deberta-v3-base",
    max_length=256,
    batch_size_train=32,
    batch_size_eval=64,
    learning_rate=2e-5,
    num_epochs=5,
    warmup_steps=500,
    weight_decay=0.01,
    gradient_accumulation_steps=1,
    fp16=True,
    dataloader_num_workers=4,
    save_steps=500,
    eval_steps=250,
    logging_steps=100,
    early_stopping_patience=3,
    gpu_memory_required="12-16GB",
    description="DeBERTa-v3 performance cao cho A5000"
)

A6000_DEBERTA_V3_LARGE = ModelConfig(
    model_name="deberta_v3_large_a6000",
    model_path="microsoft/deberta-v3-large",
    max_length=512,
    batch_size_train=16,
    batch_size_eval=32,
    learning_rate=1e-5,
    num_epochs=4,
    warmup_steps=800,
    weight_decay=0.01,
    gradient_accumulation_steps=2,
    fp16=True,
    dataloader_num_workers=6,
    save_steps=1000,
    eval_steps=500,
    logging_steps=200,
    early_stopping_patience=2,
    gpu_memory_required="20-24GB",
    description="DeBERTa-v3-Large cho A6000, best performance"
)

A5000_ROBERTA_LARGE = ModelConfig(
    model_name="roberta_large_a5000",
    model_path="roberta-large",
    max_length=256,
    batch_size_train=24,
    batch_size_eval=48,
    learning_rate=1e-5,
    num_epochs=4,
    warmup_steps=600,
    weight_decay=0.01,
    gradient_accumulation_steps=1,
    fp16=True,
    dataloader_num_workers=4,
    save_steps=500,
    eval_steps=250,
    logging_steps=100,
    early_stopping_patience=3,
    gpu_memory_required="14-18GB",
    description="RoBERTa-Large hiệu quả cho A5000"
)

# =============================================================================
# LLaMA/Mistral LoRA CONFIGS (GPU mạnh)
# =============================================================================

A6000_LLAMA2_7B_LORA = ModelConfig(
    model_name="llama2_7b_lora_a6000",
    model_path="meta-llama/Llama-2-7b-chat-hf",
    max_length=512,
    batch_size_train=4,
    batch_size_eval=8,
    learning_rate=2e-4,
    num_epochs=3,
    warmup_steps=100,
    weight_decay=0.01,
    gradient_accumulation_steps=8,
    fp16=True,
    dataloader_num_workers=2,
    save_steps=500,
    eval_steps=250,
    logging_steps=50,
    early_stopping_patience=2,
    gpu_memory_required="20-24GB",
    description="LLaMA-2 7B với LoRA, performance cao nhưng cần GPU mạnh"
)

A5000_MISTRAL_7B_LORA = ModelConfig(
    model_name="mistral_7b_lora_a5000",
    model_path="mistralai/Mistral-7B-Instruct-v0.1",
    max_length=512,
    batch_size_train=6,
    batch_size_eval=12,
    learning_rate=1e-4,
    num_epochs=3,
    warmup_steps=150,
    weight_decay=0.01,
    gradient_accumulation_steps=4,
    fp16=True,
    dataloader_num_workers=2,
    save_steps=500,
    eval_steps=250,
    logging_steps=50,
    early_stopping_patience=2,
    gpu_memory_required="16-20GB",
    description="Mistral 7B với LoRA, balance giữa performance và resource"
)

# =============================================================================
# CONFIG REGISTRY
# =============================================================================

# Demo configs cho RTX 3050
DEMO_CONFIGS = {
    "tinybert": RTX_3050_TINYBERT,
    "distilbert": RTX_3050_DISTILBERT,
    "electra_small": RTX_3050_ELECTRA_SMALL,
}

# Full configs cho GPU mạnh
FULL_CONFIGS = {
    "deberta_v3": A5000_DEBERTA_V3,
    "deberta_v3_large": A6000_DEBERTA_V3_LARGE,
    "roberta_large": A5000_ROBERTA_LARGE,
    "llama2_lora": A6000_LLAMA2_7B_LORA,
    "mistral_lora": A5000_MISTRAL_7B_LORA,
}

ALL_CONFIGS = {**DEMO_CONFIGS, **FULL_CONFIGS}

def get_config(model_name: str) -> ModelConfig:
    """Get configuration by model name"""
    if model_name not in ALL_CONFIGS:
        available = ", ".join(ALL_CONFIGS.keys())
        raise ValueError(f"Model '{model_name}' not found. Available: {available}")
    return ALL_CONFIGS[model_name]

def get_demo_configs() -> Dict[str, ModelConfig]:
    """Get all demo configurations for RTX 3050"""
    return DEMO_CONFIGS

def get_full_configs() -> Dict[str, ModelConfig]:
    """Get all full configurations for powerful GPUs"""
    return FULL_CONFIGS

def list_configs_by_gpu(gpu_type: str = "rtx3050") -> Dict[str, ModelConfig]:
    """List configurations by GPU type"""
    if gpu_type.lower() in ["rtx3050", "rtx", "demo"]:
        return DEMO_CONFIGS
    elif gpu_type.lower() in ["a5000", "a6000", "full", "production"]:
        return FULL_CONFIGS
    else:
        return ALL_CONFIGS

def print_config_summary():
    """Print summary of all configurations"""
    print("🔧 MODEL CONFIGURATIONS SUMMARY")
    print("=" * 60)
    
    print("\n📱 DEMO CONFIGS (RTX 3050 - 8GB VRAM):")
    for name, config in DEMO_CONFIGS.items():
        print(f"  • {name:15} | {config.gpu_memory_required:8} | {config.description}")
    
    print("\n🚀 FULL CONFIGS (A5000/A6000 - 24GB+ VRAM):")
    for name, config in FULL_CONFIGS.items():
        print(f"  • {name:15} | {config.gpu_memory_required:8} | {config.description}")

if __name__ == "__main__":
    print_config_summary() 