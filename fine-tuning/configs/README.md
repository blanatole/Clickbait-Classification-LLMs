# Fine-tuning Configuration for Dual RTX A6000 GPUs

## Overview

Cấu hình này được tối ưu hoá cho 2 GPU RTX A6000 (48GB VRAM mỗi GPU) để fine-tune các mô hình Mistral và Llama với LoRA/RS-LoRA.

## Cấu hình siêu tham số

### Mistral-7B-Instruct
- **LoRA Type**: Standard LoRA
- **Batch size**: 8 (per device)
- **Learning rate**: 2e-4
- **Gradient accumulation**: 4 steps
- **Weight decay**: 1e-3
- **Scheduler**: Cosine
- **Epochs**: 5
- **LoRA rank (r)**: 64 (~167M parameters, ~2.26%)

### Llama-3.1-8B-Instruct  
- **LoRA Type**: RS-LoRA
- **Batch size**: 16 (per device)
- **Learning rate**: 5e-5
- **Gradient accumulation**: 4 steps  
- **Weight decay**: 1e-2
- **Scheduler**: Cosine with 200-step warmup
- **Epochs**: 5
- **LoRA rank (r)**: 64 (~167M parameters, ~2.05%)

## Cách chạy

### 1. Single GPU (fallback)
```bash
python fine-tuning/scripts/train_llm_lora.py --model all
```

### 2. Distributed Training (Recommended cho 2 GPUs)
```bash
# Sử dụng script có sẵn
./fine-tuning/scripts/run_distributed_training.sh

# Hoặc chạy trực tiếp
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    fine-tuning/scripts/train_llm_lora.py \
    --model all
```

### 3. Chạy từng model riêng lẻ
```bash
# Chỉ Mistral
./fine-tuning/scripts/run_distributed_training.sh mistral-7b-instruct

# Chỉ Llama
./fine-tuning/scripts/run_distributed_training.sh llama3-8b
```

## Yêu cầu hệ thống

- **GPU**: 2x RTX A6000 (48GB VRAM mỗi GPU) 
- **RAM**: Tối thiểu 64GB
- **Storage**: 200GB+ free space cho model weights và checkpoints
- **CUDA**: 11.8+ hoặc 12.x
- **Python**: 3.9+

## Environment Setup

```bash
# Kích hoạt conda environment
conda activate clickbait

# Kiểm tra GPU
nvidia-smi

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Kết quả mong đợi

- **Training time**: ~6-8 giờ cho cả 2 models
- **Memory usage**: ~40-45GB per GPU
- **Model size**: ~167M trainable parameters cho mỗi model
- **Checkpoints**: Lưu trong `fine-tuning/outputs/`

## Monitoring

Kết quả training được lưu trong:
- `fine-tuning/outputs/{model_name}-lora-cuda/`
- `fine-tuning/outputs/llm_lora_summary.json` (tổng hợp)

Logs có thể theo dõi qua console output hoặc tensorboard logs. 