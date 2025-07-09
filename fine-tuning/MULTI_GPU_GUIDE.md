# 🚀 Hướng dẫn cấu hình và chạy 2x GPU RTX A6000

## 📋 Tổng quan

Hướng dẫn này sẽ giúp bạn cấu hình và chạy distributed training trên 2 GPU RTX A6000 cho việc fine-tune các mô hình Mistral và Llama.

## 🔧 Bước 1: Kiểm tra và Setup Environment

### 1.1 Chạy script setup tự động
```bash
# Chạy script kiểm tra và setup
./fine-tuning/scripts/setup_multi_gpu.sh
```

### 1.2 Kiểm tra thủ công (nếu cần)

#### Kiểm tra GPU:
```bash
nvidia-smi
# Phải thấy 2 GPU RTX A6000, mỗi GPU có ~48GB VRAM
```

#### Kiểm tra CUDA:
```bash
nvcc --version
# Cần CUDA 11.8+ hoặc 12.x
```

#### Kiểm tra PyTorch với GPU:
```bash
conda activate clickbait
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
```

## 🎯 Bước 2: Các cách chạy Training

### 2.1 Cách 1: Sử dụng script có sẵn (Đơn giản nhất)

```bash
# Kích hoạt environment
conda activate clickbait

# Chạy cả 2 models
./fine-tuning/scripts/run_distributed_training.sh

# Hoặc chỉ chạy 1 model
./fine-tuning/scripts/run_distributed_training.sh mistral-7b-instruct
./fine-tuning/scripts/run_distributed_training.sh llama3-8b
```

### 2.2 Cách 2: Chạy thủ công với torch.distributed.launch

```bash
conda activate clickbait

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

# Chạy distributed training
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    fine-tuning/scripts/train_llm_lora.py \
    --model all \
    --output_dir fine-tuning/outputs
```

### 2.3 Cách 3: Sử dụng Accelerate (Alternative)

```bash
# Cài đặt accelerate nếu chưa có
pip install accelerate

# Config accelerate (chỉ cần làm 1 lần)
accelerate config
# Chọn:
# - Distributed training: Yes
# - Number of machines: 1
# - Number of processes: 2
# - GPU IDs: 0,1
# - Mixed precision: bf16

# Chạy training
accelerate launch fine-tuning/scripts/train_llm_lora.py --model all
```

## ⚙️ Bước 3: Giải thích các tham số quan trọng

### Environment Variables:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Tối ưu memory allocation
export CUDA_LAUNCH_BLOCKING=0                           # Tăng performance
export NCCL_P2P_DISABLE=0                              # Enable peer-to-peer GPU communication
export NCCL_IB_DISABLE=1                               # Disable InfiniBand (không cần cho local)
export NCCL_DEBUG=INFO                                 # Debug info cho NCCL
```

### Training Arguments quan trọng:
- `--nproc_per_node=2`: Số GPU sử dụng
- `--master_port=29500`: Port cho distributed training
- `ddp_backend="nccl"`: Backend tối ưu cho NVIDIA GPUs
- `bf16=True`: Mixed precision cho RTX A6000

## 📊 Bước 4: Monitor Training

### 4.1 Theo dõi real-time:
```bash
# Xem GPU usage
watch -n 1 nvidia-smi

# Xem logs
tail -f fine-tuning/outputs/*/logs/*.log
```

### 4.2 Kiểm tra progress:
Training sẽ tạo các files:
- `fine-tuning/outputs/mistral-7b-instruct-lora-cuda/`
- `fine-tuning/outputs/llama3-8b-lora-cuda/`
- `fine-tuning/outputs/llm_lora_summary.json`

## 🔍 Bước 5: Troubleshooting

### 5.1 Lỗi common và cách fix:

#### "CUDA out of memory":
```bash
# Giảm batch size trong config
# Mistral: batch_size từ 8 → 4
# Llama: batch_size từ 16 → 8
```

#### "NCCL initialization failed":
```bash
# Kiểm tra GPUs có được nhận diện không
nvidia-smi

# Restart và thử lại
sudo nvidia-smi -r
```

#### "Port already in use":
```bash
# Đổi port khác
--master_port=29501
```

### 5.2 Debug commands:
```bash
# Test distributed setup
python -c "
import torch
import torch.distributed as dist
print('Testing distributed setup...')
"

# Check NCCL
python -c "
import torch
print('NCCL available:', torch.distributed.is_nccl_available())
"
```

## 📈 Bước 6: Expected Performance

### Memory Usage (per GPU):
- **Mistral-7B**: ~40GB VRAM
- **Llama-8B**: ~45GB VRAM

### Training Time:
- **Mistral**: ~3-4 giờ (5 epochs)
- **Llama**: ~4-5 giờ (5 epochs)
- **Total**: ~6-8 giờ cho cả 2 models

### Effective Batch Sizes:
- **Mistral**: 8 (per GPU) × 4 (accum) × 2 (GPUs) = 64
- **Llama**: 16 (per GPU) × 4 (accum) × 2 (GPUs) = 128

## 🎯 Quick Start Commands

```bash
# All-in-one setup and training
cd /path/to/your/project
conda activate clickbait
./fine-tuning/scripts/setup_multi_gpu.sh
./fine-tuning/scripts/run_distributed_training.sh

# Monitor
watch -n 1 nvidia-smi
```

## 📞 Support

Nếu gặp vấn đề:
1. Check `fine-tuning/outputs/*/logs/` để xem error logs
2. Đảm bảo CUDA drivers updated
3. Kiểm tra VRAM availability với `nvidia-smi`
4. Thử chạy single GPU trước để test setup 