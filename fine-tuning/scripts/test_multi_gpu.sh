#!/bin/bash

echo "🧪 Testing Multi-GPU Setup for 2x RTX A6000"
echo "============================================"

# Activate conda environment
echo "📦 Activating clickbait environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate clickbait

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

echo ""
echo "🔧 Running basic GPU test..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo ""
echo "🌐 Testing distributed setup..."
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    fine-tuning/scripts/test_distributed_setup.py

echo ""
echo "✅ Multi-GPU test completed!"
echo ""
echo "💡 If all tests passed, you can run:"
echo "   ./fine-tuning/scripts/run_distributed_training.sh" 