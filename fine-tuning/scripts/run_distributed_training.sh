#!/bin/bash

echo "🚀 Starting distributed training on dual RTX A6000 GPUs"
echo "Configuration: Mistral + Llama fine-tuning with LoRA/RS-LoRA"

# Activate conda environment
echo "📦 Activating clickbait environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate clickbait

# Check if GPUs are available
echo "🔧 Checking GPU availability..."
nvidia-smi

# Set environment variables for optimal performance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

# Parse arguments
MODEL=${1:-"all"}  # Default to all models
OUTPUT_DIR=${2:-"fine-tuning/outputs"}
RUN_TEST=${3:-"false"}  # Optional test flag

echo "🎯 Training model(s): $MODEL"
echo "📁 Output directory: $OUTPUT_DIR"

# Optional: Run test first if requested
if [ "$RUN_TEST" = "test" ] || [ "$3" = "test" ]; then
    echo ""
    echo "🧪 Running distributed setup test first..."
    ./fine-tuning/scripts/test_multi_gpu.sh
    
    echo ""
    read -p "❓ Test completed. Continue with training? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Training cancelled by user"
        exit 0
    fi
fi

# Run distributed training
echo "🏋️ Starting training..."
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    fine-tuning/scripts/train_llm_lora.py \
    --model $MODEL \
    --output_dir $OUTPUT_DIR

echo "✅ Training completed!"
echo "📊 Check results in: $OUTPUT_DIR"
echo ""
echo "📈 Training summary will be in: $OUTPUT_DIR/llm_lora_summary.json"

# Show usage if no arguments provided
if [ $# -eq 0 ]; then
    echo ""
    echo "📖 Usage examples:"
    echo "   ./run_distributed_training.sh                     # Train all models"
    echo "   ./run_distributed_training.sh mistral-7b-instruct # Train only Mistral"
    echo "   ./run_distributed_training.sh llama3-8b          # Train only Llama"
    echo "   ./run_distributed_training.sh all test           # Run test first"
fi 