#!/bin/bash

echo "🔧 Setup Multi-GPU Training Environment for 2x RTX A6000"
echo "=================================================="

# Function to check command existence
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 1. Check GPU availability
echo "1️⃣ Checking GPU configuration..."
if command_exists nvidia-smi; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "   Detected GPUs: $GPU_COUNT"
    
    if [ "$GPU_COUNT" -eq 2 ]; then
        echo "   ✅ Perfect! 2 GPUs detected for distributed training"
    elif [ "$GPU_COUNT" -eq 1 ]; then
        echo "   ⚠️  Only 1 GPU detected. Training will run on single GPU"
    else
        echo "   ℹ️  $GPU_COUNT GPUs detected. Will use all available"
    fi
else
    echo "   ❌ nvidia-smi not found. Please install CUDA drivers"
    exit 1
fi

# 2. Check CUDA version
echo ""
echo "2️⃣ Checking CUDA version..."
if command_exists nvcc; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "   CUDA Version: $CUDA_VERSION"
    
    # Extract major version
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d'.' -f1)
    if [ "$CUDA_MAJOR" -ge 11 ]; then
        echo "   ✅ CUDA version is compatible"
    else
        echo "   ⚠️  CUDA version might be too old. Recommend CUDA 11.8+"
    fi
else
    echo "   ⚠️  nvcc not found. CUDA might not be properly installed"
fi

# 3. Check conda environment
echo ""
echo "3️⃣ Checking conda environment..."
if command_exists conda; then
    echo "   Conda available: $(conda --version)"
    
    # Check if clickbait environment exists
    if conda env list | grep -q "clickbait"; then
        echo "   ✅ 'clickbait' environment found"
        
        # Activate and check key packages
        echo "   Checking key packages in environment..."
        source ~/anaconda3/etc/profile.d/conda.sh
        conda activate clickbait
        
        # Check PyTorch
        if python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')" 2>/dev/null; then
            echo "   ✅ PyTorch with CUDA support is working"
        else
            echo "   ❌ PyTorch or CUDA support issue detected"
        fi
        
        # Check transformers
        if python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null; then
            echo "   ✅ Transformers library available"
        else
            echo "   ❌ Transformers library not found"
        fi
        
        # Check PEFT
        if python -c "import peft; print(f'PEFT: {peft.__version__}')" 2>/dev/null; then
            echo "   ✅ PEFT library available"
        else
            echo "   ❌ PEFT library not found"
        fi
        
    else
        echo "   ❌ 'clickbait' environment not found"
        echo "   Please create it first: conda create -n clickbait python=3.9"
    fi
else
    echo "   ❌ Conda not found. Please install Anaconda/Miniconda"
fi

# 4. Set optimal environment variables
echo ""
echo "4️⃣ Setting optimal environment variables..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export MASTER_ADDR=localhost
export MASTER_PORT=29500

echo "   ✅ Environment variables set:"
echo "      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "      CUDA_LAUNCH_BLOCKING=0"
echo "      NCCL_P2P_DISABLE=0"
echo "      NCCL_IB_DISABLE=1"

# 5. Check disk space
echo ""
echo "5️⃣ Checking disk space..."
AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
echo "   Available space: ${AVAILABLE_SPACE}GB"

if [ "$AVAILABLE_SPACE" -ge 200 ]; then
    echo "   ✅ Sufficient disk space available"
else
    echo "   ⚠️  Warning: Recommend at least 200GB free space for model training"
fi

# 6. Create necessary directories
echo ""
echo "6️⃣ Creating necessary directories..."
mkdir -p fine-tuning/outputs
mkdir -p fine-tuning/logs
mkdir -p fine-tuning/configs
echo "   ✅ Directories created"

# 7. Set permissions
echo ""
echo "7️⃣ Setting script permissions..."
chmod +x fine-tuning/scripts/run_distributed_training.sh
chmod +x fine-tuning/scripts/setup_multi_gpu.sh
echo "   ✅ Script permissions set"

echo ""
echo "🎉 Setup completed!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate environment: conda activate clickbait"
echo "   2. Run training: ./fine-tuning/scripts/run_distributed_training.sh"
echo "   3. Monitor progress: tail -f fine-tuning/outputs/*/logs/*.log"
echo ""
echo "🚀 Ready for distributed training on 2x RTX A6000!" 