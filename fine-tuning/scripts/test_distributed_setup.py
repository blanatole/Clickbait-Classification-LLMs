#!/usr/bin/env python3
"""
Test script để kiểm tra distributed training setup
Run: python -m torch.distributed.launch --nproc_per_node=2 test_distributed_setup.py
"""

import torch
import torch.distributed as dist
import os
import time
from datetime import datetime

def test_basic_setup():
    """Test basic PyTorch and CUDA setup"""
    print(f"🔧 Testing basic setup...")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

def test_distributed_setup():
    """Test distributed training setup"""
    print(f"\n🌐 Testing distributed setup...")
    
    # Initialize distributed training
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        print(f"   Process rank: {rank}")
        print(f"   World size: {world_size}")
        print(f"   Local rank: {local_rank}")
        
        # Initialize process group
        dist.init_process_group(backend="nccl")
        
        # Set device
        torch.cuda.set_device(local_rank)
        device = torch.cuda.current_device()
        print(f"   Using device: {device}")
        
        # Test tensor operations
        print(f"   Testing tensor operations...")
        tensor = torch.randn(100, 100).cuda()
        result = torch.mm(tensor, tensor.T)
        print(f"   ✅ Tensor operations working on GPU {device}")
        
        # Test communication
        print(f"   Testing distributed communication...")
        test_tensor = torch.tensor([rank], dtype=torch.float32).cuda()
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        expected = sum(range(world_size))
        
        if test_tensor.item() == expected:
            print(f"   ✅ Distributed communication working")
        else:
            print(f"   ❌ Distributed communication failed: {test_tensor.item()} != {expected}")
        
        # Cleanup
        dist.destroy_process_group()
        
    else:
        print("   ⚠️  Not running in distributed mode")
        print("   Run with: python -m torch.distributed.launch --nproc_per_node=2 test_distributed_setup.py")

def test_memory():
    """Test GPU memory"""
    print(f"\n💾 Testing GPU memory...")
    
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        
        # Get memory info
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        cached = torch.cuda.memory_reserved(i) / 1024**3
        free = total_memory - cached
        
        print(f"   GPU {i}: {free:.1f}GB free / {total_memory:.1f}GB total")
        
        # Test memory allocation
        try:
            test_tensor = torch.randn(1000, 1000, 100).cuda(i)  # ~400MB
            print(f"   ✅ Memory allocation test passed on GPU {i}")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"   ❌ Memory allocation failed on GPU {i}: {e}")

def test_libraries():
    """Test required libraries"""
    print(f"\n📚 Testing required libraries...")
    
    try:
        import transformers
        print(f"   ✅ Transformers: {transformers.__version__}")
    except ImportError:
        print(f"   ❌ Transformers not installed")
    
    try:
        import peft
        print(f"   ✅ PEFT: {peft.__version__}")
    except ImportError:
        print(f"   ❌ PEFT not installed")
    
    try:
        import datasets
        print(f"   ✅ Datasets: {datasets.__version__}")
    except ImportError:
        print(f"   ❌ Datasets not installed")
    
    try:
        import accelerate
        print(f"   ✅ Accelerate: {accelerate.__version__}")
    except ImportError:
        print(f"   ⚠️  Accelerate not installed (optional)")

def test_nccl():
    """Test NCCL availability"""
    print(f"\n🔗 Testing NCCL...")
    
    if torch.distributed.is_nccl_available():
        print(f"   ✅ NCCL is available")
        
        # Check NCCL version if possible
        try:
            import torch.distributed as dist
            print(f"   NCCL backend supported: {dist.is_available()}")
        except:
            pass
    else:
        print(f"   ❌ NCCL not available")

def main():
    print(f"🧪 Distributed Training Setup Test")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    
    test_basic_setup()
    test_libraries()
    test_nccl()
    test_memory()
    test_distributed_setup()
    
    print(f"\n{'='*50}")
    print(f"🎉 Test completed!")
    
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        if rank == 0:
            print(f"\n💡 To run actual training:")
            print(f"   ./fine-tuning/scripts/run_distributed_training.sh")

if __name__ == "__main__":
    main() 