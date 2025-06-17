# 🎯 Clickbait Classification - Project Summary

## 📋 Tổng quan dự án

Đây là đồ án KLTN về **Phát hiện tiêu đề clickbait** sử dụng Deep Learning, được tối ưu hóa cho việc chạy trên **RTX 3050** (demo) và **A5000/A6000** (full production).

### ✨ Tính năng chính

- **Multi-GPU Support**: Tối ưu cho RTX 3050 (demo) và A5000/A6000 (production)
- **Multiple Model Architectures**: TinyBERT, DistilBERT, DeBERTa-v3, RoBERTa, LLaMA/Mistral LoRA
- **Demo & Full Training**: Quick demo với 100 samples và full training với 38K+ samples
- **Web Interface**: Gradio-based UI để test model interactively
- **Comprehensive Evaluation**: Chi tiết metrics, confusion matrix, error analysis

## 📊 Dataset

- **Total**: 38,517 samples
- **Train**: 30,812 (80%)
- **Val**: 3,851 (10%) 
- **Test**: 3,854 (10%)
- **Class Distribution**: 24% Clickbait, 76% No-Clickbait
- **Average Text Length**: ~245 characters

## 🏗️ Cấu trúc dự án

```
clickbait-classification/
├── data/                          # Dataset (đã preprocessing)
│   ├── train/                    # Training data
│   ├── val/                      # Validation data
│   └── test/                     # Test data
├── src/                          # Source code
│   ├── data_preprocessor.py      # Data preprocessing
│   ├── train_demo.py            # Demo training (RTX 3050)
│   ├── train_full.py            # Full training (A5000/A6000)
│   ├── inference.py             # Model inference
│   ├── eval.py                  # Model evaluation
│   └── utils.py                 # Utility functions
├── configs/
│   └── model_configs.py         # Model configurations
├── outputs/                      # Trained models
├── analysis_results/             # Data analysis plots
├── demo_ui.py                   # Web interface
├── run_project.py               # Main project runner
└── requirements.txt             # Dependencies
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create và activate virtual environment
conda create -n clickbait python=3.10
conda activate clickbait

# Install dependencies
pip install -r requirements.txt
```

### 2. Chạy Demo (RTX 3050)

```bash
# Chạy toàn bộ quy trình demo
python run_project.py --mode demo

# Hoặc từng bước:
python src/train_demo.py --model tinybert --quick_test
python src/inference.py --model_path outputs/tinybert_rtx3050_demo --test_examples
python demo_ui.py
```

### 3. Chạy Full Training (A5000/A6000)

```bash
# Full training với DeBERTa-v3
python run_project.py --mode full --full_model deberta_v3

# Với Weights & Biases logging
python src/train_full.py --model deberta_v3 --use_wandb
```

## 📱 Demo Models (RTX 3050 - 8GB VRAM)

| Model | Memory | Speed | Expected F1 | Use Case |
|-------|--------|-------|-------------|----------|
| **TinyBERT** | 2-4GB | Very Fast | ~0.63 | Quick testing |
| **DistilBERT** | 4-6GB | Fast | ~0.66 | Balanced demo |
| **ELECTRA-small** | 3-5GB | Fast | ~0.67 | Speed-focused |

```bash
# TinyBERT (fastest)
python src/train_demo.py --model tinybert

# DistilBERT (balanced)
python src/train_demo.py --model distilbert

# ELECTRA-small (best demo performance)
python src/train_demo.py --model electra_small
```

## 🚀 Production Models (A5000/A6000 - 24GB+ VRAM)

| Model | Memory | Expected F1 | Training Time | Use Case |
|-------|--------|-------------|---------------|----------|
| **DeBERTa-v3** | 12-16GB | ~0.72 | 2-3 hours | High performance |
| **DeBERTa-v3-Large** | 20-24GB | ~0.75 | 4-5 hours | Best performance |
| **RoBERTa-Large** | 14-18GB | ~0.70 | 3-4 hours | Reliable baseline |
| **LLaMA-2 LoRA** | 20-24GB | ~0.72 | 5-6 hours | Cutting-edge |

```bash
# DeBERTa-v3 (recommended)
python src/train_full.py --model deberta_v3

# DeBERTa-v3-Large (best performance)
python src/train_full.py --model deberta_v3_large

# LLaMA-2 với LoRA
python src/train_full.py --model llama2_lora
```

## 🎯 Usage Examples

### Interactive Testing

```bash
# Interactive mode
python src/inference.py --model_path outputs/tinybert_rtx3050_demo --interactive

# Test với examples có sẵn
python src/inference.py --model_path outputs/deberta_v3_full --test_examples

# Process file
python src/inference.py --model_path outputs/model --input_file texts.jsonl --output_file results.jsonl
```

### Web Interface

```bash
# Launch web demo
python demo_ui.py

# Access at: http://localhost:7860
```

### Model Evaluation

```bash
# Evaluate single model
python src/eval.py --model_dir outputs/deberta_v3_full --test_file data/test/data.jsonl --output_dir evaluation_results

# Compare multiple models
python src/eval.py --compare_models outputs/model1 outputs/model2 --test_file data/test/data.jsonl
```

## 📊 Results & Benchmarks

### Demo Results (100 samples, 1 epoch)

| Model | Accuracy | F1 | Precision | Recall | Training Time |
|-------|----------|----|-----------|---------| --------------|
| TinyBERT | 0.34 | 0.42 | 0.27 | 1.00 | ~30s |

*Note: Demo results với dataset nhỏ và 1 epoch chỉ để test setup*

### Expected Full Results (38K samples, 3-5 epochs)

| Model | Accuracy | F1 | Precision | Recall | Training Time |
|-------|----------|----|-----------|---------| --------------|
| DistilBERT | ~0.82 | ~0.66 | ~0.68 | ~0.64 | ~45min |
| DeBERTa-v3 | ~0.85 | ~0.72 | ~0.74 | ~0.70 | ~3h |
| DeBERTa-v3-Large | ~0.87 | ~0.75 | ~0.76 | ~0.74 | ~5h |

## 🔧 Configuration

### Model Configs

Tất cả model configs được định nghĩa trong `configs/model_configs.py`:

```python
from configs.model_configs import get_config, print_config_summary

# Show all available configs
print_config_summary()

# Get specific config
config = get_config("tinybert")
```

### Custom Training

```python
# Custom training script
from configs.model_configs import get_config
from src.utils import create_data_loaders

config = get_config("deberta_v3")
config.num_epochs = 10  # Modify epochs
config.learning_rate = 1e-5  # Modify learning rate

# Train với custom config
```

## 📈 Performance Tips

### RTX 3050 Optimization

- Sử dụng `--quick_test` flag để reduce epochs
- Batch size tối đa: 24 (TinyBERT), 16 (DistilBERT)
- Enable FP16 để save memory
- Use gradient accumulation nếu cần larger effective batch size

### A5000/A6000 Optimization

- Increase batch size: 32-64
- Use larger models: DeBERTa-v3-Large
- Enable TF32 trên A100 cho speedup
- Use multiple GPUs với `accelerate` config

## 🐛 Troubleshooting

### Common Issues

1. **GPU Out of Memory**
   ```bash
   # Reduce batch size
   config.batch_size_train = 8
   
   # Enable gradient accumulation
   config.gradient_accumulation_steps = 4
   ```

2. **Model Loading Error**
   ```bash
   # Check model path
   ls outputs/
   
   # Retrain if corrupted
   python src/train_demo.py --model tinybert
   ```

3. **Data Loading Error**
   ```bash
   # Re-run preprocessing
   python src/data_preprocessor.py
   ```

## 📚 References

- **Dataset**: Webis-Clickbait-17 Corpus
- **Models**: HuggingFace Transformers
- **Training**: PyTorch + Transformers Trainer
- **Web UI**: Gradio
- **Metrics**: scikit-learn

## 🤝 Contributing

### Development Workflow

1. Modify code in `src/`
2. Test với demo: `python src/train_demo.py --model tinybert --quick_test`
3. Full test: `python src/train_full.py --model deberta_v3`
4. Update configs nếu cần

### Adding New Models

1. Add config trong `configs/model_configs.py`
2. Test compatibility với `src/utils.py:check_gpu_compatibility()`
3. Run benchmark với `src/eval.py`

---

**🎯 Kết luận**: Dự án cung cấp solution đầy đủ cho clickbait classification từ demo nhanh trên RTX 3050 đến production-ready models trên A5000/A6000, với web interface và comprehensive evaluation tools. 