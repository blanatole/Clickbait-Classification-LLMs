# 🎯 Clickbait Classification Fine-Tuning

Dự án fine-tune các mô hình Transformers (BERT, DeBERTa, PhoBERT, LLaMA/Mistral + LoRA) cho bài toán phân loại clickbait trên tập dữ liệu Webis-Clickbait-17.

## 📂 Cấu trúc Project

```
clickbait-classification/
├── 📊 data/                      # Dữ liệu đã được chia sẵn
│   ├── train/data.jsonl         # 30,812 mẫu training
│   ├── val/data.jsonl           # Validation set  
│   └── test/data.jsonl          # Test set
├── 🚀 scripts/                   # Training & evaluation scripts
│   ├── train_deberta.py         # Fine-tune DeBERTa-v3-base
│   ├── train_lora.py            # Fine-tune với LoRA
│   ├── evaluate_model.py        # Đánh giá mô hình
│   ├── inference.py             # Inference script
│   └── setup_environment.py     # Kiểm tra môi trường
├── 🔧 utils/                     # Utility functions (legacy)
│   ├── utils.py                 # General utilities
│   ├── data_preprocessor.py     # Data preprocessing
│   └── data_analysis.py         # Data analysis tools
├── 📚 docs/                      # Documentation
│   └── FINE_TUNING_GUIDE.md     # Hướng dẫn chi tiết
├── ⚙️ configs/                   # Configuration files
│   └── model_configs.py         # Model configurations
├── 📈 outputs/                   # Training outputs
│   ├── checkpoints/             # Model checkpoints
│   └── logs/                    # Training logs
├── requirements.txt             # Dependencies
├── README.md                    # This file
└── .gitignore                   # Git ignore rules
```

## 🚀 Quick Start

### For RTX A5000 Users (Recommended)

```bash
# Interactive training guide with all optimized models
python scripts/quick_start_a5000.py
```

### Manual Training

#### 1. Check environment
```bash
python scripts/setup_environment.py
```

#### 2. Train specific models
```bash
# BERT family models (optimized batch sizes)
python scripts/train_bert_family.py --model bert-base-uncased
python scripts/train_bert_family.py --model deberta-v3-base
python scripts/train_bert_family.py --model all

# Large Language Models with QLoRA
python scripts/train_llm_lora.py --model mistral-7b-v0.2
python scripts/train_llm_lora.py --model llama3-8b
python scripts/train_llm_lora.py --model all
```

#### 3. Run full benchmark suite
```bash
python scripts/run_all_experiments.py
```

#### 4. Generate comparison results
```bash
python scripts/benchmark_results.py --save_csv
```

## 📊 Kết quả mong đợi

| Model | F1-Score | Accuracy | Training Time | VRAM |
|-------|----------|----------|---------------|------|
| DeBERTa-v3-base | **0.72** | **86%** | 45 min | 8 GB |
| LoRA (DialoGPT) | 0.68 | 82% | 25 min | 6 GB |
| LoRA (Mistral-7B) | **0.75** | **88%** | 90 min | 20 GB |

## 🔧 Yêu cầu hệ thống

- **Python**: 3.8+
- **GPU**: CUDA-compatible (khuyến nghị)
  - BERT/DeBERTa: ≥ 8 GB VRAM
  - LoRA 7B: ≥ 16-24 GB VRAM
- **RAM**: ≥ 16 GB
- **Storage**: ≥ 10 GB free space

## 📋 Dependencies

```bash
pip install -r requirements.txt
```

**Core libraries:**
- `torch>=2.1.0` - PyTorch
- `transformers>=4.35.0` - Hugging Face Transformers
- `datasets>=2.14.0` - Dataset handling
- `peft>=0.6.0` - LoRA implementation
- `scikit-learn>=1.3.0` - Metrics và evaluation

## 🎯 Usage Examples

### Training với custom parameters

```python
# Trong train_deberta.py, tùy chỉnh:
training_args = TrainingArguments(
    learning_rate=1e-5,           # Giảm learning rate
    per_device_train_batch_size=8, # Giảm batch size nếu thiếu VRAM
    num_train_epochs=3,           # Ít epochs hơn
    fp16=True,                    # Mixed precision
)
```

### Inference trên text mới

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="outputs/deberta-v3-clickbait"
)

text = "Bạn sẽ không tin điều xảy ra tiếp theo..."
result = classifier(text)
print(result)  # [{'label': 'LABEL_1', 'score': 0.95}]
```

## 📚 Documentation

Xem **[docs/FINE_TUNING_GUIDE.md](docs/FINE_TUNING_GUIDE.md)** để có hướng dẫn chi tiết từng bước.

## 🤝 Contributions

Mọi contributions đều được chào đón! Hãy:
- Report bugs
- Suggest improvements  
- Add new features
- Share your training results

## 📄 License

This project is licensed under the MIT License.

---

**Happy Fine-tuning! 🎉**
