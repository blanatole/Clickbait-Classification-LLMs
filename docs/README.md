# 🚀 Hướng dẫn Fine-tune Clickbait Classification từ A-Z

**Hướng dẫn chi tiết từ đầu đến cuối để fine-tune các mô hình phân loại clickbait trên GPU RTX A5000**

---

## 📋 Mục lục

1. [Tổng quan Project](#1-tổng-quan-project)
2. [Yêu cầu Hệ thống](#2-yêu-cầu-hệ-thống)
3. [Cài đặt Môi trường](#3-cài-đặt-môi-trường)
4. [Chuẩn bị Dữ liệu](#4-chuẩn-bị-dữ-liệu)
5. [Chạy Training](#5-chạy-training)
6. [Đánh giá Kết quả](#6-đánh-giá-kết-quả)
7. [Troubleshooting](#7-troubleshooting)
8. [Best Practices](#8-best-practices)

---

## 1. Tổng quan Project

### 🎯 Mục tiêu
Fine-tune các mô hình state-of-the-art để phân loại tiêu đề clickbait vs non-clickbait trên dataset Webis-Clickbait-17.

### 📊 Dataset
- **Training**: 30,812 samples
- **Validation**: 3,851 samples  
- **Test**: 3,854 samples
- **Format**: `{"id": "abc123", "text": "Tiêu đề", "label": 1}`

### 🤖 Mô hình hỗ trợ
- **BERT Family**: BERT-base, DeBERTa-v3-base, BERT-large
- **Large Language Models**: Mistral-7B, Llama-2-7B, Llama-3-8B, Llama-2-13B
- **Training Methods**: Full fine-tune, LoRA, QLoRA

---

## 2. Yêu cầu Hệ thống

### 💻 Hardware
```
Minimum:
- GPU: 8GB VRAM (cho BERT-base)
- RAM: 16GB+
- Storage: 50GB free space

Recommended (RTX A5000):
- GPU: 24GB VRAM
- RAM: 32GB+
- Storage: 100GB free space
```

### 🐍 Software
```
- Python: 3.8+
- CUDA: 11.8+ hoặc 12.x
- Git
- Windows 10/11 hoặc Linux
```

---

## 3. Cài đặt Môi trường

### 3.1 Clone Repository

```bash
git clone <repository-url>
cd clickbait-classification
```

### 3.2 Tạo Virtual Environment

**Windows:**
```cmd
python -m venv clickbait_env
clickbait_env\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv clickbait_env
source clickbait_env/bin/activate
```

### 3.3 Cài đặt Dependencies

```bash
# Cài đặt PyTorch với CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cài đặt các thư viện khác
pip install -r requirements.txt
```

### 3.4 Kiểm tra Cài đặt

```bash
python scripts/setup_environment.py
```

**Output mong đợi:**
```
🔧 GPU: NVIDIA GeForce RTX A5000 (24.0 GB)
✅ Python version is compatible
✅ All required packages are installed!
✅ Data structure is correct!
```

---

## 4. Chuẩn bị Dữ liệu

### 4.1 Kiểm tra Dữ liệu

```bash
# Kiểm tra số lượng samples
wc -l data/train/data.jsonl    # Should be 30812
wc -l data/val/data.jsonl      # Should be 3851  
wc -l data/test/data.jsonl     # Should be 3854

# Xem vài samples đầu tiên
head -n 3 data/train/data.jsonl
```

### 4.2 Cấu trúc Dữ liệu
```
data/
├── train/data.jsonl    # Training data
├── val/data.jsonl      # Validation data
└── test/data.jsonl     # Test data
```

---

## 5. Chạy Training

### 5.1 🌟 Phương pháp 1: Interactive Guide (Khuyến nghị)

```bash
python scripts/quick_start_a5000.py
```

**Interactive menu sẽ hiển thị:**
```
📋 AVAILABLE MODELS FOR RTX A5000 (24GB)

🔥 BERT Family (Recommended for beginners)
   1. BERT-base-uncased    | 110M params, ~45min, F1≈0.70
   2. DeBERTa-v3-base      | 184M params, ~60min, F1≈0.72
   3. BERT-large-uncased   | 340M params, ~90min, F1≈0.73

🚀 Large Language Models with LoRA
   4. Mistral-7B-v0.2      | 7B params, ~120min, F1≈0.72
   5. Mistral-7B-Instruct  | 7B params, ~90min, F1≈0.73
   6. Llama-2-7B           | 7B params, ~120min, F1≈0.70
   7. Llama-3-8B           | 8B params, ~150min, F1≈0.74
   8. Llama-2-13B          | 13B params, ~180min, F1≈0.75

   9. All BERT family models
   10. All LLM models  
   11. ALL models (full benchmark)

Enter your choice (1-11):
```

### 5.2 🔧 Phương pháp 2: Manual Training

#### A. Training BERT Family Models

```bash
# Train một model cụ thể
python scripts/train_bert_family.py --model bert-base-uncased
python scripts/train_bert_family.py --model deberta-v3-base
python scripts/train_bert_family.py --model bert-large-uncased

# Train tất cả BERT family models
python scripts/train_bert_family.py --model all
```

#### B. Training LLM với LoRA

```bash
# Train Mistral models
python scripts/train_llm_lora.py --model mistral-7b-v0.2
python scripts/train_llm_lora.py --model mistral-7b-instruct

# Train Llama models  
python scripts/train_llm_lora.py --model llama2-7b
python scripts/train_llm_lora.py --model llama3-8b
python scripts/train_llm_lora.py --model llama2-13b

# Train tất cả LLM models
python scripts/train_llm_lora.py --model all
```

### 5.3 🏁 Phương pháp 3: Full Benchmark Suite

```bash
# Chạy tất cả experiments (12-15 giờ)
python scripts/run_all_experiments.py
```

**Lưu ý:** Script sẽ hỏi xác nhận cho từng experiment:
```
🎬 EXPERIMENT 1/6: BERT Family Models
Expected duration: ~180 minutes
Models: bert-base-uncased, deberta-v3-base, bert-large-uncased

Proceed with BERT Family Models? (y/n/skip):
```

### 5.4 📊 Monitoring Training

**Trong terminal khác, monitor GPU usage:**
```bash
# Windows
nvidia-smi -l 1

# Linux  
watch -n 1 nvidia-smi
```

**Training logs sẽ hiển thị:**
```
🚀 TRAINING BERT-BASE-UNCASED
============================================================
🤖 Loading google-bert/bert-base-uncased...
📊 Loading and tokenizing dataset...
Dataset loaded: DatasetDict({
    train: Dataset({features: ['id', 'text', 'label'], num_rows: 30812})
    validation: Dataset({features: ['id', 'text', 'label'], num_rows: 3851})
    test: Dataset({features: ['id', 'text', 'label'], num_rows: 3854})
})

🏋️ Starting training...
{'train_runtime': 2847.23, 'train_samples_per_second': 43.25, ...}

📊 RESULTS FOR BERT-BASE-UNCASED:
   Accuracy: 0.8612
   F1 (weighted): 0.7089
   F1 (macro): 0.7012
   F1 (binary): 0.7089
   Training time: 0h 47m 27s
   Model saved to: outputs/bert-base-uncased-a5000
```

---

## 6. Đánh giá Kết quả

### 6.1 Generate Comparison Report

```bash
python scripts/benchmark_results.py --save_csv
```

**Output:**
```
📊 BENCHMARK RESULTS ANALYSIS
==================================================
📈 Found results for 8 models

🏆 MODEL PERFORMANCE RANKING:
================================================================================
               Model Parameters Training Method  F1 Score  Accuracy Training Time
         llama2-13b       13B     LoRA + 8bit    0.7543    0.8789         3h 12m
        llama3-8b         8B     LoRA + 4bit    0.7421    0.8712         2h 34m
  mistral-7b-instruct     7B     LoRA + 4bit    0.7356    0.8689         1h 32m
   bert-large-uncased    340M  Full Fine-tune    0.7298    0.8656         1h 28m
      deberta-v3-base    184M  Full Fine-tune    0.7203    0.8612         1h 02m
     mistral-7b-v0.2      7B     LoRA + 4bit    0.7156    0.8578         2h 08m
          llama2-7b       7B     LoRA + 4bit    0.7098    0.8534         2h 15m
    bert-base-uncased    110M  Full Fine-tune    0.7089    0.8612         0h 47m

💡 KEY INSIGHTS:
==================================================
🥇 Best Overall: llama2-13b (F1: 0.7543)
🥇 Best LoRA: llama2-13b (F1: 0.7543)
🥇 Best Full Fine-tune: bert-large-uncased (F1: 0.7298)

✅ Analysis complete! Results saved to outputs/
```

### 6.2 Detailed Model Evaluation

```bash
# Evaluate một model cụ thể
python scripts/evaluate_model.py --model_dir outputs/llama2-13b-lora-a5000 --demo

# Output sẽ bao gồm:
# - Detailed metrics
# - Classification report  
# - Confusion matrix
# - Demo inference trên sample texts
```

### 6.3 Inference trên Text Mới

```python
from transformers import pipeline

# Load model đã train
classifier = pipeline(
    "text-classification",
    model="outputs/llama2-13b-lora-a5000"
)

# Test trên text mới
texts = [
    "Bạn sẽ không tin điều xảy ra tiếp theo...",
    "Chính phủ công bố kế hoạch kinh tế mới",
    "7 bí mật mà bác sĩ không muốn bạn biết!"
]

for text in texts:
    result = classifier(text)
    print(f"'{text}' -> {result[0]['label']} ({result[0]['score']:.3f})")
```

---

## 7. Troubleshooting

### 7.1 🚨 Lỗi thường gặp

#### **CUDA Out of Memory**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Giải pháp:**
```bash
# Giảm batch size trong scripts/train_bert_family.py
"batch_size": 16,  # Thay vì 32

# Hoặc tăng gradient accumulation
"gradient_accumulation_steps": 4,  # Thay vì 2

# Hoặc dùng model nhỏ hơn
python scripts/train_bert_family.py --model bert-base-uncased  # Thay vì bert-large
```

#### **Package Import Errors**
```
ModuleNotFoundError: No module named 'transformers'
```

**Giải pháp:**
```bash
# Đảm bảo virtual environment được activate
source clickbait_env/bin/activate  # Linux/Mac
clickbait_env\Scripts\activate     # Windows

# Reinstall packages
pip install --upgrade -r requirements.txt
```

#### **Model Download Errors**
```
HTTPSConnectionPool: Max retries exceeded
```

**Giải pháp:**
```bash
# Set proxy nếu cần
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port

# Hoặc download manual
from transformers import AutoModel
model = AutoModel.from_pretrained("microsoft/deberta-v3-base", force_download=True)
```

### 7.2 🔧 Performance Optimization

#### **Để tăng tốc training:**
```python
# Trong training arguments
training_args = TrainingArguments(
    fp16=True,                    # Mixed precision
    dataloader_num_workers=4,     # Parallel data loading
    gradient_checkpointing=True,  # Save memory
    warmup_steps=100,             # Faster convergence
)
```

#### **Để giảm memory usage:**
```python
training_args = TrainingArguments(
    per_device_train_batch_size=8,    # Smaller batches
    gradient_accumulation_steps=4,    # Maintain effective batch size
    fp16=True,                        # Half precision
    dataloader_pin_memory=False,      # Save memory
)
```

---

## 8. Best Practices

### 8.1 🎯 Lựa chọn Model

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| **Quick Prototype** | BERT-base-uncased | Nhanh, ít tài nguyên |
| **Best Accuracy** | Llama-2-13B + QLoRA | Performance cao nhất |
| **Balanced** | DeBERTa-v3-base | Tốt và training nhanh |
| **Production** | Mistral-7B-Instruct | Instruction-tuned, reliable |

### 8.2 ⚙️ Training Configuration

#### **Cho RTX A5000 (24GB):**
```python
# BERT Family - Aggressive settings
BERT_CONFIG = {
    "batch_size": 48,
    "learning_rate": 2e-5,
    "epochs": 4,
    "fp16": True
}

# LLM LoRA - Conservative settings  
LLM_CONFIG = {
    "batch_size": 10,
    "learning_rate": 5e-6,
    "epochs": 3,
    "quantization": "4bit"
}
```

#### **Cho GPU nhỏ hơn (8-16GB):**
```python
# Giảm batch size và tăng gradient accumulation
SMALL_GPU_CONFIG = {
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "fp16": True
}
```

### 8.3 📊 Monitoring & Logging

```bash
# Monitor GPU real-time
nvidia-smi -l 1

# Monitor training logs
tail -f outputs/model-name/trainer_state.json

# Check disk space
df -h outputs/
```

### 8.4 💾 Save Storage Space

```bash
# Chỉ keep best checkpoints
training_args = TrainingArguments(
    save_total_limit=1,           # Keep only best checkpoint
    load_best_model_at_end=True,
)

# Clean up sau training
rm -rf outputs/*/checkpoint-*     # Remove intermediate checkpoints
```

### 8.5 🔄 Experiment Management

```bash
# Organize experiments by date
mkdir outputs/2024-01-15-bert-family
mv outputs/bert-* outputs/2024-01-15-bert-family/

# Log all experiments
python scripts/run_all_experiments.py > experiment_log_$(date +%Y%m%d).txt 2>&1
```

---

## 🎉 Hoàn thành!

Sau khi hoàn thành các bước trên, bạn sẽ có:

✅ **Trained Models**: Multiple fine-tuned models trong `outputs/`
✅ **Benchmark Results**: Comparison table và visualizations
✅ **Production-ready Models**: Saved models có thể deploy
✅ **Detailed Logs**: Training history và performance metrics

### 📁 Output Structure
```
outputs/
├── bert-base-uncased-a5000/         # Trained BERT model
├── llama2-13b-lora-a5000/          # Best performing model
├── benchmark_results.csv           # Comparison table
├── benchmark_summary.json          # Summary results
├── experiment_log.json             # Full experiment history
└── experiment_log_20240115.txt     # Training logs
```

### 🚀 Next Steps
1. **Deploy model** cho production
2. **Fine-tune thêm** trên domain-specific data
3. **Optimize inference** speed cho real-time applications
4. **A/B test** different models trong production

---

**💡 Need Help?**
- Xem `FINE_TUNING_GUIDE.md` cho advanced topics
- Check GitHub Issues cho community support
- Monitor GPU với `nvidia-smi` khi training

**Happy Fine-tuning! 🎯** 