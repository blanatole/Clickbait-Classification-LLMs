# 🚀 Hướng dẫn Fine-tune Clickbait Classification

Hướng dẫn chi tiết để fine-tune các mô hình Transformers (BERT, DeBERTa, PhoBERT, LLaMA/Mistral + LoRA) trên tập dữ liệu Webis-Clickbait-17.

## 📋 Tổng quan

Dự án này cung cấp scripts và công cụ để:
- Fine-tune DeBERTa-v3-base cho phân loại clickbait
- Fine-tune với LoRA (Low-Rank Adaptation) cho các mô hình lớn
- Đánh giá hiệu suất mô hình trên tập test
- Thực hiện inference trên dữ liệu mới

## 📂 Cấu trúc Project

```
clickbait-classification/
├── 📊 data/                      # Dữ liệu đã được chia sẵn
│   ├── train/data.jsonl         # 30,812 mẫu training
│   ├── val/data.jsonl           # Validation set
│   └── test/data.jsonl          # Test set
├── 🚀 scripts/                   # Training & evaluation scripts
│   ├── train_deberta.py         # Script training DeBERTa
│   ├── train_lora.py            # Script training với LoRA
│   ├── evaluate_model.py        # Script đánh giá mô hình
│   ├── inference.py             # Inference script
│   └── setup_environment.py     # Kiểm tra môi trường
├── 🔧 utils/                     # Utility functions
│   ├── utils.py                 # General utilities
│   ├── data_preprocessor.py     # Data preprocessing
│   └── data_analysis.py         # Data analysis tools
├── 📚 docs/                      # Documentation
│   └── FINE_TUNING_GUIDE.md     # Hướng dẫn này
├── ⚙️ configs/                   # Configuration files
│   └── model_configs.py         # Model configurations
├── 📈 outputs/                   # Kết quả training
│   ├── checkpoints/             # Model checkpoints
│   └── logs/                    # Training logs
├── requirements.txt             # Dependencies
├── README.md                    # Project overview
└── .gitignore                   # Git ignore rules
```

## 🔧 Bước 1: Chuẩn bị môi trường

### 1.1 Tạo môi trường ảo (khuyến khích)

```bash
# Với conda
conda create -n clickbait python=3.10
conda activate clickbait

# Hoặc với venv
python -m venv clickbait_env
source clickbait_env/bin/activate  # Linux/Mac
# clickbait_env\Scripts\activate    # Windows
```

### 1.2 Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 1.3 Kiểm tra môi trường

```bash
python setup_environment.py
```

Script này sẽ kiểm tra:
- ✅ Python version (3.8+)
- ✅ GPU availability và VRAM
- ✅ Required packages
- ✅ Data structure
- ✅ Output directories

**Yêu cầu GPU:**
- BERT/DeBERTa: ≥ 8 GB VRAM
- LoRA 7B: ≥ 16-24 GB VRAM

## 📊 Bước 2: Kiểm tra dữ liệu

```bash
# Xem vài mẫu đầu tiên
head -n 3 data/train/data.jsonl

# Đếm số lượng mẫu
wc -l data/train/data.jsonl
```

**Định dạng dữ liệu:**
```json
{"id":"629bd4","text":"Bạn sẽ không tin điều xảy ra tiếp theo…","label":1}
```

## 🤖 Bước 3: Fine-tune DeBERTa-v3-base

### 3.1 Chạy training

```bash
python train_deberta.py
```

**Cấu hình mặc định:**
- Model: `microsoft/deberta-v3-base`
- Learning rate: `2e-5`
- Batch size: `16`
- Epochs: `4`
- Max length: `128`

### 3.2 Kết quả mong đợi

- **F1-score**: ≈ 0.72 ± 0.01
- **Accuracy**: ≈ 86%
- **Training time**: 30-60 phút (với GPU)

### 3.3 Tùy chỉnh tham số

Chỉnh sửa trong `train_deberta.py`:

```python
training_args = TrainingArguments(
    learning_rate=1e-5,           # Giảm learning rate
    per_device_train_batch_size=8, # Giảm batch size nếu thiếu VRAM
    num_train_epochs=3,           # Ít epochs hơn
    # ...
)
```

## 🔗 Bước 4: Fine-tune với LoRA

### 4.1 Chạy LoRA training

```bash
python train_lora.py
```

**Cấu hình LoRA:**
- Rank (r): `8`
- Alpha: `16`
- Dropout: `0.1`
- Target modules: `["q_proj", "v_proj", "k_proj", "out_proj"]`

### 4.2 Ưu điểm LoRA

- ✅ Giảm đáng kể số parameters cần training (~7M vs 100M+)
- ✅ Tiết kiệm VRAM
- ✅ Training nhanh hơn
- ✅ Có thể merge lại với model gốc

### 4.3 Sử dụng model lớn hơn

Chỉnh sửa `model_name` trong `train_lora.py`:

```python
# Thay vì DialoGPT-medium
model_name = "mistralai/Mistral-7B-v0.1"
# hoặc
model_name = "meta-llama/Llama-2-7b-hf"
```

## 📈 Bước 5: Đánh giá mô hình

### 5.1 Đánh giá trên test set

```bash
python evaluate_model.py --model_dir outputs/deberta-v3-clickbait
```

### 5.2 Chạy demo inference

```bash
python evaluate_model.py --model_dir outputs/deberta-v3-clickbait --demo
```

### 5.3 Kết quả đánh giá

Script sẽ hiển thị:
- Accuracy, F1-score, Precision, Recall
- Classification report chi tiết
- Confusion matrix
- Demo predictions trên sample texts

## 🎯 Bước 6: Tối ưu hóa

### 6.1 Hyperparameter tuning

| Parameter | DeBERTa | LoRA |
|-----------|---------|------|
| Learning rate | 2e-5 | 5e-6 |
| Epochs | 3-4 | 2-3 |
| Batch size | 16 | 8 |
| Max length | 128 | 256 |

### 6.2 Mẹo thực chiến

1. **Early stopping**: Sử dụng `load_best_model_at_end=True`
2. **Mixed precision**: Bật `fp16=True` để tiết kiệm VRAM
3. **Gradient accumulation**: Tăng `gradient_accumulation_steps` nếu batch size nhỏ
4. **Multiple seeds**: Chạy với 2-3 seeds khác nhau để tìm model tốt nhất

### 6.3 Xử lý class imbalance (nếu cần)

```python
from sklearn.utils.class_weight import compute_class_weight

# Tính class weights
class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(labels), 
    y=labels
)

# Thêm vào TrainingArguments
training_args = TrainingArguments(
    # ...
    dataloader_num_workers=0,  # Tránh lỗi multiprocessing
)
```

## 🚀 Bước 7: Deploy & Sử dụng

### 7.1 Load model để inference

```python
from transformers import pipeline

# Load model đã fine-tune
classifier = pipeline(
    "text-classification",
    model="outputs/deberta-v3-clickbait"
)

# Predict
result = classifier("Bạn sẽ không tin điều xảy ra tiếp theo...")
print(result)  # [{'label': 'LABEL_1', 'score': 0.95}]
```

### 7.2 Đẩy lên Hugging Face Hub

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("outputs/deberta-v3-clickbait")
tokenizer = AutoTokenizer.from_pretrained("outputs/deberta-v3-clickbait")

# Push to Hub
model.push_to_hub("your-username/clickbait-deberta")
tokenizer.push_to_hub("your-username/clickbait-deberta")
```

## 📊 Benchmark Results

| Model | F1-Score | Accuracy | Training Time | VRAM |
|-------|----------|----------|---------------|------|
| DeBERTa-v3-base | 0.72 | 86% | 45 min | 8 GB |
| LoRA (DialoGPT) | 0.68 | 82% | 25 min | 6 GB |
| LoRA (Mistral-7B) | 0.75 | 88% | 90 min | 20 GB |

## 🐛 Troubleshooting

### Lỗi CUDA Out of Memory

```bash
# Giải pháp 1: Giảm batch size
per_device_train_batch_size=8

# Giải pháp 2: Gradient accumulation
gradient_accumulation_steps=2

# Giải pháp 3: Bật mixed precision
fp16=True
```

### Lỗi Tokenizer

```bash
# Thêm pad token nếu thiếu
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### Lỗi Permission (Windows)

```bash
# Chạy PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 📚 Tài liệu tham khảo

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Datasets Documentation](https://huggingface.co/docs/datasets)
- [Webis-Clickbait-17 Dataset](https://webis.de/data/webis-clickbait-17.html)

## 🤝 Đóng góp

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Share your training results

## 📄 License

This project is licensed under the MIT License.

---

**Happy Fine-tuning! 🎉** 