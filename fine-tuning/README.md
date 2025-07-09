# Fine-tuning Module

Thư mục này chứa tất cả các script và công cụ liên quan đến việc fine-tuning các mô hình LLM cho bài toán phân loại clickbait.

## Cấu trúc thư mục

```
fine-tuning/
├── scripts/           # Các script training và evaluation
│   ├── train_llm_lora.py      # Fine-tuning LLM với LoRA
│   ├── train_bert_family.py   # Fine-tuning BERT family models
│   └── evaluate_model.py      # Đánh giá mô hình đã train
├── outputs/           # Kết quả training (checkpoints, logs)
└── models/           # Mô hình đã fine-tune
```

## Các script chính

### 1. train_llm_lora.py
Fine-tuning các mô hình LLM lớn (như Llama, Qwen) sử dụng kỹ thuật LoRA để tiết kiệm tài nguyên.

**Sử dụng:**
```bash
python fine-tuning/scripts/train_llm_lora.py
```

### 2. train_bert_family.py  
Fine-tuning các mô hình BERT family (BERT, RoBERTa, DistilBERT) cho classification task.

**Sử dụng:**
```bash
python fine-tuning/scripts/train_bert_family.py
```

### 3. evaluate_model.py
Đánh giá hiệu suất của các mô hình đã được fine-tune.

**Sử dụng:**
```bash
python fine-tuning/scripts/evaluate_model.py
```

## Ghi chú

- Tất cả các mô hình được lưu trong thư mục `models/`
- Logs và checkpoints được lưu trong `outputs/`
- Cần sử dụng dữ liệu từ `../shared/data/` để training
- Các utility functions từ `../shared/utils/` hỗ trợ quá trình training 