# Prompting Module

Thư mục này chứa tất cả các script và công cụ liên quan đến việc sử dụng prompting techniques cho bài toán phân loại clickbait.

## Cấu trúc thư mục

```
prompting/
├── scripts/              # Các script prompting và inference
│   ├── prompting_example.py     # Các ví dụ prompting cơ bản
│   ├── prompting_unified.py     # Prompting với unified approach
│   ├── improved_prompting.py    # Kỹ thuật prompting cải tiến
│   └── inference.py            # Inference với pre-trained models
├── outputs/              # Kết quả prompting
└── results/             # Kết quả đánh giá prompting
```

## Các script chính

### 1. prompting_example.py
Cung cấp các ví dụ prompting cơ bản và template cho việc phân loại clickbait.

**Sử dụng:**
```bash
python prompting/scripts/prompting_example.py
```

### 2. prompting_unified.py
Approach thống nhất cho prompting với nhiều kỹ thuật khác nhau (few-shot, chain-of-thought, etc.).

**Sử dụng:**
```bash
python prompting/scripts/prompting_unified.py
```

### 3. improved_prompting.py
Các kỹ thuật prompting cải tiến và optimization cho hiệu suất tốt hơn.

**Sử dụng:**
```bash
python prompting/scripts/improved_prompting.py
```

### 4. inference.py
Thực hiện inference với các pre-trained models sử dụng prompting techniques.

**Sử dụng:**
```bash
python prompting/scripts/inference.py
```

## Kỹ thuật Prompting

- **Zero-shot prompting**: Không cần ví dụ
- **Few-shot prompting**: Sử dụng một số ví dụ mẫu
- **Chain-of-thought**: Hướng dẫn mô hình suy luận từng bước
- **Self-consistency**: Tạo nhiều câu trả lời và chọn kết quả tốt nhất

## Ghi chú

- Kết quả được lưu trong thư mục `outputs/` và `results/`
- Sử dụng dữ liệu từ `../shared/data/` để test
- Tham khảo `../docs/PROMPTING_GUIDE.md` để hiểu rõ hơn về prompting techniques 