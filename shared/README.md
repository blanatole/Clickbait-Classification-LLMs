# Shared Module

Thư mục này chứa các resources và utilities chung được sử dụng bởi cả fine-tuning và prompting modules.

## Cấu trúc thư mục

```
shared/
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── utils.py             # Các hàm tiện ích chung
│   ├── data_preprocessor.py # Xử lý dữ liệu
│   ├── data_analysis.py     # Phân tích dữ liệu
│   └── preprocess_clickbait_alpaca.py # Preprocessing cho Alpaca format
├── scripts/                 # Scripts chung và setup
│   ├── __init__.py
│   ├── setup_environment.py     # Setup môi trường
│   ├── run_all_experiments.py   # Chạy tất cả experiments
│   ├── benchmark_results.py     # Benchmark và so sánh kết quả
│   ├── run_evaluation.py        # Evaluation script chung
│   ├── test_api.py              # Test API connections
│   ├── test_api_unified.py      # Test unified API
│   └── test_available_models.py # Test model availability
└── data/                    # Dữ liệu training và testing
    ├── train/
    ├── val/
    └── test/
```

## Utility Functions

### utils.py
Chứa các hàm tiện ích chung cho cả fine-tuning và prompting.

### data_preprocessor.py
Xử lý và chuẩn bị dữ liệu cho training và testing.

### data_analysis.py
Phân tích và thống kê dữ liệu clickbait.

## Scripts chung

### setup_environment.py
Setup môi trường phát triển, cài đặt dependencies và cấu hình.

**Sử dụng:**
```bash
python shared/scripts/setup_environment.py
```

### run_all_experiments.py
Chạy tất cả các experiments cho cả fine-tuning và prompting.

**Sử dụng:**
```bash
python shared/scripts/run_all_experiments.py
```

### benchmark_results.py
So sánh và benchmark kết quả giữa các phương pháp khác nhau.

**Sử dụng:**
```bash
python shared/scripts/benchmark_results.py
```

## Dữ liệu

Thư mục `data/` chứa:
- `train/`: Dữ liệu training
- `val/`: Dữ liệu validation
- `test/`: Dữ liệu testing

Dữ liệu được chia sẵn và sẵn sàng sử dụng cho cả fine-tuning và prompting.

## Ghi chú

- Tất cả modules khác đều phụ thuộc vào shared module
- Cần chạy setup_environment.py trước khi sử dụng các script khác
- Utils functions được import và sử dụng bởi các script trong fine-tuning và prompting 