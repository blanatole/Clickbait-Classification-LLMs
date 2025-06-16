# Cấu Trúc Dự Án: Khai thác LLM để Phát hiện Clickbait

## 📁 Cấu Trúc Thư Mục

```
clickbait-classification/
│
├── 📁 data/                              # Dữ liệu thô và đã xử lý
│   ├── raw/                              # Dữ liệu gốc từ Webis-Clickbait-17
│   ├── processed/                        # Dữ liệu đã tiền xử lý
│   └── splits/                           # Dữ liệu đã chia train/val/test
│
├── 📁 src/                               # Source code chính
│   ├── 📁 data_processing/               # Xử lý dữ liệu
│   │   ├── __init__.py
│   │   ├── data_loader.py               # Load dữ liệu Webis-Clickbait-17
│   │   ├── text_cleaner.py              # Làm sạch văn bản social media
│   │   ├── data_splitter.py             # Chia dữ liệu train/val/test
│   │   └── label_converter.py           # Chuyển đổi truthMean -> binary
│   │
│   ├── 📁 fine_tuning/                  # Fine-tuning approach
│   │   ├── __init__.py
│   │   ├── models/                      # Các mô hình hỗ trợ
│   │   │   ├── __init__.py
│   │   │   ├── bert_model.py           # BERT/RoBERTa fine-tuning
│   │   │   ├── mistral_model.py        # Mistral-7B fine-tuning
│   │   │   └── llama_model.py          # LLaMA fine-tuning
│   │   ├── trainers/                   # Training utilities
│   │   │   ├── __init__.py
│   │   │   ├── base_trainer.py         # Base trainer class
│   │   │   ├── peft_trainer.py         # PEFT/LoRA trainer
│   │   │   └── unsloth_trainer.py      # Unsloth optimized trainer
│   │   ├── tokenizers/                 # Tokenization utilities
│   │   │   ├── __init__.py
│   │   │   └── advanced_tokenizer.py  # Specialized tokenizers
│   │   └── evaluation/                 # Evaluation utilities
│   │       ├── __init__.py
│   │       ├── metrics.py              # Metrics calculation
│   │       └── visualizations.py      # Result visualization
│   │
│   ├── 📁 prompting/                   # Prompting approach
│   │   ├── __init__.py
│   │   ├── prompt_templates/           # Prompt designs
│   │   │   ├── __init__.py
│   │   │   ├── zero_shot.py           # Zero-shot prompts
│   │   │   └── few_shot.py            # Few-shot prompts
│   │   ├── llm_clients/               # LLM API clients
│   │   │   ├── __init__.py
│   │   │   ├── openai_client.py       # GPT-4/GPT-4o
│   │   │   ├── anthropic_client.py    # Claude 3
│   │   │   └── huggingface_client.py  # Open-source LLMs
│   │   └── evaluation/                # Prompting evaluation
│   │       ├── __init__.py
│   │       ├── batch_evaluator.py     # Batch evaluation
│   │       └── cost_calculator.py     # API cost calculation
│   │
│   ├── 📁 utils/                      # Utilities chung
│   │   ├── __init__.py
│   │   ├── config.py                  # Configuration management
│   │   ├── logging_utils.py           # Logging utilities
│   │   ├── gpu_utils.py               # GPU management
│   │   └── file_utils.py              # File I/O utilities
│   │
│   └── 📁 comparison/                 # So sánh hai approach
│       ├── __init__.py
│       ├── benchmark.py               # Benchmark comparison
│       ├── cost_analysis.py           # Cost analysis
│       └── recommendation.py         # Final recommendations
│
├── 📁 notebooks/                      # Jupyter notebooks cho EDA
│   ├── 01_data_exploration.ipynb      # Khám phá dữ liệu
│   ├── 02_text_cleaning_analysis.ipynb # Phân tích text cleaning
│   ├── 03_fine_tuning_experiments.ipynb # Thí nghiệm fine-tuning
│   ├── 04_prompting_experiments.ipynb # Thí nghiệm prompting
│   └── 05_final_comparison.ipynb      # So sánh cuối cùng
│
├── 📁 configs/                        # Configuration files
│   ├── data_config.yaml              # Data processing configs
│   ├── model_config.yaml             # Model configurations
│   ├── training_config.yaml          # Training parameters
│   └── api_config.yaml               # API configurations
│
├── 📁 scripts/                        # Scripts chạy từng bước
│   ├── 01_setup_environment.sh        # Setup môi trường
│   ├── 02_download_data.py            # Download dữ liệu
│   ├── 03_process_data.py             # Xử lý dữ liệu
│   ├── 04_fine_tune_models.py         # Fine-tuning pipeline
│   ├── 05_run_prompting.py            # Prompting pipeline
│   └── 06_compare_results.py          # So sánh kết quả
│
├── 📁 experiments/                    # Kết quả thí nghiệm
│   ├── fine_tuning/                   # Kết quả fine-tuning
│   │   ├── roberta_base/              # RoBERTa results
│   │   ├── mistral_7b/                # Mistral results
│   │   └── llama_3_8b/                # LLaMA results
│   ├── prompting/                     # Kết quả prompting
│   │   ├── gpt4_zero_shot/            # GPT-4 zero-shot
│   │   ├── gpt4_few_shot/             # GPT-4 few-shot
│   │   ├── claude3_zero_shot/         # Claude 3 zero-shot
│   │   └── claude3_few_shot/          # Claude 3 few-shot
│   └── comparison/                    # So sánh tổng thể
│       ├── performance_metrics.json   # Metrics comparison
│       ├── cost_analysis.json         # Cost comparison
│       └── visualizations/            # Charts and plots
│
├── 📁 models/                         # Saved models
│   ├── fine_tuned/                    # Fine-tuned models
│   └── checkpoints/                   # Training checkpoints
│
├── 📁 outputs/                        # Output files
│   ├── predictions/                   # Model predictions
│   ├── reports/                       # Generated reports
│   └── logs/                          # Log files
│
├── 📁 tests/                          # Unit tests
│   ├── __init__.py
│   ├── test_data_processing.py        # Test data processing
│   ├── test_fine_tuning.py            # Test fine-tuning
│   └── test_prompting.py              # Test prompting
│
├── 📄 requirements.txt                # Python dependencies
├── 📄 environment.yml                 # Conda environment
├── 📄 setup.py                        # Package setup
├── 📄 README.md                       # Project overview
├── 📄 INSTALLATION.md                 # Installation guide
├── 📄 USAGE.md                        # Usage instructions
└── 📄 RESULTS.md                      # Final results summary
```

## 🎯 Workflow Tổng Thể

### Phase 1: Data Processing
1. Download và extract Webis-Clickbait-17
2. Phân tích và làm sạch dữ liệu
3. Chuyển đổi labels và chia train/val/test
4. Xử lý class imbalance

### Phase 2: Fine-tuning Approach
1. Thí nghiệm với RoBERTa-base (baseline)
2. Fine-tuning Mistral-7B với PEFT/LoRA
3. Fine-tuning LLaMA-3-8B với Unsloth
4. Evaluation và optimization

### Phase 3: Prompting Approach  
1. Thiết kế zero-shot prompts
2. Thiết kế few-shot prompts
3. Test với GPT-4/GPT-4o
4. Test với Claude 3
5. Test với open-source LLMs

### Phase 4: Comparison & Analysis
1. Performance comparison
2. Cost analysis
3. Speed benchmarking
4. Resource requirements
5. Final recommendations

## 🔧 Tech Stack

### Core Libraries
- **Data Processing**: pandas, numpy, datasets
- **Machine Learning**: scikit-learn, transformers
- **Deep Learning**: torch, accelerate
- **Optimization**: peft, unsloth, bitsandbytes
- **Evaluation**: seaborn, matplotlib, wandb
- **APIs**: openai, anthropic, huggingface_hub

### Infrastructure
- **GPU**: A5000 (để training)
- **Environment**: conda/venv
- **Monitoring**: wandb, tensorboard
- **Storage**: HuggingFace Hub, local storage 