# Hướng Dẫn Triển Khai: Khai thác LLM để Phát hiện Clickbait

## 📋 Tổng Quan Dự Án

Dự án này thực hiện việc **khai thác Mô hình Ngôn ngữ Lớn (LLM) để phát hiện clickbait** trên tập dữ liệu **Webis-Clickbait-17** thông qua hai phương pháp chính:

1. **Fine-tuning**: Tinh chỉnh mô hình ngôn ngữ với PEFT/LoRA và Unsloth
2. **Prompting**: Sử dụng các LLM nền tảng thông qua gợi ý (zero-shot và few-shot)

## 🏗️ Kiến Trúc Dự Án

```
clickbait-classification/
├── 📁 src/                           # Source code chính
│   ├── data_processing/              # Xử lý dữ liệu
│   ├── fine_tuning/                  # Fine-tuning approach
│   ├── prompting/                    # Prompting approach
│   ├── utils/                        # Utilities chung
│   └── comparison/                   # So sánh kết quả
├── 📁 configs/                       # Configuration files
├── 📁 scripts/                       # Scripts chạy pipeline
├── 📁 notebooks/                     # Jupyter notebooks
├── 📁 data/                          # Dữ liệu
├── 📁 models/                        # Models đã fine-tuned
├── 📁 experiments/                   # Kết quả thí nghiệm
└── 📁 outputs/                       # Outputs và logs
```

## 🚀 Hướng Dẫn Cài Đặt

### Bước 1: Clone và Setup Môi Trường

```bash
# Clone repository (nếu có)
git clone <repository-url>
cd clickbait-classification

# Chạy script setup
python scripts/setup_project.py
```

### Bước 2: Cài Đặt Dependencies

#### Trên Windows:
```cmd
install.bat
```

#### Trên Linux/MacOS:
```bash
./install.sh
```

#### Cài đặt thủ công:
```bash
# Tạo virtual environment
python -m venv clickbait_env

# Kích hoạt environment
# Windows:
clickbait_env\Scripts\activate
# Linux/MacOS:
source clickbait_env/bin/activate

# Cài đặt packages
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Bước 3: Cấu Hình API Keys

```bash
# Copy template và cấu hình API keys
cp .env.template .env

# Chỉnh sửa .env với API keys thực:
# OPENAI_API_KEY=your_actual_key
# ANTHROPIC_API_KEY=your_actual_key
# HUGGINGFACE_API_KEY=your_actual_key
```

## 📊 Pipeline Xử Lý Dữ Liệu

### Bước 1: Xử Lý Dữ Liệu

```bash
# Chạy pipeline xử lý dữ liệu hoàn chỉnh
python scripts/run_data_processing.py
```

**Quá trình bao gồm:**
- ✅ Load dữ liệu từ JSONL files
- ✅ Trích xuất text features, loại bỏ media components  
- ✅ Làm sạch văn bản chuyên biệt cho social media:
  - Chuyển thành chữ thường
  - Loại bỏ URLs, mentions (@username)
  - Xử lý hashtags (giữ lại text content)
  - Chuẩn hóa ký tự lặp và whitespace
- ✅ Chuyển đổi labels từ truthMean → binary (>= 0.5 = clickbait)
- ✅ Chia dữ liệu train/val/test với stratification
- ✅ Xử lý class imbalance (~1:3 ratio)

## 🔧 Approach 1: Fine-tuning

### Mô Hình Được Hỗ Trợ

1. **RoBERTa-base** (baseline - đã chứng minh hiệu quả trên Webis-Clickbait-17)
2. **Mistral-7B** (modern approach với PEFT/LoRA)  
3. **LLaMA-3-8B** (với Unsloth optimization)

### Chạy Fine-tuning

```bash
# RoBERTa baseline
python scripts/run_fine_tuning.py --model roberta

# Mistral với PEFT/LoRA
python scripts/run_fine_tuning.py --model mistral --use-peft

# LLaMA với Unsloth
python scripts/run_fine_tuning.py --model llama --use-peft --use-unsloth
```

### Kỹ Thuật Optimization

- **PEFT/LoRA**: Chỉ fine-tune <1% parameters
- **Unsloth**: Tăng tốc 2x, giảm 60% memory cho Mistral/LLaMA
- **Quantization**: 4-bit loading với bitsandbytes
- **Gradient Checkpointing**: Tiết kiệm memory

## 🎯 Approach 2: Prompting

### Zero-shot Prompting

```bash
# GPT-4 zero-shot
python scripts/run_prompting.py --model gpt4 --strategy zero_shot --sample-size 1000

# Claude-3 zero-shot  
python scripts/run_prompting.py --model claude3 --strategy zero_shot --sample-size 1000
```

### Few-shot Prompting

```bash
# GPT-4 few-shot với 5 examples
python scripts/run_prompting.py --model gpt4 --strategy few_shot --sample-size 1000

# GPT-4o few-shot
python scripts/run_prompting.py --model gpt4o --strategy few_shot --sample-size 1000
```

### Mô Hình Được Hỗ Trợ

- **OpenAI**: GPT-4, GPT-4-turbo, GPT-4o
- **Anthropic**: Claude-3-Sonnet, Claude-3-Opus
- **Open-source**: Mixtral, LLaMA via HuggingFace API

## 📈 Đánh Giá và So Sánh

### Metrics Theo Dõi

- **Accuracy**, **Precision**, **Recall**, **F1-score**
- **Confusion Matrix** cho phân tích lỗi chi tiết
- **Cost Analysis** cho prompting approach
- **Speed Benchmarking** cho cả hai approaches

### Benchmark Targets

| Method | Accuracy | F1-score | Source |
|--------|----------|----------|---------|
| Clickbait Challenge 2017 Winner | 0.876 | 0.741 | Official |
| RoBERTa (Prior Work) | 0.8575 | 0.6901 | Research |
| **Target for This Project** | **> 0.85** | **> 0.70** | Goal |

### Chạy So Sánh Toàn Diện

```bash
# So sánh tất cả approaches
python scripts/run_comparison.py

# Tạo báo cáo cuối cùng
python scripts/generate_final_report.py
```

## 💰 Chiến Lược Triển Khai (theo README)

### Giai đoạn 1: Prototyping
- Sử dụng **Prompting** với GPT-4o
- Mục đích: Proof of Concept nhanh chóng
- Chi phí thấp, validation ý tưởng

### Giai đoạn 2: Production
- Chuyển sang **Fine-tuning** 
- Stack đề xuất: **Mistral-7B + PEFT/LoRA + Unsloth**
- Mục đích: Hiệu quả chi phí, tốc độ suy luận nhanh, quyền riêng tư

## 🖥️ Yêu Cầu Phần Cứng

### Cho Fine-tuning (Khuyến nghị)

- **GPU**: NVIDIA A5000 (24GB VRAM) - để thuê khi ready
- **RAM**: 32GB+
- **Storage**: 100GB+ free space

### Cho Development (Hiện tại)

- **CPU**: Xử lý dữ liệu và prompting
- **RAM**: 16GB+
- **Storage**: 50GB+

### Cloud Alternatives

- **Google Colab Pro/Pro+** 
- **Paperspace Gradient**
- **AWS/GCP GPU instances**

## 📝 Configuration Management

### Data Configuration (`configs/data_config.yaml`)
- Text fields selection
- Cleaning parameters 
- Label conversion settings
- Class imbalance handling

### Model Configuration (`configs/model_config.yaml`)
- Model selections
- Quantization settings
- Memory optimization
- Tokenization parameters

### Training Configuration (`configs/training_config.yaml`)
- Hyperparameters
- PEFT/LoRA settings
- Hardware-specific configs
- Logging and monitoring

### API Configuration (`configs/api_config.yaml`)
- Provider settings
- Prompt templates
- Rate limiting
- Cost tracking

## 🔍 Monitoring và Logging

### Experiment Tracking
- **Structured logging** cho tất cả experiments
- **Metrics history** được lưu JSON format
- **TensorBoard** integration cho fine-tuning
- **Weights & Biases** support (optional)

### Log Files
```
outputs/logs/
├── all_YYYYMMDD_HHMMSS.log          # Tất cả logs
├── errors_YYYYMMDD_HHMMSS.log       # Chỉ errors
├── experiments_YYYYMMDD_HHMMSS.log  # Experiment tracking
└── experiment_[name]_YYYYMMDD.log   # Per-experiment logs
```

## 🚦 Quy Trình Development

### 1. Setup và Validation
```bash
python scripts/setup_project.py
# Kiểm tra PROJECT_STATUS.json
```

### 2. Data Processing
```bash
python scripts/run_data_processing.py
```

### 3. Quick Testing với Prompting
```bash
# Test nhỏ với 100 samples
python scripts/run_prompting.py --sample-size 100
```

### 4. Fine-tuning (trên GPU)
```bash
# Bắt đầu với RoBERTa
python scripts/run_fine_tuning.py --model roberta

# Thử nghiệm advanced
python scripts/run_fine_tuning.py --model mistral --use-peft
```

### 5. Comparison và Analysis
```bash
python scripts/run_comparison.py
python scripts/generate_final_report.py
```

## 🔧 Troubleshooting

### Common Issues

1. **NLTK punkt_tab error**:
```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

2. **GPU memory issues**:
- Giảm batch size trong config
- Enable gradient checkpointing
- Sử dụng 4-bit quantization

3. **API rate limits**:
- Điều chỉnh batch_size trong api_config.yaml
- Increase retry_delay

4. **Import errors**:
- Đảm bảo virtual environment activated
- Check sys.path trong scripts

## 📚 Code Structure Details

### Key Components

#### Data Processing
- `WebisClickbaitLoader`: Load và extract text features
- `SocialMediaTextCleaner`: Làm sạch văn bản chuyên biệt
- `LabelConverter`: Chuyển đổi truthMean → binary
- `StratifiedDataSplitter`: Chia dữ liệu có stratification

#### Fine-tuning
- `ModelManager`: Quản lý models và tokenizers
- `PEFTTrainer`: PEFT/LoRA training
- `UnslothTrainer`: Unsloth optimization
- `MetricsCalculator`: Tính toán và tracking metrics

#### Prompting
- `PromptTemplateManager`: Quản lý prompt templates
- `LLMClients`: API clients cho các providers
- `BatchEvaluator`: Evaluation batch processing
- `CostCalculator`: Tracking API costs

#### Utilities
- `LoggingUtils`: Centralized logging
- `ConfigManager`: Configuration management
- `GPUManager`: GPU utilities
- `FileManager`: File I/O helpers

## 🎯 Expected Outcomes

### Performance Targets
- **Accuracy**: > 85% (so với 87.6% của winner 2017)
- **F1-score**: > 70% (so với 74.1% của winner 2017)
- **Speed**: < 100ms inference time cho production

### Deliverables
1. **Trained Models**: RoBERTa, Mistral, LLaMA fine-tuned
2. **Evaluation Results**: Comprehensive comparison
3. **Cost Analysis**: Detailed breakdown cho cả approaches
4. **Recommendations**: Strategic recommendations
5. **Deployment Ready**: Production-ready pipeline

## 📖 Next Steps

1. ✅ **Hoàn tất setup**: Chạy `python scripts/setup_project.py`
2. ⏳ **Data processing**: Xử lý và clean dữ liệu  
3. ⏳ **Prompting experiments**: Quick validation
4. ⏳ **Thuê GPU A5000**: Cho fine-tuning phase
5. ⏳ **Fine-tuning experiments**: Comprehensive training
6. ⏳ **Comparison và analysis**: Final evaluation
7. ⏳ **Report generation**: Documentation và recommendations

---

**📧 Contact**: Để hỗ trợ hoặc câu hỏi về implementation, vui lòng tham khảo logs trong `outputs/logs/` hoặc check `PROJECT_STATUS.json` để biết trạng thái hiện tại của dự án. 