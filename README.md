# Clickbait Classification using LLM Fine-tuning

A complete pipeline for detecting clickbait headlines using fine-tuned BERT-family models and LoRA/QLoRA-adapted large language models. Optimized for high-end CUDA GPUs but configurable for various hardware setups.

## 📑 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [📊 Dataset](#-dataset)
- [🛠️ Installation & Setup](#️-installation--setup)
- [📁 Project Structure](#-project-structure)
- [🤖 Supported Models](#-supported-models)
- [🚀 Quick Start](#-quick-start)
- [📊 Performance Results](#-performance-results)
- [🔧 Technical Fixes](#-technical-fixes-implemented)
- [🐛 Troubleshooting](#-troubleshooting)
- [📈 Monitoring & Logging](#-monitoring--logging)
- [🔍 Model Configuration Details](#-model-configuration-details)
- [🎯 Future Improvements](#-future-improvements)
- [📋 Requirements](#-requirements)
- [🤝 Contributing](#-contributing)
- [🚀 Push to GitHub (Repo Owner)](#-push-to-github-repo-owner)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)
- [📞 Support](#-support)

## 🎯 Project Overview

This repository implements two complementary approaches to classifying Twitter headlines as clickbait or not-clickbait using the Webis-Clickbait-17 corpus:

### 🔬 Fine-tuning Module
Complete pipeline for fine-tuning models with your own data:
- **BERT family models**: Full fine-tuning of BERT-base, BERT-large, RoBERTa
- **Large Language Models**: Parameter-efficient fine-tuning with LoRA/QLoRA for Mistral, Llama models
- **Comprehensive evaluation**: Metrics, analysis, and model comparison tools

### 🎯 Prompting Module  
Zero-shot and few-shot approaches using pre-trained models:
- **Zero-shot prompting**: Direct classification without training
- **Few-shot prompting**: With example demonstrations
- **Advanced techniques**: Chain-of-thought, self-consistency, improved prompting strategies
- **Multiple model support**: OpenAI GPT, Claude, local models via API

### 🔧 Shared Resources
Common utilities and resources used by both approaches:
- **Data preprocessing**: Standardized data handling and analysis
- **Evaluation framework**: Unified metrics and benchmarking
- **Environment setup**: Automated dependency management
- **Comparison tools**: Side-by-side performance analysis

All scripts are optimized for high-end CUDA GPUs but configurable for smaller GPUs through CLI parameters.

## 📊 Dataset 

- **Source**: Webis-Clickbait-17 dataset
- **Total samples**: 38,517 Twitter headlines
- **Split**: 
  - Train: 30,812 samples
  - Validation: 3,851 samples  
  - Test: 3,854 samples
- **Labels**: Binary classification (0: non-clickbait, 1: clickbait)
- **Format**: JSONL files with `text` and `label` fields

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (high-end GPU recommended)
- 16GB+ VRAM for optimal performance

### Environment Setup

```bash
# Install conda if not already installed
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda   # -b = batch (no prompts)
eval "$($HOME/miniconda/bin/conda shell.bash hook)"  # add conda command to shell
conda init      # write to ~/.bashrc then open new shell or source ~/.bashrc

# Create conda environment
conda create -n clickbait python=3.10 -y
conda activate clickbait

# Deactivate virtual environment
deactivate

# Install PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install required packages
pip install -r requirements.txt

# Install PEFT for LoRA training
pip install peft bitsandbytes accelerate
```

### Hugging Face Authentication

For accessing gated models (Mistral, Llama):

```bash
# Login to Hugging Face
huggingface-cli login

# Or set environment variable
export HUGGINGFACE_HUB_TOKEN="your_token_here"
```

## 📁 Project Structure

```
clickbait-classification-LLM/
├── 🔬 fine-tuning/              # Fine-tuning module
│   ├── scripts/                 # Training scripts
│   │   ├── train_llm_lora.py    # Fine-tune LLMs with LoRA
│   │   ├── train_bert_family.py # Fine-tune BERT family models
│   │   └── evaluate_model.py    # Model evaluation
│   ├── outputs/                 # Training results (checkpoints, logs)
│   ├── models/                  # Fine-tuned models
│   └── README.md                # Fine-tuning guide
├── 🎯 prompting/                # Prompting module
│   ├── scripts/                 # Prompting scripts
│   │   ├── prompting_example.py     # Basic prompting examples
│   │   ├── prompting_unified.py     # Unified prompting approach
│   │   ├── improved_prompting.py    # Advanced prompting techniques
│   │   └── inference.py            # Inference with prompting
│   ├── outputs/                 # Prompting results
│   ├── results/                 # Evaluation results
│   └── README.md                # Prompting guide
├── 🔧 shared/                   # Shared resources
│   ├── utils/                   # Utility functions
│   │   ├── utils.py             # General utilities
│   │   ├── data_preprocessor.py # Data preprocessing
│   │   ├── data_analysis.py     # Data analysis tools
│   │   └── preprocess_clickbait_alpaca.py # Alpaca format preprocessing
│   ├── scripts/                 # Common scripts
│   │   ├── setup_environment.py     # Environment setup
│   │   ├── run_all_experiments.py   # Run all experiments
│   │   ├── benchmark_results.py     # Benchmark results
│   │   ├── run_evaluation.py        # Common evaluation
│   │   ├── test_api.py              # API testing
│   │   ├── test_api_unified.py      # Unified API testing
│   │   └── test_available_models.py # Model availability testing
│   └── data/                    # Datasets
│       ├── train/               # Training data
│       ├── val/                 # Validation data
│       └── test/                # Testing data
├── 📚 docs/                     # Documentation
│   └── PROMPTING_GUIDE.md       # Prompting techniques guide
├── requirements.txt             # Dependencies
├── README.md                    # This file
└── .gitignore                   # Git ignore rules
```

## 🤖 Supported Models

### BERT Family Models

| Model | Batch Size | Learning Rate | Epochs | Max Length | Training Time |
|-------|------------|---------------|--------|------------|---------------|
| BERT-base-uncased | 96 | 2e-5 | 4 | 128 | ~45 min |
| BERT-large-uncased | 32 | 1e-5 | 3 | 128 | ~1.5 hours |

### Large Language Models (LoRA)

| Model | Quantization | LoRA Rank | Batch Size | Training Time |
|-------|--------------|-----------|------------|---------------|
| Mistral-7B-Instruct-v0.3 | 4-bit | 8 | 20 | ~2 hours |
| Llama-3.1-8B-Instruct | 4-bit | 8 | 16 | ~3 hours |

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Setup environment and dependencies
python shared/scripts/setup_environment.py

# Run all experiments (both fine-tuning and prompting)
python shared/scripts/run_all_experiments.py
```

### 2. Fine-tuning Approach

```bash
# Train BERT family models
python fine-tuning/scripts/train_bert_family.py --model bert-base-uncased
python fine-tuning/scripts/train_bert_family.py --model bert-large-uncased

# Train LLM with LoRA (ensure Hugging Face authentication first)
huggingface-cli login
python fine-tuning/scripts/train_llm_lora.py --model mistral-7b-instruct
python fine-tuning/scripts/train_llm_lora.py --model llama3-8b

# Evaluate fine-tuned models
python fine-tuning/scripts/evaluate_model.py --model_path fine-tuning/models/bert-base-uncased
```

### 3. Prompting Approach

```bash
# Basic prompting examples
python prompting/scripts/prompting_example.py

# Unified prompting approach
python prompting/scripts/prompting_unified.py

# Advanced prompting techniques
python prompting/scripts/improved_prompting.py

# Inference with prompting
python prompting/scripts/inference.py --text "You won't believe what happened next!"
```

### 4. Benchmark and Compare Results

```bash
# Compare results between fine-tuning and prompting
python shared/scripts/benchmark_results.py

# Run comprehensive evaluation
python shared/scripts/run_evaluation.py
```

## 📊 Performance Results

### BERT Models

- **BERT-base-uncased**: 
  - Accuracy: 83.2%
  - F1-score: 85.1%
  - Training time: 45 minutes
  
- **BERT-large-uncased**:
  - Accuracy: 85.7%
  - F1-score: 87.3%
  - Training time: 1.5 hours

### LLM Models (LoRA)

- **Mistral-7B-Instruct-v0.3**:
  - Accuracy: 87.9%
  - F1-score: 89.2%
  - Training time: 2 hours
  - Parameters trained: ~0.5% of total

- **Llama-3.1-8B-Instruct**:
  - Accuracy: 88.5%
  - F1-score: 90.1%
  - Training time: 3 hours
  - Parameters trained: ~0.5% of total

## 🔧 Technical Fixes Implemented

### 1. PyTorch Security Issue
- **Problem**: DeBERTa model blocked due to PyTorch vulnerability (CVE-2025-32434)
- **Solution**: 
  - Updated PyTorch to 2.5.1+
  - Removed DeBERTa from training pipeline
  - Focus on stable BERT models

### 2. Transformers API Compatibility
- **Problem**: `evaluation_strategy` parameter deprecated
- **Solution**: Updated to `eval_strategy` for newer transformers versions

### 3. Padding Token Issues
- **Problem**: LLM models missing padding tokens causing batch processing errors
- **Solution**:
  ```python
  if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token
      tokenizer.pad_token_id = tokenizer.eos_token_id
  tokenizer.padding_side = "right"
  ```

### 4. Memory Optimization
- **Techniques used**:
  - Gradient checkpointing
  - FP16/BF16 mixed precision
  - Gradient accumulation
  - 4-bit/8-bit quantization for LLMs

## 🐛 Troubleshooting

### Common Issues

#### GPU Memory Errors
```bash
# Reduce batch size in model configs
# Enable gradient checkpointing
# Use mixed precision training
```

#### Hugging Face Authentication
```bash
# Check login status
huggingface-cli whoami

# Re-login if needed
huggingface-cli logout
huggingface-cli login
```

#### Import Errors
```bash
# Install missing dependencies
pip install transformers datasets torch
pip install accelerate bitsandbytes peft
pip install scikit-learn pandas numpy
```

#### Data Loading Issues
```bash
# Verify data files exist
ls -la data/train/data.jsonl
ls -la data/val/data.jsonl
ls -la data/test/data.jsonl
```

## 📈 Monitoring & Logging

### Training Monitoring
- **Weights & Biases**: Automatic logging of metrics
- **Tensorboard**: Local training visualization
- **Console output**: Real-time training progress

### Log Locations
- Training logs: `outputs/{model_name}/runs/`
- Model checkpoints: `outputs/{model_name}/checkpoint-*/`
- Results: `outputs/{model_name}/results.json`

## 🔍 Model Configuration Details

### BERT Training Arguments
```python
TrainingArguments(
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=48,
    num_train_epochs=4,
    weight_decay=0.01,
    warmup_steps=500,
    fp16=True,
    gradient_checkpointing=True,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)
```

### LoRA Configuration
```python
LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none"
)
```

## 🎯 Future Improvements

### Planned Features

1. **Prompting Approaches**
   - Zero-shot classification with GPT-4
   - Few-shot learning with Claude
   - Chain-of-thought prompting

2. **Ensemble Methods**
   - Model averaging
   - Voting classifiers
   - Stacking approaches

3. **Data Augmentation**
   - Paraphrasing with T5
   - Back-translation
   - Synthetic data generation

4. **Advanced Techniques**
   - Adversarial training
   - Knowledge distillation
   - Multi-task learning

## 📋 Requirements

### Python Dependencies
See `requirements.txt` for complete list.

### Hardware Requirements
- **Minimum**: 16GB VRAM GPU
- **Recommended**: High-end GPU with 16GB+ VRAM
- **RAM**: 32GB+ system memory
- **Storage**: 50GB+ free space

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🚀 Push to GitHub (Repo Owner)

Direct push method for repository owners:

### 1. Check for existing SSH keys
```bash
ls ~/.ssh/id_ed25519.pub  # if no file exists, create new one below
```

### 2. Generate SSH key pair (ED25519 - strong and short)
```bash
ssh-keygen -t ed25519 -C "your-email@example.com"    # press Enter 3 times for defaults
# Creates ~/.ssh/id_ed25519 & id_ed25519.pub
```

### 3. Add key to ssh-agent (helps git not ask for passphrase every time)
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

### 4. Copy public key
```bash
cat ~/.ssh/id_ed25519.pub
```

### 5. Add to GitHub
- Go to Settings → Deploy keys → Add deploy key
- Enter title and paste key into text area → Add key

### 6. Test connection in project terminal
```bash
ssh -T git@github.com
# First time will ask "Are you sure you want to continue connecting?" → type yes
# Should see message: "Hi <username>! You've successfully authenticated..."
```

### 7. Point remote to SSH instead of HTTPS
```bash
# Check current remote (should be https)
git remote -v

# Change to SSH
git remote set-url origin git@github.com:<username>/<repo>.git
```

### 8. Now you can push without password prompts
```bash
git add .
git commit -m "Fix BERT training"
git push origin main      # won't ask for user/password anymore
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Webis-Clickbait-17 dataset creators
- Hugging Face for transformers library
- Microsoft for PEFT library
- The open-source ML community

## 📞 Support

For questions and support:
- Create an issue in this repository
- Check the troubleshooting section
- Review the documentation in `docs/`

---

**Note**: This project is optimized for high-end CUDA GPUs. Adjust batch sizes and configurations for different hardware setups.