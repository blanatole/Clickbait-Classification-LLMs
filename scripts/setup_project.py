#!/usr/bin/env python3
"""
Project Setup Script for Clickbait Detection
===========================================

This script sets up the complete project environment and prepares all
components for running the clickbait detection pipeline including:
- Environment validation
- Data preparation
- Model download preparation
- Configuration validation
- Directory structure creation
"""

import sys
import os
from pathlib import Path
import subprocess
import yaml
import json
from typing import Dict, List, Optional
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.logging_utils import setup_logger, setup_project_logging

logger = setup_logger(__name__)


class ProjectSetup:
    """Handles complete project setup and validation."""
    
    def __init__(self, project_root: Optional[str] = None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.config_dir = self.project_root / "configs"
        self.data_dir = self.project_root / "data"
        self.src_dir = self.project_root / "src"
        
    def check_python_version(self) -> bool:
        """Check if Python version meets requirements."""
        logger.info("Checking Python version...")
        
        if sys.version_info < (3, 8):
            logger.error("Python 3.8+ is required")
            return False
        
        logger.info(f"Python version: {sys.version}")
        return True
    
    def check_gpu_availability(self) -> Dict[str, any]:
        """Check GPU availability and CUDA setup."""
        logger.info("Checking GPU availability...")
        
        gpu_info = {
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_names": [],
            "cuda_version": None,
            "pytorch_cuda": False
        }
        
        try:
            import torch
            gpu_info["cuda_available"] = torch.cuda.is_available()
            gpu_info["gpu_count"] = torch.cuda.device_count()
            gpu_info["pytorch_cuda"] = torch.backends.cudnn.enabled
            
            if gpu_info["cuda_available"]:
                gpu_info["cuda_version"] = torch.version.cuda
                for i in range(gpu_info["gpu_count"]):
                    gpu_info["gpu_names"].append(torch.cuda.get_device_name(i))
            
            logger.info(f"GPU Info: {gpu_info}")
            
        except ImportError:
            logger.warning("PyTorch not installed yet - GPU check skipped")
        
        return gpu_info
    
    def create_directory_structure(self) -> bool:
        """Create all necessary project directories."""
        logger.info("Creating directory structure...")
        
        directories = [
            # Data directories
            "data/raw",
            "data/processed", 
            "data/splits",
            
            # Output directories
            "outputs/logs",
            "outputs/predictions",
            "outputs/reports",
            "outputs/cache",
            "outputs/fine_tuning",
            "outputs/prompting",
            "outputs/tensorboard",
            
            # Model directories
            "models/fine_tuned",
            "models/checkpoints",
            
            # Experiment directories
            "experiments/fine_tuning/roberta_base",
            "experiments/fine_tuning/mistral_7b", 
            "experiments/fine_tuning/llama_3_8b",
            "experiments/prompting/gpt4_zero_shot",
            "experiments/prompting/gpt4_few_shot",
            "experiments/prompting/claude3_zero_shot",
            "experiments/prompting/claude3_few_shot",
            "experiments/comparison",
            
            # Cache directories
            "cache/models",
            "cache/datasets",
            "cache/api_responses"
        ]
        
        for dir_path in directories:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {full_path}")
        
        logger.info(f"Created {len(directories)} directories")
        return True
    
    def validate_configs(self) -> bool:
        """Validate all configuration files."""
        logger.info("Validating configuration files...")
        
        config_files = [
            "data_config.yaml",
            "model_config.yaml", 
            "training_config.yaml",
            "api_config.yaml"
        ]
        
        all_valid = True
        
        for config_file in config_files:
            config_path = self.config_dir / config_file
            
            if not config_path.exists():
                logger.error(f"Missing config file: {config_file}")
                all_valid = False
                continue
            
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"✓ {config_file} is valid")
            except yaml.YAMLError as e:
                logger.error(f"Invalid YAML in {config_file}: {e}")
                all_valid = False
            except Exception as e:
                logger.error(f"Error reading {config_file}: {e}")
                all_valid = False
        
        return all_valid
    
    def check_data_availability(self) -> Dict[str, bool]:
        """Check if required data files are available."""
        logger.info("Checking data availability...")
        
        data_status = {
            "train_instances": False,
            "train_truth": False,
            "test_instances": False,
            "test_truth": False,
            "extracted": False
        }
        
        # Check for extracted data
        train_instances = self.data_dir / "clickbait17-validation-170630" / "instances.jsonl"
        train_truth = self.data_dir / "clickbait17-validation-170630" / "truth.jsonl"
        test_instances = self.data_dir / "clickbait17-test-170720" / "instances.jsonl"
        test_truth = self.data_dir / "clickbait17-test-170720" / "truth.jsonl"
        
        data_status["train_instances"] = train_instances.exists()
        data_status["train_truth"] = train_truth.exists()
        data_status["test_instances"] = test_instances.exists()
        data_status["test_truth"] = test_truth.exists()
        data_status["extracted"] = all([
            data_status["train_instances"],
            data_status["train_truth"],
            data_status["test_instances"],
            data_status["test_truth"]
        ])
        
        logger.info(f"Data status: {data_status}")
        return data_status
    
    def create_env_template(self) -> None:
        """Create .env template file for API keys."""
        logger.info("Creating .env template...")
        
        env_template = """# API Keys for Prompting Approach
# Copy this file to .env and fill in your actual API keys

# OpenAI API Key (for GPT-4, GPT-4o)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (for Claude 3)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# HuggingFace API Key (for open-source models)
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Weights & Biases API Key (optional, for experiment tracking)
WANDB_API_KEY=your_wandb_api_key_here

# HuggingFace Token (for accessing gated models like Llama)
HF_TOKEN=your_huggingface_token_here
"""
        
        env_template_path = self.project_root / ".env.template"
        with open(env_template_path, 'w') as f:
            f.write(env_template)
        
        logger.info(f"Created .env template at {env_template_path}")
        logger.info("Please copy .env.template to .env and fill in your API keys")
    
    def create_installation_script(self) -> None:
        """Create installation script for different environments."""
        logger.info("Creating installation scripts...")
        
        # Basic installation script
        install_script = """#!/bin/bash
# Installation script for Clickbait Detection Project

echo "Installing Clickbait Detection Project..."

# Create virtual environment
python -m venv clickbait_env

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    # Windows
    source clickbait_env/Scripts/activate
else
    # Linux/MacOS
    source clickbait_env/bin/activate
fi

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

echo "Installation completed!"
echo "Activate environment with:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    echo "  clickbait_env\\Scripts\\activate"
else
    echo "  source clickbait_env/bin/activate"
fi
"""
        
        install_path = self.project_root / "install.sh"
        with open(install_path, 'w') as f:
            f.write(install_script)
        
        # Make executable on Unix systems
        if os.name != 'nt':
            os.chmod(install_path, 0o755)
        
        # Windows batch script
        install_bat = """@echo off
REM Installation script for Windows

echo Installing Clickbait Detection Project...

REM Create virtual environment
python -m venv clickbait_env

REM Activate virtual environment
call clickbait_env\\Scripts\\activate.bat

REM Upgrade pip
pip install --upgrade pip

REM Install requirements
pip install -r requirements.txt

REM Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

echo Installation completed!
echo Activate environment with: clickbait_env\\Scripts\\activate.bat
pause
"""
        
        install_bat_path = self.project_root / "install.bat"
        with open(install_bat_path, 'w') as f:
            f.write(install_bat)
        
        logger.info("Created installation scripts: install.sh and install.bat")
    
    def create_run_scripts(self) -> None:
        """Create scripts to run different parts of the pipeline."""
        logger.info("Creating run scripts...")
        
        scripts = {
            "run_data_processing.py": """#!/usr/bin/env python3
\"\"\"Run data processing pipeline.\"\"\"
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_processing.data_loader import WebisClickbaitLoader
from data_processing.text_cleaner import SocialMediaTextCleaner
from utils.logging_utils import setup_project_logging

def main():
    setup_project_logging()
    
    # Load and process data
    loader = WebisClickbaitLoader()
    
    # Process training data
    train_features, train_labels = loader.load_dataset(
        "data/clickbait17-validation-170630/instances.jsonl",
        "data/clickbait17-validation-170630/truth.jsonl"
    )
    
    # Process test data
    test_features, test_labels = loader.load_dataset(
        "data/clickbait17-test-170720/instances.jsonl",
        "data/clickbait17-test-170720/truth.jsonl"
    )
    
    # Save processed data
    loader.save_processed_data(train_features, train_labels, "data/processed/train")
    loader.save_processed_data(test_features, test_labels, "data/processed/test")
    
    print("Data processing completed!")

if __name__ == "__main__":
    main()
""",
            
            "run_fine_tuning.py": """#!/usr/bin/env python3
\"\"\"Run fine-tuning pipeline.\"\"\"
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.logging_utils import setup_project_logging
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run fine-tuning")
    parser.add_argument("--model", choices=["roberta", "mistral", "llama"], 
                       default="roberta", help="Model to fine-tune")
    parser.add_argument("--use-peft", action="store_true", help="Use PEFT/LoRA")
    parser.add_argument("--use-unsloth", action="store_true", help="Use Unsloth optimization")
    
    args = parser.parse_args()
    
    setup_project_logging()
    
    print(f"Starting fine-tuning with {args.model}")
    print(f"PEFT enabled: {args.use_peft}")
    print(f"Unsloth enabled: {args.use_unsloth}")
    
    # TODO: Implement fine-tuning pipeline
    print("Fine-tuning pipeline will be implemented here")

if __name__ == "__main__":
    main()
""",
            
            "run_prompting.py": """#!/usr/bin/env python3
\"\"\"Run prompting pipeline.\"\"\"
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.logging_utils import setup_project_logging
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run prompting evaluation")
    parser.add_argument("--model", choices=["gpt4", "gpt4o", "claude3"], 
                       default="gpt4", help="Model to use")
    parser.add_argument("--strategy", choices=["zero_shot", "few_shot"], 
                       default="zero_shot", help="Prompting strategy")
    parser.add_argument("--sample-size", type=int, default=100, 
                       help="Number of samples to evaluate")
    
    args = parser.parse_args()
    
    setup_project_logging()
    
    print(f"Starting prompting evaluation with {args.model}")
    print(f"Strategy: {args.strategy}")
    print(f"Sample size: {args.sample_size}")
    
    # TODO: Implement prompting pipeline
    print("Prompting pipeline will be implemented here")

if __name__ == "__main__":
    main()
"""
        }
        
        scripts_dir = self.project_root / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        for script_name, script_content in scripts.items():
            script_path = scripts_dir / script_name
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make executable on Unix systems
            if os.name != 'nt':
                os.chmod(script_path, 0o755)
        
        logger.info(f"Created {len(scripts)} run scripts")
    
    def generate_project_summary(self) -> Dict:
        """Generate a comprehensive project summary."""
        logger.info("Generating project summary...")
        
        gpu_info = self.check_gpu_availability()
        data_status = self.check_data_availability()
        
        summary = {
            "project_name": "Clickbait Detection with Large Language Models",
            "python_version": sys.version,
            "project_root": str(self.project_root),
            "gpu_info": gpu_info,
            "data_status": data_status,
            "config_valid": self.validate_configs(),
            "approaches": {
                "fine_tuning": {
                    "models": ["RoBERTa-base", "Mistral-7B", "LLaMA-3-8B"],
                    "techniques": ["PEFT/LoRA", "Unsloth optimization"],
                    "ready": data_status["extracted"]
                },
                "prompting": {
                    "models": ["GPT-4", "GPT-4o", "Claude-3"],
                    "strategies": ["Zero-shot", "Few-shot"],
                    "ready": True  # Only needs API keys
                }
            },
            "next_steps": self._get_next_steps(gpu_info, data_status)
        }
        
        return summary
    
    def _get_next_steps(self, gpu_info: Dict, data_status: Dict) -> List[str]:
        """Generate recommended next steps."""
        steps = []
        
        if not data_status["extracted"]:
            steps.append("Extract the Webis-Clickbait-17 dataset (data already downloaded)")
        
        if not gpu_info["cuda_available"]:
            steps.append("Consider using Google Colab or rent GPU (A5000) for fine-tuning")
        
        steps.extend([
            "Copy .env.template to .env and add your API keys",
            "Run data processing: python scripts/run_data_processing.py",
            "Start with prompting approach for quick testing",
            "Run fine-tuning experiments on GPU-enabled environment"
        ])
        
        return steps
    
    def run_complete_setup(self) -> bool:
        """Run complete project setup."""
        logger.info("Starting complete project setup...")
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Create directory structure
        self.create_directory_structure()
        
        # Validate configs
        config_valid = self.validate_configs()
        
        # Create template files
        self.create_env_template()
        self.create_installation_script()
        self.create_run_scripts()
        
        # Generate and save summary
        summary = self.generate_project_summary()
        summary_path = self.project_root / "PROJECT_STATUS.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Project summary saved to {summary_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("PROJECT SETUP COMPLETE")
        print("="*60)
        print(f"Project: {summary['project_name']}")
        print(f"GPU Available: {summary['gpu_info']['cuda_available']}")
        print(f"Data Ready: {summary['data_status']['extracted']}")
        print(f"Configs Valid: {summary['config_valid']}")
        
        print("\nNext Steps:")
        for i, step in enumerate(summary['next_steps'], 1):
            print(f"{i}. {step}")
        
        print(f"\nProject status saved to: {summary_path}")
        print("="*60)
        
        return True


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Clickbait Detection Project")
    parser.add_argument("--project-root", help="Project root directory")
    parser.add_argument("--skip-gpu-check", action="store_true", 
                       help="Skip GPU availability check")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_project_logging()
    
    # Initialize setup
    setup = ProjectSetup(args.project_root)
    
    # Run setup
    success = setup.run_complete_setup()
    
    if success:
        logger.info("Project setup completed successfully!")
        sys.exit(0)
    else:
        logger.error("Project setup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 