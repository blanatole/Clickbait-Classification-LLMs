"""
Logging Utilities for Clickbait Detection Project
================================================

Centralized logging configuration with support for:
- File and console logging
- Different log levels for different modules
- Structured logging for experiments
- Integration with experiment tracking
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname_color = (
                self.COLORS[record.levelname] + 
                record.levelname + 
                self.COLORS['RESET']
            )
        else:
            record.levelname_color = record.levelname
            
        return super().format(record)


class StructuredLogger:
    """Structured logger for experiment tracking."""
    
    def __init__(self, name: str, log_file: Optional[str] = None):
        self.name = name
        self.logger = logging.getLogger(name)
        
        if log_file:
            self.log_file = Path(log_file)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_experiment_start(self, experiment_config: Dict[str, Any]):
        """Log experiment start with configuration."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'experiment_start',
            'experiment_name': self.name,
            'config': experiment_config
        }
        self.logger.info(f"EXPERIMENT_START: {json.dumps(log_entry)}")
    
    def log_training_step(self, step: int, metrics: Dict[str, float]):
        """Log training step with metrics."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'training_step',
            'step': step,
            'metrics': metrics
        }
        self.logger.info(f"TRAINING_STEP: {json.dumps(log_entry)}")
    
    def log_evaluation(self, split: str, metrics: Dict[str, float]):
        """Log evaluation results."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'evaluation',
            'split': split,
            'metrics': metrics
        }
        self.logger.info(f"EVALUATION: {json.dumps(log_entry)}")
    
    def log_experiment_end(self, final_metrics: Dict[str, float]):
        """Log experiment end with final metrics."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'experiment_end',
            'final_metrics': final_metrics
        }
        self.logger.info(f"EXPERIMENT_END: {json.dumps(log_entry)}")


def setup_logger(name: str, 
                level: str = "INFO",
                log_file: Optional[str] = None,
                console_output: bool = True,
                structured: bool = False) -> logging.Logger:
    """
    Set up a logger with the specified configuration.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console_output: Whether to output to console
        structured: Whether to use structured logging
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        if structured:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            console_formatter = ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname_color)s - %(message)s'
            )
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def configure_logging(config_dict: Optional[Dict] = None,
                     log_dir: str = "outputs/logs") -> None:
    """
    Configure logging for the entire project.
    
    Args:
        config_dict: Custom logging configuration
        log_dir: Directory for log files
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    if config_dict is None:
        config_dict = get_default_logging_config(log_dir)
    
    logging.config.dictConfig(config_dict)


def get_default_logging_config(log_dir: str) -> Dict:
    """Get default logging configuration."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            },
            'structured': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file_all': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'formatter': 'detailed',
                'filename': f'{log_dir}/all_{timestamp}.log',
                'encoding': 'utf-8'
            },
            'file_errors': {
                'level': 'ERROR',
                'class': 'logging.FileHandler',
                'formatter': 'detailed',
                'filename': f'{log_dir}/errors_{timestamp}.log',
                'encoding': 'utf-8'
            },
            'file_experiments': {
                'level': 'INFO',
                'class': 'logging.FileHandler',
                'formatter': 'structured',
                'filename': f'{log_dir}/experiments_{timestamp}.log',
                'encoding': 'utf-8'
            }
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['console', 'file_all'],
                'level': 'INFO',
                'propagate': False
            },
            'src.data_processing': {
                'handlers': ['console', 'file_all'],
                'level': 'INFO',
                'propagate': False
            },
            'src.fine_tuning': {
                'handlers': ['console', 'file_all'],
                'level': 'INFO',
                'propagate': False
            },
            'src.prompting': {
                'handlers': ['console', 'file_all'],
                'level': 'INFO',
                'propagate': False
            },
            'experiments': {
                'handlers': ['file_experiments', 'console'],
                'level': 'INFO',
                'propagate': False
            },
            'errors': {
                'handlers': ['file_errors', 'console'],
                'level': 'ERROR',
                'propagate': False
            }
        }
    }


class ExperimentLogger:
    """Specialized logger for experiment tracking."""
    
    def __init__(self, experiment_name: str, log_dir: str = "outputs/logs"):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment-specific log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"experiment_{experiment_name}_{timestamp}.log"
        
        self.logger = setup_logger(
            f"experiment.{experiment_name}",
            level="INFO",
            log_file=str(log_file),
            structured=True
        )
        
        self.start_time = datetime.now()
        self.metrics_history = []
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.logger.info(f"EXPERIMENT_CONFIG: {json.dumps(config, indent=2)}")
    
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information."""
        self.logger.info(f"DATASET_INFO: {json.dumps(dataset_info)}")
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model information."""
        self.logger.info(f"MODEL_INFO: {json.dumps(model_info)}")
    
    def log_training_progress(self, epoch: int, step: int, metrics: Dict[str, float]):
        """Log training progress."""
        log_entry = {
            'epoch': epoch,
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        self.metrics_history.append(log_entry)
        self.logger.info(f"TRAINING_PROGRESS: {json.dumps(log_entry)}")
    
    def log_validation_results(self, epoch: int, metrics: Dict[str, float]):
        """Log validation results."""
        log_entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'validation_metrics': metrics
        }
        self.logger.info(f"VALIDATION_RESULTS: {json.dumps(log_entry)}")
    
    def log_test_results(self, metrics: Dict[str, float]):
        """Log test results."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'test_metrics': metrics
        }
        self.logger.info(f"TEST_RESULTS: {json.dumps(log_entry)}")
    
    def log_experiment_summary(self):
        """Log experiment summary."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'total_training_steps': len(self.metrics_history)
        }
        
        self.logger.info(f"EXPERIMENT_SUMMARY: {json.dumps(summary)}")
    
    def save_metrics_history(self):
        """Save metrics history to JSON file."""
        metrics_file = self.log_dir / f"metrics_{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        self.logger.info(f"Metrics history saved to {metrics_file}")


def setup_project_logging(log_level: str = "INFO", 
                         log_dir: str = "outputs/logs") -> None:
    """
    Set up logging for the entire project.
    
    Args:
        log_level: Default log level
        log_dir: Directory for log files
    """
    # Ensure log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure project-wide logging
    configure_logging(log_dir=log_dir)
    
    # Set up specific loggers for different modules
    loggers = [
        "src.data_processing",
        "src.fine_tuning", 
        "src.prompting",
        "src.utils",
        "src.comparison"
    ]
    
    for logger_name in loggers:
        setup_logger(logger_name, level=log_level, console_output=True)
    
    # Log the setup completion
    logger = logging.getLogger("src.utils.logging_utils")
    logger.info(f"Project logging configured with level {log_level}")
    logger.info(f"Log files will be saved to {log_dir}")


def main():
    """Example usage of logging utilities."""
    
    # Set up project logging
    setup_project_logging()
    
    # Example usage
    logger = setup_logger("example", level="INFO", log_file="outputs/logs/example.log")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Example experiment logger
    exp_logger = ExperimentLogger("test_experiment")
    exp_logger.log_config({"model": "roberta-base", "lr": 2e-5})
    exp_logger.log_training_progress(1, 100, {"loss": 0.5, "accuracy": 0.8})
    exp_logger.log_validation_results(1, {"val_loss": 0.4, "val_accuracy": 0.85})
    exp_logger.log_experiment_summary()


if __name__ == "__main__":
    main() 