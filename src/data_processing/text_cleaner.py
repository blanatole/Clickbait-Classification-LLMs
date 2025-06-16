"""
Social Media Text Cleaner for Clickbait Detection
================================================

Specialized text cleaning for social media content as outlined in README section 2.2.2:
- Convert to lowercase
- Remove URLs, user mentions (@username), special characters
- Process hashtags (keep text content)
- Handle social media specific patterns
- Normalize repetitive characters and whitespace
"""

import re
import string
import unicodedata
from typing import List, Dict, Optional, Union
import logging
import yaml
from pathlib import Path

from ..utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class SocialMediaTextCleaner:
    """
    Specialized text cleaner for social media content in clickbait detection.
    
    Implements the text cleaning strategy outlined in the README:
    - Lowercase conversion
    - URL removal
    - User mention removal (@username)
    - Special character handling
    - Hashtag processing (keep text content)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the text cleaner.
        
        Args:
            config_path: Path to text cleaning configuration
        """
        self.config = self._load_config(config_path)
        self.cleaning_config = self.config.get('text_cleaning', {})
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'urls_removed': 0,
            'mentions_removed': 0,
            'hashtags_processed': 0,
            'emails_removed': 0,
            'phone_numbers_removed': 0,
            'repetitive_chars_normalized': 0
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = "configs/data_config.yaml"
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default settings.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default configuration for text cleaning."""
        return {
            'text_cleaning': {
                'remove_urls': True,
                'remove_short_urls': True,
                'remove_mentions': True,
                'process_hashtags': True,
                'keep_hashtag_text': True,
                'remove_emails': True,
                'remove_phone_numbers': True,
                'emoji_strategy': 'remove',
                'to_lowercase': True,
                'normalize_repetitive_chars': True,
                'max_char_repeat': 2,
                'keep_punctuation': True,
                'remove_special_chars': True,
                'normalize_whitespace': True
            }
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient processing."""
        
        # URL patterns
        self.url_pattern = re.compile(
            r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
            re.IGNORECASE
        )
        
        # Short URL patterns (bit.ly, t.co, etc.)
        self.short_url_pattern = re.compile(
            r'(?:(?:http[s]?://)?(?:www\.)?(?:bit\.ly|t\.co|tinyurl\.com|goo\.gl|short\.link|ow\.ly)/\S+)',
            re.IGNORECASE
        )
        
        # User mention pattern
        self.mention_pattern = re.compile(r'@\w+', re.IGNORECASE)
        
        # Hashtag pattern
        self.hashtag_pattern = re.compile(r'#(\w+)', re.IGNORECASE)
        
        # Email pattern
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            re.IGNORECASE
        )
        
        # Phone number pattern (basic)
        self.phone_pattern = re.compile(
            r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            re.IGNORECASE
        )
        
        # Repetitive character pattern
        max_repeat = self.cleaning_config.get('max_char_repeat', 2)
        self.repetitive_pattern = re.compile(rf'(.)\1{{{max_repeat},}}')
        
        # Multiple whitespace pattern
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Special characters (keeping basic punctuation)
        if self.cleaning_config.get('keep_punctuation', True):
            # Keep basic punctuation: .,!?;:'"()-
            self.special_chars_pattern = re.compile(r'[^\w\s.,!?;:\'"()\-]')
        else:
            self.special_chars_pattern = re.compile(r'[^\w\s]')
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        if not self.cleaning_config.get('remove_urls', True):
            return text
            
        original_count = len(self.url_pattern.findall(text))
        text = self.url_pattern.sub(' ', text)
        
        if self.cleaning_config.get('remove_short_urls', True):
            short_url_count = len(self.short_url_pattern.findall(text))
            text = self.short_url_pattern.sub(' ', text)
            original_count += short_url_count
        
        self.stats['urls_removed'] += original_count
        return text
    
    def remove_mentions(self, text: str) -> str:
        """Remove user mentions (@username) from text."""
        if not self.cleaning_config.get('remove_mentions', True):
            return text
            
        mentions_count = len(self.mention_pattern.findall(text))
        text = self.mention_pattern.sub(' ', text)
        self.stats['mentions_removed'] += mentions_count
        return text
    
    def process_hashtags(self, text: str) -> str:
        """Process hashtags according to configuration."""
        if not self.cleaning_config.get('process_hashtags', True):
            return text
        
        hashtags = self.hashtag_pattern.findall(text)
        self.stats['hashtags_processed'] += len(hashtags)
        
        if self.cleaning_config.get('keep_hashtag_text', True):
            # Replace #hashtag with hashtag (keep the text content)
            text = self.hashtag_pattern.sub(r'\1', text)
        else:
            # Remove hashtags entirely
            text = self.hashtag_pattern.sub(' ', text)
        
        return text
    
    def remove_contact_info(self, text: str) -> str:
        """Remove email addresses and phone numbers."""
        # Remove emails
        if self.cleaning_config.get('remove_emails', True):
            emails_count = len(self.email_pattern.findall(text))
            text = self.email_pattern.sub(' ', text)
            self.stats['emails_removed'] += emails_count
        
        # Remove phone numbers
        if self.cleaning_config.get('remove_phone_numbers', True):
            phones_count = len(self.phone_pattern.findall(text))
            text = self.phone_pattern.sub(' ', text)
            self.stats['phone_numbers_removed'] += phones_count
        
        return text
    
    def handle_emojis(self, text: str) -> str:
        """Handle emojis according to strategy."""
        emoji_strategy = self.cleaning_config.get('emoji_strategy', 'remove')
        
        if emoji_strategy == 'remove':
            # Remove all emoji and emoticons
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                "\U00002700-\U000027BF"  # dingbats
                "\U0001F900-\U0001F9FF"  # supplemental symbols
                "]+", 
                flags=re.UNICODE
            )
            text = emoji_pattern.sub(' ', text)
            
        elif emoji_strategy == 'replace':
            # Replace with text descriptions (would need emoji library)
            # For now, just remove
            text = self.handle_emojis_remove(text)
            
        # 'keep' strategy - do nothing
        return text
    
    def normalize_repetitive_chars(self, text: str) -> str:
        """Normalize repetitive characters (e.g., 'sooooo' -> 'soo')."""
        if not self.cleaning_config.get('normalize_repetitive_chars', True):
            return text
        
        max_repeat = self.cleaning_config.get('max_char_repeat', 2)
        
        # Count normalization instances
        matches = self.repetitive_pattern.findall(text)
        self.stats['repetitive_chars_normalized'] += len(matches)
        
        # Replace repetitive characters
        replacement = '\\1' * max_repeat
        text = self.repetitive_pattern.sub(replacement, text)
        
        return text
    
    def normalize_case(self, text: str) -> str:
        """Convert text to lowercase."""
        if self.cleaning_config.get('to_lowercase', True):
            return text.lower()
        return text
    
    def remove_special_characters(self, text: str) -> str:
        """Remove special characters while optionally keeping punctuation."""
        if not self.cleaning_config.get('remove_special_chars', True):
            return text
        
        return self.special_chars_pattern.sub(' ', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace (multiple spaces, tabs, newlines to single space)."""
        if not self.cleaning_config.get('normalize_whitespace', True):
            return text
        
        # Replace multiple whitespace with single space
        text = self.whitespace_pattern.sub(' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def clean_text(self, text: str) -> str:
        """
        Apply complete text cleaning pipeline.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        if not text.strip():
            return ""
        
        # Track processing
        self.stats['total_processed'] += 1
        
        # Apply cleaning steps in order
        text = self.remove_urls(text)
        text = self.remove_mentions(text)  
        text = self.process_hashtags(text)
        text = self.remove_contact_info(text)
        text = self.handle_emojis(text)
        text = self.normalize_repetitive_chars(text)
        text = self.normalize_case(text)
        text = self.remove_special_characters(text)
        text = self.normalize_whitespace(text)
        
        return text
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean a batch of texts efficiently.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of cleaned texts
        """
        logger.info(f"Cleaning batch of {len(texts)} texts")
        
        cleaned_texts = []
        for text in texts:
            cleaned_text = self.clean_text(text)
            cleaned_texts.append(cleaned_text)
        
        logger.info("Batch cleaning completed")
        return cleaned_texts
    
    def get_statistics(self) -> Dict:
        """Return cleaning statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset cleaning statistics."""
        for key in self.stats:
            self.stats[key] = 0
    
    def analyze_text_quality(self, texts: List[str]) -> Dict:
        """
        Analyze quality metrics of input texts before cleaning.
        
        Args:
            texts: List of input texts
            
        Returns:
            Dictionary with quality metrics
        """
        logger.info("Analyzing text quality metrics")
        
        total_texts = len(texts)
        
        # Count various patterns
        url_count = sum(len(self.url_pattern.findall(text)) for text in texts)
        mention_count = sum(len(self.mention_pattern.findall(text)) for text in texts)
        hashtag_count = sum(len(self.hashtag_pattern.findall(text)) for text in texts)
        email_count = sum(len(self.email_pattern.findall(text)) for text in texts)
        
        # Length statistics
        lengths = [len(text) for text in texts if text]
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        
        # Character variety
        total_chars = ''.join(texts)
        unique_chars = len(set(total_chars))
        
        quality_metrics = {
            'total_texts': total_texts,
            'average_length': avg_length,
            'urls_per_text': url_count / total_texts if total_texts > 0 else 0,
            'mentions_per_text': mention_count / total_texts if total_texts > 0 else 0,
            'hashtags_per_text': hashtag_count / total_texts if total_texts > 0 else 0,
            'emails_per_text': email_count / total_texts if total_texts > 0 else 0,
            'unique_characters': unique_chars,
            'empty_texts': sum(1 for text in texts if not text.strip())
        }
        
        logger.info(f"Quality analysis completed: {quality_metrics}")
        return quality_metrics


def main():
    """Example usage of SocialMediaTextCleaner."""
    
    # Sample social media texts with various patterns
    sample_texts = [
        "OMG!!! You won't BELIEVE what happened next... 😱 #viral #clickbait http://bit.ly/fake-news @user123",
        "This simple trick will change your life FOREVER!!! Doctors HATE this method 🔥🔥🔥",
        "Breaking: Scientists discover new method... Check our website www.example.com for more info!!!",
        "URGENT: Limited time offer!!! Call 555-123-4567 or email info@scam.com #deal #urgent",
        "You'll never guess what celebrity X did today... The answer will SHOCK you! 😂😂😂"
    ]
    
    # Initialize cleaner
    cleaner = SocialMediaTextCleaner()
    
    # Analyze quality before cleaning
    print("=== Text Quality Analysis (Before Cleaning) ===")
    quality_metrics = cleaner.analyze_text_quality(sample_texts)
    for metric, value in quality_metrics.items():
        print(f"{metric}: {value}")
    
    print("\n=== Cleaning Examples ===")
    
    # Clean each text and show results
    for i, text in enumerate(sample_texts):
        cleaned = cleaner.clean_text(text)
        print(f"\nOriginal {i+1}: {text}")
        print(f"Cleaned {i+1}:  {cleaned}")
    
    # Show cleaning statistics
    print("\n=== Cleaning Statistics ===")
    stats = cleaner.get_statistics()
    for stat, count in stats.items():
        print(f"{stat}: {count}")


if __name__ == "__main__":
    main() 