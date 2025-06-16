"""
Data Processing Module for Clickbait Detection
============================================

This module handles all data processing tasks for the Webis-Clickbait-17 dataset including:
- Data loading and extraction
- Text cleaning specialized for social media content
- Label conversion from continuous to binary
- Data splitting with stratification
- Class imbalance handling
"""

from .data_loader import WebisClickbaitLoader
from .text_cleaner import SocialMediaTextCleaner
from .label_converter import LabelConverter
from .data_splitter import StratifiedDataSplitter

__all__ = [
    "WebisClickbaitLoader",
    "SocialMediaTextCleaner", 
    "LabelConverter",
    "StratifiedDataSplitter"
] 