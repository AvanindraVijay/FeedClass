#!/usr/bin/env python3
"""
Setup script for mBERT model
This script helps download and configure mBERT for the classification project
"""

import os
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

def download_mbert(model_dir, model_name="bert-base-multilingual-cased"):
    """Download mBERT model from HuggingFace"""
    print(f"🔄 Downloading mBERT model: {model_name}")
    print(f"📁 Target directory: {model_dir}")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Download tokenizer
        print("📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(model_dir)
        print("✅ Tokenizer downloaded successfully")