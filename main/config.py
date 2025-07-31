# config.py
import os

class Config:
    """Configuration settings for the Chainlit mBERT application"""
    
    # Model paths - Update these according to your setup
    MODEL_DIR = "/content/drive/MyDrive/newJKproject/finetune_model"  # Fine-tuned model
    BASE_MODEL_DIR = "/content/drive/MyDrive/newJKproject/model"      # Base mBERT model
    
    # Alternative paths for local development
    # MODEL_DIR = "./models/finetune_model"
    # BASE_MODEL_DIR = "./models/base_model"
    
    # Model settings
    MAX_LENGTH = 512
    TORCH_DTYPE = "float16"  # or "float32" for CPU
    
    # Inference settings
    BATCH_SIZE = 1
    USE_CUDA = True  # Set to False to force CPU usage
    
    # UI settings
    APP_TITLE = "mBERT Binary Classifier"
    WELCOME_MESSAGE = """
# 🤖 mBERT Binary Classifier

Welcome! This application uses your fine-tuned mBERT model to classify text into **Yes** or **No** responses.

## How to use:
1. Provide a **description** of your input
2. Add any **remarks** or additional context
3. The model will classify and return:
   - **Prediction**: Yes or No
   - **Confidence**: How confident the model is in its prediction

## Example formats:
```
Description: The weather is sunny today
Remarks: Perfect for outdoor activities
```

Or simply type your text naturally!
    """
    
    # Logging
    LOG_LEVEL = "INFO"
    
    @classmethod
    def validate_paths(cls):
        """Validate that model paths exist"""
        paths_to_check = [cls.MODEL_DIR]
        if cls.BASE_MODEL_DIR:
            paths_to_check.append(cls.BASE_MODEL_DIR)
        
        for path in paths_to_check:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model path not found: {path}")
        
        return True