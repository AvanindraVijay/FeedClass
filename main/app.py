import chainlit as cl
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle
import os
import logging
from typing import List, Tuple, Dict, Any
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentenceTransformerClassifier:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Sentence Transformer Classifier
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.sentence_model = None
        self.classifier = None
        self.label_encoder = {"no": 0, "yes": 1}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_trained = False
        
        # Example training data (replace with your actual data)
        self.training_data = [
            ("The weather is sunny and beautiful", "Perfect for outdoor activities", "yes"),
            ("It's raining heavily outside", "Not good for picnic", "no"),
            ("The system is working perfectly", "All tests passed", "yes"),
            ("Error occurred in the application", "Multiple failures detected", "no"),
            ("Task completed successfully", "Within the deadline", "yes"),
            ("Project delayed significantly", "Missing key requirements", "no"),
            ("Great performance today", "Exceeded expectations", "yes"),
            ("Poor results obtained", "Below average performance", "no"),
            ("Excellent customer feedback", "Very satisfied with service", "yes"),
            ("Negative customer reviews", "Complaints about quality", "no"),
        ]
        
    def load_sentence_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.sentence_model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("✅ Sentence transformer model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading sentence transformer model: {e}")
            return False
    
    def prepare_training_data(self, data: List[Tuple[str, str, str]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data by creating embeddings
        
        Args:
            data: List of (description, remarks, label) tuples
            
        Returns:
            Tuple of (embeddings, labels)
        """
        texts = []
        labels = []
        
        for description, remarks, label in data:
            # Combine description and remarks
            combined_text = f"Description: {description} Remarks: {remarks}"
            texts.append(combined_text)
            labels.append(self.label_encoder[label])
        
        # Generate embeddings
        embeddings = self.sentence_model.encode(texts)
        
        return np.array(embeddings), np.array(labels)
    
    def train_classifier(self, classifier_type: str = "logistic"):
        """
        Train the classifier on the training data
        
        Args:
            classifier_type: Type of classifier ('logistic', 'random_forest', 'svm')
        """
        try:
            if self.sentence_model is None:
                raise ValueError("Sentence transformer model not loaded")
            
            logger.info("Preparing training data...")
            X_train, y_train = self.prepare_training_data(self.training_data)
            
            logger.info(f"Training {classifier_type} classifier...")
            
            if classifier_type == "logistic":
                self.classifier = LogisticRegression(random_state=42)
            elif classifier_type == "random_forest":
                self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            elif classifier_type == "svm":
                self.classifier = SVC(probability=True, random_state=42)
            else:
                raise ValueError(f"Unknown classifier type: {classifier_type}")
            
            self.classifier.fit(X_train, y_train)
            self.is_trained = True
            
            logger.info("✅ Classifier trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training classifier: {e}")
            return False
    
    def predict(self, description: str, remarks: str) -> Tuple[str, float, Dict[str, Any]]:
        """
        Make prediction based on description and remarks
        
        Args:
            description: Input description
            remarks: Input remarks
            
        Returns:
            Tuple of (prediction, confidence, additional_info)
        """
        try:
            if not self.is_trained:
                raise ValueError("Classifier not trained. Call train_classifier() first.")
            
            # Combine description and remarks
            combined_text = f"Description: {description} Remarks: {remarks}"
            
            # Generate embedding
            embedding = self.sentence_model.encode([combined_text])
            
            # Make prediction
            prediction_prob = self.classifier.predict_proba(embedding)[0]
            predicted_class = np.argmax(prediction_prob)
            confidence = float(prediction_prob[predicted_class])
            
            # Convert to label
            predicted_label = "yes" if predicted_class == 1 else "no"
            
            # Additional information
            additional_info = {
                "probabilities": {
                    "no": float(prediction_prob[0]),
                    "yes": float(prediction_prob[1])
                },
                "embedding_dim": len(embedding[0]),
                "text_length": len(combined_text),
                "timestamp": datetime.now().isoformat()
            }
            
            return predicted_label, confidence, additional_info
            
        except Exception as e:
            logger.error(f"❌ Error during prediction: {e}")
            return "error", 0.0, {}
    
    def find_similar_examples(self, description: str, remarks: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Find similar examples from training data
        
        Args:
            description: Input description
            remarks: Input remarks
            top_k: Number of top similar examples to return
            
        Returns:
            List of (example_text, similarity_score) tuples
        """
        try:
            if self.sentence_model is None:
                raise ValueError("Sentence transformer model not loaded")
            
            # Current input
            current_text = f"Description: {description} Remarks: {remarks}"
            current_embedding = self.sentence_model.encode([current_text])
            
            # Training examples
            training_texts = []
            for desc, rem, label in self.training_data:
                training_texts.append(f"Description: {desc} Remarks: {rem} (Label: {label})")
            
            training_embeddings = self.sentence_model.encode(training_texts)
            
            # Calculate similarities
            similarities = cosine_similarity(current_embedding, training_embeddings)[0]
            
            # Get top k similar examples
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            similar_examples = [(training_texts[i], float(similarities[i])) for i in top_indices]
            
            return similar_examples
            
        except Exception as e:
            logger.error(f"❌ Error finding similar examples: {e}")
            return []
    
    def save_model(self, filepath: str):
        """Save the trained classifier"""
        try:
            model_data = {
                "classifier": self.classifier,
                "label_encoder": self.label_encoder,
                "model_name": self.model_name,
                "is_trained": self.is_trained,
                "training_data": self.training_data
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"✅ Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str):
        """Load a trained classifier"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.classifier = model_data["classifier"]
            self.label_encoder = model_data["label_encoder"]
            self.model_name = model_data["model_name"]
            self.is_trained = model_data["is_trained"]
            self.training_data = model_data.get("training_data", [])
            
            logger.info(f"✅ Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False

# Global classifier instance
classifier = None

@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    global classifier
    
    # Configuration
    MODEL_NAME = "all-MiniLM-L6-v2"  # You can change this to other models
    CLASSIFIER_TYPE = "logistic"  # Options: logistic, random_forest, svm
    SAVED_MODEL_PATH = "./sentence_transformer_model.pkl"
    
    # Show loading message
    loading_msg = cl.Message(content="🔄 Loading Sentence Transformer classification model...")
    await loading_msg.send()
    
    try:
        # Initialize classifier
        classifier = SentenceTransformerClassifier(MODEL_NAME)
        
        # Load sentence transformer model
        if not classifier.load_sentence_model():
            await loading_msg.remove()
            await cl.Message(content="❌ Failed to load sentence transformer model.").send()
            return
        
        # Try to load existing trained model
        if os.path.exists(SAVED_MODEL_PATH):
            await loading_msg.remove()
            loading_msg = cl.Message(content="🔄 Loading existing trained model...")
            await loading_msg.send()
            
            if classifier.load_model(SAVED_MODEL_PATH):
                await loading_msg.remove()
                await cl.Message(content="✅ Pre-trained model loaded successfully!").send()
            else:
                await loading_msg.remove()
                await cl.Message(content="⚠️ Failed to load existing model. Training new model...").send()
                
                # Train new model
                if classifier.train_classifier(CLASSIFIER_TYPE):
                    # Save the trained model
                    classifier.save_model(SAVED_MODEL_PATH)
                    await cl.Message(content="✅ New model trained and saved successfully!").send()
                else:
                    await cl.Message(content="❌ Failed to train classifier.").send()
                    return
        else:
            await loading_msg.remove()
            training_msg = cl.Message(content="🔄 Training new classifier...")
            await training_msg.send()
            
            # Train new model
            if classifier.train_classifier(CLASSIFIER_TYPE):
                # Save the trained model
                classifier.save_model(SAVED_MODEL_PATH)
                await training_msg.remove()
                await cl.Message(content="✅ Model trained and saved successfully!").send()
            else:
                await training_msg.remove()
                await cl.Message(content="❌ Failed to train classifier.").send()
                return
        
        # Send welcome message
        welcome_msg = f"""
# 🤖 Sentence Transformer Binary Classifier

Welcome! This application uses **{MODEL_NAME}** sentence transformer with **{CLASSIFIER_TYPE}** classifier to classify text into **Yes** or **No** responses.

## Features:
- **Semantic Understanding**: Uses sentence embeddings for better text understanding
- **Similarity Search**: Find similar examples from training data
- **Confidence Scoring**: Get prediction confidence scores
- **Multiple Classifiers**: Support for Logistic Regression, Random Forest, and SVM

## How to use:
1. Provide a **description** of your input
2. Add any **remarks** or additional context
3. The model will classify and return:
   - **Prediction**: Yes or No
   - **Confidence**: How confident the model is
   - **Similar Examples**: Related examples from training data

## Commands:
- **classify**: Normal classification
- **similar**: Find similar examples only
- **retrain**: Retrain the model with new data
- **info**: Show model information

## Example:
```
Description: The project is completed on time
Remarks: All requirements met successfully
```

Ready for classification! 🚀
        """
        
        await cl.Message(content=welcome_msg).send()
        
    except Exception as e:
        await loading_msg.remove()
        await cl.Message(content=f"❌ Error initializing model: {str(e)}").send()
        logger.error(f"Initialization error: {e}")

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    global classifier
    
    if classifier is None:
        await cl.Message(content="❌ Model not loaded. Please restart the application.").send()
        return
    
    user_input = message.content.strip()
    
    # Handle special commands
    if user_input.lower() == "info":
        await handle_info_command()
        return
    elif user_input.lower() == "retrain":
        await handle_retrain_command()
        return
    elif user_input.lower().startswith("similar"):
        await handle_similar_command(user_input)
        return
    
    # Show processing message
    processing_msg = cl.Message(content="🔄 Processing your input...")
    await processing_msg.send()
    
    try:
        # Parse input
        description, remarks = parse_input(user_input)
        
        if not description and not remarks:
            await processing_msg.remove()
            await cl.Message(content="⚠️ Please provide at least a description or remarks.").send()
            return
        
        # Make prediction
        prediction, confidence, additional_info = classifier.predict(description, remarks)
        
        if prediction == "error":
            await processing_msg.remove()
            await cl.Message(content="❌ Error during prediction. Please try again.").send()
            return
        
        # Find similar examples
        similar_examples = classifier.find_similar_examples(description, remarks, top_k=3)
        
        # Format response
        response = format_prediction_response(
            description, remarks, prediction, confidence, 
            additional_info, similar_examples
        )
        
        await processing_msg.remove()
        await cl.Message(content=response).send()
        
        # Log the prediction
        logger.info(f"Prediction: {prediction}, Confidence: {confidence:.3f}")
        
    except Exception as e:
        await processing_msg.remove()
        await cl.Message(content=f"❌ Error processing your request: {str(e)}").send()
        logger.error(f"Message processing error: {e}")

def parse_input(user_input: str) -> Tuple[str, str]:
    """Parse user input to extract description and remarks"""
    description = ""
    remarks = ""
    
    if "Description:" in user_input and "Remarks:" in user_input:
        # Structured input
        parts = user_input.split("Remarks:")
        description = parts[0].replace("Description:", "").strip()
        remarks = parts[1].strip()
    elif "Description:" in user_input:
        # Only description provided
        description = user_input.replace("Description:", "").strip()
    elif "Remarks:" in user_input:
        # Only remarks provided
        remarks = user_input.replace("Remarks:", "").strip()
    else:
        # Free text - treat as description
        description = user_input
    
    return description, remarks

def format_prediction_response(description: str, remarks: str, prediction: str, 
                             confidence: float, additional_info: Dict[str, Any], 
                             similar_examples: List[Tuple[str, float]]) -> str:
    """Format the prediction response"""
    confidence_percentage = confidence * 100
    emoji = "✅" if prediction.lower() == "yes" else "❌"
    
    # Confidence level
    if confidence_percentage >= 80:
        confidence_level = "🟢 High"
    elif confidence_percentage >= 60:
        confidence_level = "🟡 Medium"
    else:
        confidence_level = "🔴 Low"
    
    response = f"""
## {emoji} Classification Result

### 📝 Input:
- **Description**: {description if description else "Not provided"}
- **Remarks**: {remarks if remarks else "Not provided"}

### 🎯 Prediction:
- **Result**: **{prediction.upper()}**
- **Confidence**: {confidence_percentage:.1f}% ({confidence_level})

### 📊 Probability Distribution:
- **No**: {additional_info.get('probabilities', {}).get('no', 0)*100:.1f}%
- **Yes**: {additional_info.get('probabilities', {}).get('yes', 0)*100:.1f}%

### 🔍 Similar Examples:
"""
    
    for i, (example, similarity) in enumerate(similar_examples, 1):
        response += f"{i}. **{similarity:.3f}** - {example}\n"
    
    response += f"""
### 🔧 Technical Info:
- **Embedding Dimension**: {additional_info.get('embedding_dim', 'N/A')}
- **Text Length**: {additional_info.get('text_length', 'N/A')} characters
- **Timestamp**: {additional_info.get('timestamp', 'N/A')}

---
*Confidence bar:* {'█' * int(confidence_percentage // 10)}{'░' * (10 - int(confidence_percentage // 10))} {confidence_percentage:.1f}%
    """
    
    return response

async def handle_info_command():
    """Handle info command"""
    global classifier
    
    info_msg = f"""
# 📊 Model Information

## 🔧 Configuration:
- **Sentence Transformer**: {classifier.model_name}
- **Device**: {classifier.device}
- **Classifier Type**: {type(classifier.classifier).__name__ if classifier.classifier else 'Not trained'}
- **Training Status**: {'✅ Trained' if classifier.is_trained else '❌ Not trained'}

## 📈 Training Data:
- **Total Examples**: {len(classifier.training_data)}
- **Classes**: {list(classifier.label_encoder.keys())}

## 🎯 Available Commands:
- `info` - Show this information
- `retrain` - Retrain the model
- `similar [text]` - Find similar examples
- Regular text - Classify the input

## 📝 Example Training Data:
"""
    
    for i, (desc, rem, label) in enumerate(classifier.training_data[:3], 1):
        info_msg += f"{i}. **{label.upper()}** - {desc} | {rem}\n"
    
    if len(classifier.training_data) > 3:
        info_msg += f"... and {len(classifier.training_data) - 3} more examples\n"
    
    await cl.Message(content=info_msg).send()

async def handle_retrain_command():
    """Handle retrain command"""
    global classifier
    
    retraining_msg = cl.Message(content="🔄 Retraining the model...")
    await retraining_msg.send()
    
    try:
        if classifier.train_classifier():
            # Save the retrained model
            classifier.save_model("./sentence_transformer_model.pkl")
            await retraining_msg.remove()
            await cl.Message(content="✅ Model retrained successfully!").send()
        else:
            await retraining_msg.remove()
            await cl.Message(content="❌ Failed to retrain the model.").send()
            
    except Exception as e:
        await retraining_msg.remove()
        await cl.Message(content=f"❌ Error during retraining: {str(e)}").send()

async def handle_similar_command(user_input: str):
    """Handle similar command"""
    global classifier
    
    # Extract text after "similar" command
    text = user_input[7:].strip()  # Remove "similar" and whitespace
    
    if not text:
        await cl.Message(content="⚠️ Please provide text after 'similar' command.\nExample: `similar The weather is nice today`").send()
        return
    
    description, remarks = parse_input(text)
    
    if not description and not remarks:
        await cl.Message(content="⚠️ Please provide at least a description or remarks.").send()
        return
    
    try:
        similar_examples = classifier.find_similar_examples(description, remarks, top_k=5)
        
        response = f"""
## 🔍 Similar Examples

### 📝 Input:
- **Description**: {description if description else "Not provided"}
- **Remarks**: {remarks if remarks else "Not provided"}

### 🎯 Most Similar Examples:
"""
        
        for i, (example, similarity) in enumerate(similar_examples, 1):
            response += f"{i}. **Similarity: {similarity:.3f}** - {example}\n"
        
        await cl.Message(content=response).send()
        
    except Exception as e:
        await cl.Message(content=f"❌ Error finding similar examples: {str(e)}").send()

@cl.on_stop
async def stop():
    """Clean up when stopping the application"""
    global classifier
    if classifier:
        # Clean up model resources
        if hasattr(classifier, 'sentence_model') and classifier.sentence_model:
            del classifier.sentence_model
        if hasattr(classifier, 'classifier') and classifier.classifier:
            del classifier.classifier
        classifier = None
        logger.info("Model resources cleaned up")

if __name__ == "__main__":
    # This file should be run with: chainlit run app.py
    pass