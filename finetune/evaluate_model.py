import os
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import PeftModel
import numpy as np

# Optional: Ollama import for inference
try:
    import ollama
except ImportError:
    ollama = None

def load_dataset(jsonl_path):
    """Load dataset from JSONL file"""
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    print(f"✅ Loaded {len(data)} samples from {jsonl_path}")
    return data

def predict_huggingface(model, tokenizer, text, label2id, id2label):
    """Generate prediction using HuggingFace mBERT model"""
    try:
        # Tokenize input
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=512
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        # Get prediction
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        predicted_label = id2label[predicted_class_id]
        confidence = torch.softmax(logits, dim=-1).max().item()
        
        return predicted_label, confidence
            
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return "error", 0.0

def predict_ollama(prompt, model_name="llama3.2"):
    """Generate prediction using Ollama (fallback option)"""
    try:
        response = ollama.chat(model=model_name, messages=[
            {"role": "user", "content": prompt}
        ])
        
        content = response['message']['content'].strip().lower()
        
        # Extract yes/no from response
        if "yes" in content and "no" not in content:
            return "yes", 1.0
        elif "no" in content and "yes" not in content:
            return "no", 1.0
        else:
            return content, 0.5
            
    except Exception as e:
        print(f"❌ Error during Ollama prediction: {e}")
        return "error", 0.0

def evaluate(data_path, mode="huggingface", model_dir="mbert_lora_output", 
             base_model_dir=None, model_name="llama3.2"):
    """Evaluate model performance"""
    
    # Load dataset
    data = load_dataset(data_path)
    
    results = []
    correct = 0
    total = len(data)
    
    if mode == "huggingface":
        print(f"🔄 Loading mBERT model from: {model_dir}")
        
        # Check if this is a LoRA model
        if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
            print("📦 Detected LoRA adapter, loading base model + adapter...")
            
            if base_model_dir is None:
                base_model_dir = r"E:\newJKproject\model"  # Default base model path
            
            # Load base model config first
            config = AutoConfig.from_pretrained(base_model_dir)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
            
            # Load base model for sequence classification
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_dir,
                config=config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load LoRA adapter
            model = PeftModel.from_pretrained(base_model, model_dir)
            
            # Get label mappings from config
            label2id = config.label2id if hasattr(config, 'label2id') and config.label2id else {"no": 0, "yes": 1}
            id2label = config.id2label if hasattr(config, 'id2label') and config.id2label else {0: "no", 1: "yes"}
            
        else:
            print("📦 Loading full fine-tuned mBERT model...")
            
            # Load config
            config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            
            # Load model
            model = AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                config=config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Get label mappings
            label2id = config.label2id if hasattr(config, 'label2id') and config.label2id else {"no": 0, "yes": 1}
            id2label = config.id2label if hasattr(config, 'id2label') and config.id2label else {0: "no", 1: "yes"}
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
        
        model.eval()
        print("✅ mBERT model loaded successfully")
        print(f"📋 Label mappings: {label2id}")
    
    elif mode == "ollama":
        if ollama is None:
            raise ImportError("❌ You must install the `ollama` library to use Ollama inference.")
        print(f"🔄 Using Ollama model: {model_name}")
        label2id = {"no": 0, "yes": 1}
        id2label = {0: "no", 1: "yes"}
    
    # Evaluate each sample
    print("\n🔍 Starting evaluation...")
    confidences = []
    
    for idx, item in enumerate(data):
        # Prepare input text
        if mode == "huggingface":
            # For mBERT classification, combine instruction and input
            text = f"{item['instruction']} {item['input']}"
            predicted, confidence = predict_huggingface(model, tokenizer, text, label2id, id2label)
        elif mode == "ollama":
            # For Ollama, use the prompt format
            prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:"
            predicted, confidence = predict_ollama(prompt, model_name)
        else:
            raise ValueError("❌ Mode must be 'huggingface' or 'ollama'")
        
        # Get expected output
        expected = item["output"].strip().lower()
        
        # Normalize prediction
        if isinstance(predicted, str):
            predicted = predicted.strip().lower()
        
        # Check if prediction is correct
        is_correct = (expected == predicted)
        
        if is_correct:
            correct += 1
        
        confidences.append(confidence)
        
        # Store result
        result = {
            "id": idx + 1,
            "instruction": item["instruction"],
            "input": item["input"],
            "expected": expected,
            "predicted": predicted,
            "confidence": confidence,
            "is_correct": is_correct
        }
        results.append(result)
        
        # Print progress
        status = "✅" if is_correct else "❌"
        print(f"{idx + 1:03d}/{total:03d}) {status} Expected: '{expected}' | Predicted: '{predicted}' | Confidence: {confidence:.3f}")
    
    # Calculate metrics
    accuracy = (correct / total) * 100
    avg_confidence = np.mean(confidences) if confidences else 0.0
    
    print(f"\n📊 Final Results:")
    print(f"✅ Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"📈 Average Confidence: {avg_confidence:.3f}")
    print(f"✅ Correct predictions: {correct}")
    print(f"❌ Incorrect predictions: {total - correct}")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    
    # Determine output filename
    if "manual" in data_path.lower():
        filename = "eval_report_manual_mbert.json"
    else:
        filename = "eval_report_test_split_mbert.json"
    
    output_path = os.path.join("results", filename)
    
    # Create comprehensive report
    report = {
        "evaluation_config": {
            "mode": mode,
            "model_dir": model_dir,
            "base_model_dir": base_model_dir,
            "model_name": model_name,
            "data_path": data_path,
            "model_type": "mBERT"
        },
        "results": {
            "accuracy": accuracy,
            "average_confidence": avg_confidence,
            "total_samples": total,
            "correct_predictions": correct,
            "incorrect_predictions": total - correct
        },
        "label_mappings": {
            "label2id": label2id if mode == "huggingface" else {"no": 0, "yes": 1},
            "id2label": id2label if mode == "huggingface" else {0: "no", 1: "yes"}
        },
        "detailed_results": results
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📁 Detailed report saved to: {output_path}")
    
    return accuracy, results

def main():
    parser = argparse.ArgumentParser(description="Evaluate mBERT classification model")
    parser.add_argument("--mode", choices=["huggingface", "ollama"], default="huggingface", 
                       help="Model source for inference")
    parser.add_argument("--data", type=str, required=True, 
                       help="Path to evaluation .jsonl file")
    parser.add_argument("--model_dir", type=str, default="mbert_lora_output", 
                       help="Path to fine-tuned mBERT model directory")
    parser.add_argument("--base_model_dir", type=str, default=None,
                       help="Path to base mBERT model directory (for LoRA adapters)")
    parser.add_argument("--model_name", type=str, default="llama3.2", 
                       help="Ollama model name (fallback option)")
    
    args = parser.parse_args()
    
    try:
        accuracy, results = evaluate(
            data_path=args.data,
            mode=args.mode,
            model_dir=args.model_dir,
            base_model_dir=args.base_model_dir,
            model_name=args.model_name
        )
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"Final accuracy: {accuracy:.2f}%")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()