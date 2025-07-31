import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    AutoModel,
    BertForSequenceClassification,
    BertConfig
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn as nn

# === Configuration ===
MODEL_DIR = r"E:\newJKproject\model"  # Your mBERT model folder
DATA_DIR = r"E:\newJKproject\test"    # Your JSONL files folder
OUTPUT_DIR = "mbert_finetuned_output"  # Where to save fine-tuned model

def main():
    print("🚀 Starting mBERT Fine-tuning (No PEFT Version)...")
    
    # Check if model directory exists
    if not os.path.exists(MODEL_DIR):
        print(f"❌ Model directory not found: {MODEL_DIR}")
        print("Please download mBERT model first")
        return
    
    # Check if data files exist
    train_file = os.path.join(DATA_DIR, "train.jsonl")
    test_file = os.path.join(DATA_DIR, "test.jsonl")
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"❌ Data files not found. Please run prepare_dataset.py first")
        return
    
    print("✅ All required files found")
    
    # === Load Dataset First to Determine Labels ===
    print("🔄 Loading dataset to determine labels...")
    try:
        dataset = load_dataset("json", data_files={
            "train": train_file,
            "test": test_file
        })
        
        # Determine labels
        if 'label' in dataset['train'].column_names:
            # If we have explicit labels
            label_list = dataset['train'].unique('label')
            num_labels = len(label_list)
            print(f"📋 Found {num_labels} labels: {label_list}")
        else:
            # If we need to create labels from outputs
            label_list = dataset['train'].unique('output')
            num_labels = len(label_list)
            print(f"📋 Creating {num_labels} labels from outputs: {label_list}")
            
            # Create label mapping
            label2id = {label: idx for idx, label in enumerate(label_list)}
            id2label = {idx: label for idx, label in enumerate(label_list)}
            
            # Function to map outputs to labels
            def map_labels(examples):
                return {'labels': [label2id[output] for output in examples['output']]}
            
            dataset = dataset.map(map_labels, batched=True)
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # === Load Tokenizer ===
    print("🔄 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

        # Set pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token

        print("✅ Tokenizer loaded successfully")

    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return

    # === Load and Inspect Model Config ===
    print("🔄 Loading model configuration...")
    try:
        config = AutoConfig.from_pretrained(MODEL_DIR)
        print(f"📋 Original config type: {type(config)}")
        print(f"📋 Model architecture: {config.architectures if hasattr(config, 'architectures') else 'Not specified'}")
        
        # Check if it's an encoder-decoder model
        if hasattr(config, 'encoder') and hasattr(config, 'decoder'):
            print("⚠️ This is an encoder-decoder model. Converting to sequence classification...")
            
            # Use the encoder part for sequence classification
            encoder_config = config.encoder
            
            # Create a new BERT config for sequence classification
            if hasattr(encoder_config, 'hidden_size'):
                bert_config = BertConfig(
                    vocab_size=encoder_config.vocab_size,
                    hidden_size=encoder_config.hidden_size,
                    num_hidden_layers=encoder_config.num_hidden_layers,
                    num_attention_heads=encoder_config.num_attention_heads,
                    intermediate_size=encoder_config.intermediate_size,
                    hidden_act=encoder_config.hidden_act,
                    hidden_dropout_prob=getattr(encoder_config, 'hidden_dropout_prob', 0.1),
                    attention_probs_dropout_prob=getattr(encoder_config, 'attention_probs_dropout_prob', 0.1),
                    max_position_embeddings=encoder_config.max_position_embeddings,
                    type_vocab_size=getattr(encoder_config, 'type_vocab_size', 2),
                    initializer_range=getattr(encoder_config, 'initializer_range', 0.02),
                    layer_norm_eps=getattr(encoder_config, 'layer_norm_eps', 1e-12),
                    num_labels=num_labels,
                    id2label=id2label if 'id2label' in locals() else None,
                    label2id=label2id if 'label2id' in locals() else None
                )
                config = bert_config
            else:
                print("❌ Cannot extract encoder configuration")
                return
        else:
            # Regular BERT config, just update for sequence classification
            config.num_labels = num_labels
            if 'id2label' in locals():
                config.id2label = id2label
                config.label2id = label2id
                
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return

    # === Load Model ===
    print("🔄 Loading model...")
    try:
        # Try loading as BERT for sequence classification
        model = BertForSequenceClassification.from_pretrained(
            MODEL_DIR,
            config=config,
            ignore_mismatched_sizes=True  # This helps with size mismatches
        )
        print("✅ Model loaded successfully as BertForSequenceClassification")
        
    except Exception as e:
        print(f"⚠️ Failed to load as BertForSequenceClassification: {e}")
        print("🔄 Trying alternative loading method...")
        
        try:
            # Load base model and add classification head
            base_model = AutoModel.from_pretrained(MODEL_DIR)
            
            # Create a custom classification model
            class CustomSequenceClassifier(nn.Module):
                def __init__(self, base_model, num_labels):
                    super().__init__()
                    self.base_model = base_model
                    self.num_labels = num_labels
                    
                    # Get hidden size from base model
                    if hasattr(base_model.config, 'hidden_size'):
                        hidden_size = base_model.config.hidden_size
                    elif hasattr(base_model.config, 'encoder') and hasattr(base_model.config.encoder, 'hidden_size'):
                        hidden_size = base_model.config.encoder.hidden_size
                    else:
                        hidden_size = 768  # Default for BERT
                    
                    self.classifier = nn.Linear(hidden_size, num_labels)
                    self.dropout = nn.Dropout(0.1)
                    
                def forward(self, input_ids, attention_mask=None, labels=None):
                    outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    # Get pooled output or use CLS token
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        pooled_output = outputs.pooler_output
                    else:
                        # Use CLS token (first token)
                        pooled_output = outputs.last_hidden_state[:, 0]
                    
                    pooled_output = self.dropout(pooled_output)
                    logits = self.classifier(pooled_output)
                    
                    loss = None
                    if labels is not None:
                        loss_fct = nn.CrossEntropyLoss()
                        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                    
                    return {"loss": loss, "logits": logits}
            
            model = CustomSequenceClassifier(base_model, num_labels)
            model.config = config  # Add config for compatibility
            print("✅ Model loaded successfully with custom classification head")
            
        except Exception as e2:
            print(f"❌ All model loading methods failed: {e2}")
            return

    print("ℹ️ Performing standard fine-tuning (no LoRA)")
    
    # === Tokenize Dataset ===
    print("🔄 Tokenizing dataset...")
    def preprocess_function(examples):
        # Combine instruction and input
        texts = [f"{inst} {inp}".strip() for inst, inp in zip(examples['instruction'], examples['input'])]
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=512
        )
        
        # Add labels if they exist
        if 'labels' in examples:
            tokenized['labels'] = examples['labels']
            
        return tokenized
    
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )

    # === Define metrics ===
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    # === Training Arguments ===
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,  # Further reduced for full fine-tuning
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # Increased to compensate for smaller batch
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        num_train_epochs=3,
        logging_steps=20,
        save_total_limit=2,
        learning_rate=2e-5,  # Lower learning rate for full fine-tuning
        fp16=False,  # Disable fp16 for CPU training
        remove_unused_columns=False,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        dataloader_num_workers=0  # Disable multiprocessing for Windows
    )
    
    # === Data Collator ===
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest"
    )
    
    # === Initialize Trainer ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        processing_class=tokenizer,  # Use processing_class instead
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # === Start Training ===
    print("🚀 Starting training...")
    try:
        trainer.train()
        
        # === Save Model ===
        print("💾 Saving model...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        print(f"✅ Training completed! Model saved to: {OUTPUT_DIR}")
        
        # === Training Summary ===
        print("\n📊 Training Summary:")
        print(f"Training samples: {len(tokenized_dataset['train'])}")
        print(f"Validation samples: {len(tokenized_dataset['test'])}")
        print(f"Number of labels: {num_labels}")
        print(f"Epochs: {training_args.num_train_epochs}")
        print(f"Batch size: {training_args.per_device_train_batch_size}")
        print(f"Learning rate: {training_args.learning_rate}")
        
        # === Final Evaluation ===
        print("\n📈 Final Evaluation:")
        eval_results = trainer.evaluate()
        for key, value in eval_results.items():
            print(f"{key}: {value:.4f}")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()