📊 Customer Satisfaction Classification using mBERT & Sentence Transformers
This project fine-tunes the Multilingual BERT (mBERT) model for binary text classification — classifying whether a customer is satisfied (yes) or not (no) — using structured customer feedback (description + remarks). It includes:

🧠 Full fine-tuning with HuggingFace Transformers

🗃️ CSV to JSONL dataset preparation

📈 Evaluation with metrics & confidence scores

🤖 Chainlit-powered interactive classifier using sentence transformers

🔧 Project Structure
├── prepare_dataset.py          # Prepares dataset from CSV → JSONL
├── train_lora.py               # Fine-tunes mBERT on the JSONL data
├── evaluate_model.py           # Evaluates model on test JSONL
├── run_pipeline.py             # Downloads mBERT from HuggingFace
├── app.py                      # Chainlit app using sentence transformers
├── combine_csv.ipynb          # Combines and pre-processes CSVs (Jupyter notebook)
├── requirements.txt            # Python dependencies
🚀 Setup Instructions
1. Clone & Install Dependencies

git clone <your-repo-url>
cd <your-repo-folder>

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
2. 🔄 Prepare Dataset
Ensure you have a clean CSV with the following columns:

uniqid

description

remarks

satisfied (must be yes or no)

Then run:

python prepare_dataset.py
This will generate:

train.jsonl (for training)

test.jsonl (for evaluation)

manual_eval.jsonl (optional subset for manual review)

Output saved to: E:\newJKproject\test (customize path in script if needed)

3. ⬇️ Download mBERT Model
Before training, download mBERT:

python run_pipeline.py
This will save bert-base-multilingual-cased locally in the configured directory.

4. 🎯 Train the Model
Fine-tune mBERT on your dataset:

python train_lora.py
Loads JSONL dataset

Tokenizes and prepares for training

Trains classification head

Saves model and tokenizer to mbert_finetuned_output/

This script performs full fine-tuning (no LoRA by default).

5. 📊 Evaluate the Model
Evaluate model on test data:

python evaluate_model.py --data E:\newJKproject\test\test.jsonl --mode huggingface --model_dir mbert_finetuned_output
Supports huggingface and ollama backends

Reports accuracy, confidence scores, and detailed prediction logs

Evaluation results saved to results/ folder.

🧪 Chainlit Interface (Optional)
Run an interactive web-based classifier:

chainlit run app.py
Features:
Sentence-transformer based UI (MiniLM)

Logistic/Random Forest/SVM classifier options

Confidence bars and similar examples

Commands like info, retrain, similar <input>

Accessible locally via: http://localhost:8000/

✅ Model Inputs and Outputs
Each data point follows this structure:

{
  "instruction": "Classify whether the customer is satisfied based on the following details.",
  "input": "Description: <text>\nRemarks:\n- <remark1>\n- <remark2>",
  "output": "yes" or "no"
}
⚙️ Configuration
Adjust paths and settings inside:

prepare_dataset.py: Dataset CSV path & output

train_lora.py: MODEL_DIR, DATA_DIR, and OUTPUT_DIR

evaluate_model.py: Command-line arguments

app.py: Sentence transformer model name & classifier type

📦 Dependencies
See requirements.txt for exact versions. Key libraries:

transformers, datasets, peft

scikit-learn, sentence-transformers

torch, numpy, pandas

chainlit (for UI)

Install via:

pip install -r requirements.txt
📁 Output Files
File / Folder	Description
train.jsonl / test.jsonl	Processed dataset for training/testing
mbert_finetuned_output/	Trained model weights + tokenizer
results/*.json	Evaluation reports
sentence_transformer_model.pkl	Pickled classifier for Chainlit UI

📌 Notes
If you're using LoRA adapters, modify paths in evaluate_model.py

Chainlit uses a separate embedding-based classifier — not mBERT

Code includes GPU and CPU fallback logic

CSV parsing is robust to encoding/memory issues

✨ Example Prompt
{
  "instruction": "Classify whether the customer is satisfied based on the following details.",
  "input": "Description: The app works well.\nRemarks:\n- Very smooth experience\n- Fast loading",
  "output": "yes"
}
🤝 Credits
Developed by: Avanindra Vijay
Technologies: Hugging Face Transformers, PEFT, Chainlit, Scikit-learn, mBERT

