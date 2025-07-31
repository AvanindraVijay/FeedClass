import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
import gc

# Path to your dataset
CSV_PATH = r"E:\newJKproject\data\cleaned_file.csv"

# Output folder
OUTPUT_DIR = r"E:\newJKproject\test"
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.jsonl")
TEST_FILE = os.path.join(OUTPUT_DIR, "test.jsonl")
MANUAL_FILE = os.path.join(OUTPUT_DIR, "manual_eval.jsonl")

# Make sure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_file_size(filepath):
    """Get file size in MB"""
    size_bytes = os.path.getsize(filepath)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def read_csv_safely(filepath):
    """Read CSV with memory-efficient options"""
    file_size = get_file_size(filepath)
    print(f"📊 File size: {file_size:.2f} MB")
    
    # Strategy 1: Try standard reading with C engine
    print("🔄 Attempting standard reading...")
    try:
        df = pd.read_csv(filepath, 
                        low_memory=False,
                        dtype=str,  # Read all columns as strings initially
                        encoding='utf-8')
        print(f"✅ Standard reading successful: {len(df)} rows")
        return df
    except Exception as e:
        print(f"❌ Standard reading failed: {e}")
    
    # Strategy 2: Try with Python engine (no low_memory option)
    print("🔄 Attempting Python engine reading...")
    try:
        df = pd.read_csv(filepath, 
                        dtype=str,  # Read all columns as strings initially
                        engine='python',  # Use Python engine (slower but more robust)
                        encoding='utf-8',
                        on_bad_lines='skip')  # Skip problematic lines
        print(f"✅ Python engine reading successful: {len(df)} rows")
        return df
    except Exception as e:
        print(f"❌ Python engine reading failed: {e}")
    
    # Strategy 3: Try chunked reading
    print("🔄 Attempting chunked reading...")
    try:
        chunks = []
        chunk_size = 5000  # Read 5k rows at a time
        
        for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunk_size, dtype=str, encoding='utf-8')):
            chunks.append(chunk)
            print(f"✅ Read chunk {i+1} with {len(chunk)} rows")
            
            # Limit chunks to avoid memory issues
            if len(chunks) > 50:  # Stop after 250k rows
                print("⚠️ Limiting to first 250k rows to avoid memory issues")
                break
        
        df = pd.concat(chunks, ignore_index=True)
        print(f"✅ Combined all chunks: {len(df)} total rows")
        return df
        
    except Exception as e:
        print(f"❌ Chunked reading failed: {e}")
    
    # Strategy 4: Try with different encodings
    print("🔄 Attempting different encodings...")
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            print(f"   Trying encoding: {encoding}")
            df = pd.read_csv(filepath, 
                            dtype=str,
                            encoding=encoding,
                            on_bad_lines='skip')
            print(f"✅ Reading with {encoding} successful: {len(df)} rows")
            return df
        except Exception as e:
            print(f"   Failed with {encoding}: {str(e)[:50]}...")
    
    return None

def main():
    print("🔄 Loading dataset...")
    
    # Check if file exists
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV file not found: {CSV_PATH}")
        return
    
    # Try to read the CSV with different strategies
    df = read_csv_safely(CSV_PATH)
    
    if df is None:
        print("❌ Failed to read CSV file. Try these manual fixes:")
        print("1. Check if the file is corrupted")
        print("2. Open the file in a text editor to check for formatting issues")
        print("3. Try splitting the file into smaller chunks")
        print("4. Check for memory issues (close other applications)")
        return
    
    print(f"✅ Loaded {len(df)} rows from CSV")
    print(f"📊 Columns: {list(df.columns)}")
    
    # Check required columns
    required_columns = ['uniqid', 'description', 'remarks', 'satisfied']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Missing columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Memory optimization: work with smaller chunks
    print("🔄 Optimizing memory usage...")
    
    # Fill NaN values
    df = df.fillna("")
    
    # Clean and normalize 'satisfied' column
    df['satisfied'] = df['satisfied'].astype(str).str.strip().str.lower()
    
    # Filter only valid satisfied values
    valid_satisfied = df['satisfied'].isin(['yes', 'no'])
    df = df[valid_satisfied]
    print(f"✅ Filtered to {len(df)} rows with valid 'satisfied' values")
    
    # Free up memory
    gc.collect()
    
    # Group by uniqid and combine remarks
    print("🔄 Grouping data by uniqid...")
    try:
        grouped = df.groupby("uniqid").agg({
            "description": "first",
            "remarks": lambda x: list(set([r for r in x if r.strip()])),  # Remove empty remarks
            "satisfied": "first"
        }).reset_index()
        
        print(f"✅ Grouped into {len(grouped)} unique records")
        
        # Free up memory from original dataframe
        del df
        gc.collect()
        
    except MemoryError:
        print("❌ Memory error during grouping. Try processing in smaller batches.")
        return
    
    # Create formatted prompts
    def create_prompt(row):
        instruction = "Classify whether the customer is satisfied based on the following details."
        
        # Handle remarks properly
        remarks_list = row['remarks']
        if not remarks_list or (len(remarks_list) == 1 and remarks_list[0] == ''):
            remarks_text = "No specific remarks provided."
        else:
            remarks_text = "- " + "\n- ".join(remarks_list)
        
        input_text = f"Description: {row['description']}\nRemarks:\n{remarks_text}"
        output = row["satisfied"]
        
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output
        }
    
    # Apply formatting
    print("🔄 Creating formatted prompts...")
    formatted_data = []
    
    # Process in batches to avoid memory issues
    batch_size = 1000
    for i in range(0, len(grouped), batch_size):
        batch = grouped.iloc[i:i+batch_size]
        batch_formatted = [create_prompt(row) for _, row in batch.iterrows()]
        formatted_data.extend(batch_formatted)
        print(f"✅ Processed batch {i//batch_size + 1}/{(len(grouped)-1)//batch_size + 1}")
    
    # Free up memory
    del grouped
    gc.collect()
    
    # Split train-test
    print("🔄 Splitting dataset...")
    train_data, test_data = train_test_split(formatted_data, test_size=0.3, random_state=42)
    
    # Save as .jsonl
    def save_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
    
    print("🔄 Saving files...")
    save_jsonl(train_data, TRAIN_FILE)
    save_jsonl(test_data, TEST_FILE)
    
    # Save first 100 samples for manual evaluation
    manual_eval_data = test_data[:min(100, len(test_data))]
    save_jsonl(manual_eval_data, MANUAL_FILE)
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"📊 Train samples: {len(train_data)}")
    print(f"📊 Test samples: {len(test_data)}")
    print(f"📊 Manual eval samples: {len(manual_eval_data)}")
    print(f"📁 Files saved to: {OUTPUT_DIR}")
    
    # Show sample data
    print("\n📋 Sample training data:")
    print(json.dumps(train_data[0], indent=2))

if __name__ == "__main__":
    main()