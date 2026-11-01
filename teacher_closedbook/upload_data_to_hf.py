import json
import os
from datasets import Dataset
import argparse

# Configuration
# TARGET_COUNT = 1000
# INPUT_FILE = "hotpot_reasoning_{TARGET_COUNT}.jsonl"
HF_REPO_ID = "fsiddiqui2/hotpot-qa-closedbook-cot-reasoning"  # CHANGE THIS

def upload_dataset(input_file):
    # 1. Check if file exists
    if not os.path.exists(input_file):
        print(f"❌ File '{input_file}' not found.")
        print("   Please run 'generate_hotpot_cot.py' first to generate the data.")
        return

    # 2. Load the JSONL data
    print(f"📖 Reading {input_file}...")
    data = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    if not data:
        print("⚠️ Dataset is empty. Nothing to upload.")
        return

    print(f"✅ Loaded {len(data)} examples.")

    # 3. Convert to Hugging Face Dataset and Push
    print(f"🚀 Pushing to Hugging Face Hub: {HF_REPO_ID}...")
    
    try:
        # Convert list of dicts to a Dataset object
        hf_dataset = Dataset.from_list(data)
        
        # Push to Hub
        hf_dataset.push_to_hub(HF_REPO_ID)
        
        print("\n🎉 Success!")
        print(f"🔗 View your dataset here: https://huggingface.co/{HF_REPO_ID}")
        
    except Exception as e:
        print(f"\n❌ Failed to push to Hugging Face: {e}")
        print("💡 Troubleshooting tips:")
        print("   1. Ensure you have the 'datasets' and 'huggingface_hub' libraries installed.")
        print("   2. Make sure you are logged in via terminal: `huggingface-cli login`")
        print("   3. Verify that 'HF_REPO_ID' is correct and you have write permissions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="The name of the json file to upload to HF.")
    args = parser.parse_args()
    
    upload_dataset(args.data_path)