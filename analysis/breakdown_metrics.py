import json
import numpy as np
import os
from datasets import load_dataset
from collections import defaultdict

# --- Configuration ---
RESULTS_FILE = "qwen_base_closedbook_full_results.json"
DATASET_NAME = "hotpot_qa"
SUBSET_NAME = "distractor"
SPLIT = "validation"
OUTPUT_FOLDER = "breakdown"
OUTPUT_FILENAME = "hotpot_qa_level_breakdown.json"

def main():
    # 1. Load Original Dataset for Level Mapping
    print(f"Loading original dataset ({DATASET_NAME}/{SUBSET_NAME}/{SPLIT}) to retrieve difficulty levels...")
    dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split=SPLIT)
    
    # Create lookup: id -> level (easy, medium, hard)
    id_to_level = {item['id']: item['level'] for item in dataset}
    
    # 2. Load Your Generated Results
    print(f"Loading generation results from {RESULTS_FILE}...")
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: File {RESULTS_FILE} not found.")
        return

    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # 3. Aggregate Metrics
    metrics_by_level = defaultdict(lambda: {'em': [], 'f1': []})
    missing_ids = 0

    print("Aggregating metrics by difficulty...")
    for item in results:
        uid = item['id']
        
        if uid in id_to_level:
            level = id_to_level[uid]
            
            # Assuming metrics in JSON are 0.0-1.0 based on your previous script
            em = item['metrics']['exact_match']
            f1 = item['metrics']['f1_score']
            
            metrics_by_level[level]['em'].append(em)
            metrics_by_level[level]['f1'].append(f1)
        else:
            missing_ids += 1

    if missing_ids > 0:
        print(f"Warning: {missing_ids} IDs not found in the original dataset.")

    # 4. Print & Prepare Data for Saving
    print("\n" + "="*50)
    print(f"HOTPOT QA RESULTS BY DIFFICULTY LEVEL")
    print("="*50)
    print(f"{'Level':<10} | {'Count':<8} | {'EM (%)':<10} | {'F1 (%)':<10}")
    print("-" * 50)

    order = ['easy', 'medium', 'hard']
    final_breakdown = {}

    for level in order:
        if level in metrics_by_level:
            data = metrics_by_level[level]
            count = len(data['em'])
            # Convert 0-1 scores to percentages
            avg_em = np.mean(data['em']) * 100 if count > 0 else 0
            avg_f1 = np.mean(data['f1']) * 100 if count > 0 else 0
            
            print(f"{level.capitalize():<10} | {count:<8} | {avg_em:<10.2f} | {avg_f1:<10.2f}")
            
            final_breakdown[level] = {
                "count": count,
                "em": avg_em,
                "f1": avg_f1
            }

    # Overall stats for verification
    all_em = [x for level in metrics_by_level for x in metrics_by_level[level]['em']]
    all_f1 = [x for level in metrics_by_level for x in metrics_by_level[level]['f1']]
    overall_em = np.mean(all_em) * 100 if all_em else 0
    overall_f1 = np.mean(all_f1) * 100 if all_f1 else 0
    
    print("-" * 50)
    print(f"{'Overall':<10} | {len(all_em):<8} | {overall_em:<10.2f} | {overall_f1:<10.2f}")
    print("="*50)

    # 5. Save to Folder
    # Create directory if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"\nCreated directory: {OUTPUT_FOLDER}")

    output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_breakdown, f, indent=2)
    
    print(f"Breakdown saved successfully to: {output_path}")

if __name__ == "__main__":
    main()