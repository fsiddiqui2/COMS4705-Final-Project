
import json
import pandas as pd
import os
import argparse

# --- Configuration ---
# Define the files for each group
GROUPS = {
    "VM_MODELS": {
        "base": "results/qwen_base_closedbook_full_results.json",
        "sft": "results/qwen_sft_closedbook_full_results.json",
        "cot": "results/qwen_ftcot_closedbook_rauq_results.json",
        "abstain": "results/qwen_abstention_rauq_results.json"
    },
    "TINKER_MODELS": {
        "base": "tinker/results/tinker_vanilla_results.json",
        "sft": "tinker/results/tinker_eval_results.json",
        "cot": "tinker/results/tinker_cot_results.json",
        "abstain": "tinker/results/tinker_abstention_results.json"
    }
}


def load_and_normalize(filename, model_name, id_map=None):
    """
    Loads a results JSON and returns a DataFrame with:
    [id, {model}_em, {model}_abstain, {model}_pred, question, gold_answers]
    
    id_map: dict {question_text: id} to backfill missing IDs
    """
    if not os.path.exists(filename):
        print(f"Warning: File not found: {filename}")
        return None

    with open(filename, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Normalize ID
    if 'id' not in df.columns:
        if 'question' in df.columns and id_map is not None:
            # Map ID from question
            df['id'] = df['question'].map(id_map)
            # Drop rows where ID couldn't be found
            missing_count = df['id'].isna().sum()
            if missing_count > 0:
                print(f"  Warning: {missing_count} rows dropped due to missing ID mapping for {model_name}")
            df = df.dropna(subset=['id'])
        else:
            print(f"  Error: {model_name} missing 'id' column and no map/question provided.")
            return None


    # Normalize EM
    # Check for 'metrics' dict or direct keys
    if 'metrics' in df.columns:
        # Check first row
        first_metric = df['metrics'].iloc[0] if len(df) > 0 else {}
        metric_key = 'exact_match' if 'exact_match' in first_metric else 'em'
        df[f'{model_name}_em'] = df['metrics'].apply(lambda x: x.get(metric_key, 0.0))
    else:
        # Fallback (maybe 'exact_match' is top level)
        if 'exact_match' in df.columns:
            df[f'{model_name}_em'] = df['exact_match']
        elif 'em' in df.columns:
            df[f'{model_name}_em'] = df['em']
        else:
            df[f'{model_name}_em'] = 0.0

    # Normalize Abstention
    if 'is_abstention' in df.columns:
        df[f'{model_name}_abstain'] = df['is_abstention']
    elif 'prediction' in df.columns:
        # Heuristic check for "I don't know" if flag missing
        df[f'{model_name}_abstain'] = df['prediction'].astype(str).str.lower().str.contains("i don't know")
    else:
        df[f'{model_name}_abstain'] = False

    # Normalize Prediction
    pred_col = 'parsed_answer' if 'parsed_answer' in df.columns else 'prediction'
    if pred_col in df.columns:
        df[f'{model_name}_pred'] = df[pred_col]
    else:
        df[f'{model_name}_pred'] = ""

    # Keep key columns
    cols = ['id', f'{model_name}_em', f'{model_name}_abstain', f'{model_name}_pred']
    if 'question' in df.columns: cols.append('question')
    if 'gold_answers' in df.columns: cols.append('gold_answers')
    
    return df[cols]

    # Analyze VM Models
    analyze_group("VM (GCP) MODELS", GROUPS['VM_MODELS'])
    
    # Analyze Tinker Models
    analyze_group("TINKER MODELS", GROUPS['TINKER_MODELS'])

def analyze_group(group_name, files_dict):
    print(f"\n{'='*40}")
    print(f"ANALYZE GROUP: {group_name}")
    print(f"{'='*40}")

    dfs = []
    loaded_models = []
    
    # 1A. Load Base First (Canonical ID Source)
    base_file = files_dict.get('base')
    id_map = None
    
    if base_file:
        df_base = load_and_normalize(base_file, 'base')
        if df_base is not None:
            dfs.append(df_base)
            loaded_models.append('base')
            print(f"Loaded base: {len(df_base)} samples")
            # Build ID map
            if 'question' in df_base.columns:
                id_map = dict(zip(df_base['question'], df_base['id']))
    
    # 1B. Load Others
    for model_role, filepath in files_dict.items():
        if model_role == 'base': continue # Already loaded
        
        df = load_and_normalize(filepath, model_role, id_map=id_map)
        if df is not None:
            dfs.append(df)
            loaded_models.append(model_role)
            print(f"Loaded {model_role}: {len(df)} samples")
    
    if not dfs:
        print("No files loaded.")
        return

    # 2. Merge on ID (Intersection)
    merged = dfs[0]
    for df in dfs[1:]:
        # Find common columns to avoid duplicates (except ID)
        cols_to_use = df.columns.difference(merged.columns).tolist()
        cols_to_use.append('id')
        merged = pd.merge(merged, df[cols_to_use], on='id', how='inner')
    
    print(f"\nIntersection Size: {len(merged)} samples")
    if len(merged) == 0:
        print("Empty intersection. Stopping analysis.")
        return

    # 3. Overall Accuracy Table
    print("\n--- Overall Accuracy (Exact Match) ---")
    acc_rows = []
    for model in loaded_models:
        acc = merged[f'{model}_em'].mean()
        acc_rows.append({"Model": model, "EM": f"{acc:.2%}"})
    print(pd.DataFrame(acc_rows).to_markdown(index=False))

    # 4. Correction Analysis (Base vs Others)
    if 'base' in loaded_models:
        print("\n--- Improvement Analysis (vs Base) ---")
        ws_rows = [] # Wrong -> Right
        rw_rows = [] # Right -> Wrong
        
        for model in loaded_models:
            if model == 'base': continue
            
            # Corrections: Base Wrong (0) -> Model Right (1)
            # Ensure we check numeric equality (0.0 vs 1.0)
            corrections = merged[(merged['base_em'] == 0) & (merged[f'{model}_em'] == 1)]
            
            # Regressions: Base Right (1) -> Model Wrong (0)
            regressions = merged[(merged['base_em'] == 1) & (merged[f'{model}_em'] == 0)]
            
            ws_rows.append({
                "Model": model, 
                "Corrected (Base Wrong->Right)": len(corrections),
                "Regressed (Base Right->Wrong)": len(regressions),
                "Net Gain": len(corrections) - len(regressions)
            })
        
        if ws_rows:
            print(pd.DataFrame(ws_rows).to_markdown(index=False))

    # 5. Abstention Statistics (Precision/Recall)
    if 'abstain' in loaded_models:
        print("\n--- Abstention Analysis (Precision/Recall) ---")
        print("Assumption: 'Model Should Abstain' iff Reference Model is Wraong (EM=0)")
        
        # We check the 'abstain' model against every other loaded model as reference
        abs_rows = []
        
        for ref_model in loaded_models:
            if ref_model == 'abstain': continue
            
            # subset where reference is WRONG (Positive case for abstention)
            should_abstain = merged[f'{ref_model}_em'] == 0
            # subset where abstention model DID abstain
            did_abstain = merged['abstain_abstain'] == True  # Normalize func creates {model}_abstain column
            
            # TP: Should Abstain AND Did Abstain
            tp = len(merged[should_abstain & did_abstain])
            
            # FP: Should Answer (Ref Right) BUT Did Abstain (Abstain Flag True)
            # Wait, standard precision formula: 
            # Precision = TP / (TP + FP) = (Good Abstention) / (Total Abstention)
            # Recall = TP / (TP + FN) = (Good Abstention) / (Total Errors)
            
            should_answer = merged[f'{ref_model}_em'] == 1
            fp = len(merged[should_answer & did_abstain])
            
            # FN: Should Abstain (Ref Wrong) BUT Did NOT Abstain
            fn = len(merged[should_abstain & ~did_abstain])
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            abs_rows.append({
                "Reference Model": ref_model,
                "Precision": f"{precision:.2%}",
                "Recall": f"{recall:.2%}",
                "TP (Good Abs)": tp,
                "FP (Bad Abs)": fp,
                "FN (Missed Abs)": fn
            })
            
        print(pd.DataFrame(abs_rows).to_markdown(index=False))

def main():
    try:
        import tabulate # Required for to_markdown
    except ImportError:
        pass # Pandas might degrade gracefully or error, usually fine if just printing strings

    # Analyze VM Models
    analyze_group("VM (GCP) MODELS", GROUPS['VM_MODELS'])
    
    # Analyze Tinker Models
    analyze_group("TINKER MODELS", GROUPS['TINKER_MODELS'])

if __name__ == "__main__":
    main()
