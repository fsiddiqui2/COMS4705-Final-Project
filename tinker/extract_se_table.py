
import json
import numpy as np
import argparse
import pandas as pd

def get_f1_at_rejection(data, metric_key='f1', rejection_rate=0.0):
    """
    Calculates Mean F1 at a specific rejection rate.
    Rejection Rate R means we remove the top R% highest entropy items.
    """
    if not data: return 0.0
    
    # Sort by Entropy (Ascending Preference, i.e., Descending Entropy)
    # We want to KEEP low entropy.
    sorted_data = sorted(data, key=lambda x: x['entropy'])
    
    # Number to keep
    n_total = len(data)
    n_keep = int(n_total * (1 - rejection_rate))
    if n_keep == 0: return 0.0
    
    kept_data = sorted_data[:n_keep]
    
    # Calculate Mean F1 of VALID (kept) items
    # Note: 'f1' is already in the item dict
    scores = [x[metric_key] for x in kept_data]
    return np.mean(scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help="Path to tinker_sem_entropy_abstention_results.json")
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        results = json.load(f)

    # 1. Prepare Data for Greedy and Consensus
    # We need lists of dicts: {'entropy': float, 'f1': float}
    
    greedy_full_items = []
    greedy_selective_items = [] # Only native non-abstentions
    
    consensus_full_items = []
    consensus_selective_items = [] # Only native non-abstentions

    for item in results:
        # --- GREEDY ---
        g_dec = item['decisions']['greedy']
        g_f1 = g_dec['metrics']['f1']
        g_is_abstain = g_dec['is_abstain']
        # Entropy is shared (calculated on samples)
        entropy = item['entropy']
        
        # Adjust F1 for native abstention if not already handled
        # (The eval script usually sets F1=1.0 if Abstain is correct, else 0.0.
        # But for 'Selective' we typically exclude native abstentions entirely.)
        
        # FULL: Include everything. Native Abstentions have their assigned F1 (0 or 1).
        greedy_full_items.append({'entropy': entropy, 'f1': g_f1})
        
        # SELECTIVE: Only include if model attempted to answer
        if not g_is_abstain:
            greedy_selective_items.append({'entropy': entropy, 'f1': g_f1})

        # --- CONSENSUS ---
        c_dec = item['decisions']['consensus']
        c_f1 = c_dec['metrics']['f1']
        c_is_abstain = c_dec['is_abstain']
        
        consensus_full_items.append({'entropy': entropy, 'f1': c_f1})
        
        if not c_is_abstain:
            consensus_selective_items.append({'entropy': entropy, 'f1': c_f1})

    # 2. Calculate Metrics
    rows = []
    
    # Greedy
    rows.append({
        "Strategy": "Greedy",
        "Threshold": "0%",
        "Full F1": get_f1_at_rejection(greedy_full_items, rejection_rate=0.0),
        "Selective F1": get_f1_at_rejection(greedy_selective_items, rejection_rate=0.0)
    })
    rows.append({
        "Strategy": "Greedy",
        "Threshold": "50%",
        "Full F1": get_f1_at_rejection(greedy_full_items, rejection_rate=0.5),
        "Selective F1": get_f1_at_rejection(greedy_selective_items, rejection_rate=0.5)
    })

    # Consensus
    rows.append({
        "Strategy": "Consensus",
        "Threshold": "0%",
        "Full F1": get_f1_at_rejection(consensus_full_items, rejection_rate=0.0),
        "Selective F1": get_f1_at_rejection(consensus_selective_items, rejection_rate=0.0)
    })
    rows.append({
        "Strategy": "Consensus",
        "Threshold": "50%",
        "Full F1": get_f1_at_rejection(consensus_full_items, rejection_rate=0.5),
        "Selective F1": get_f1_at_rejection(consensus_selective_items, rejection_rate=0.5)
    })

    # 3. Print Table
    df = pd.DataFrame(rows)
    print("\n" + "="*40)
    print("      SEMANTIC ENTROPY REJECTION TABLE")
    print("="*40)
    print(df.to_markdown(index=False, floatfmt=".4f"))
    print("="*40 + "\n")

    # Also stats
    print(f"Total Samples: {len(results)}")
    print(f"Greedy Selective Count: {len(greedy_selective_items)}")
    print(f"Consensus Selective Count: {len(consensus_selective_items)}")

if __name__ == "__main__":
    main()
