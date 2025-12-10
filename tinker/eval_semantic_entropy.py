import os
import tinker
from tinker import types as ttypes
from dotenv import load_dotenv
import json
import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

load_dotenv()

# ==========================================
# 1. Configuration
# ==========================================

# Model to Evaluate
ADAPTER_PATH = "tinker://0c59c24e-9098-51ca-b421-192c15a5a1f3:train:0/sampler_weights/qwen-hotpot-adapter"
BASE_MODEL = "Qwen/Qwen3-8B"

# Data
DATASET_NAME = "hotpot_qa"
SUBSET_NAME = "distractor"
SPLIT = "validation"
NUM_SAMPLES_PER_QUESTION = 10  # Number of generations to estimate entropy
TOTAL_EVAL_SAMPLES = 100       # Limit total questions for speed (set to None for full)

# NLI Model (Local)
# User requested larger model for better accuracy (requires GPU)
NLI_MODEL_NAME = "microsoft/deberta-large-mnli" 

OUTPUT_FILE = "tinker_sem_entropy_results.json"
METRICS_FILE = "tinker_sem_entropy_metrics.json"

import re
import string
from collections import Counter

# ==========================================
# 2. Metric Utils (Copied from tinker_eval.py)
# ==========================================

def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0.0, 0.0, 0.0
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0.0, 0.0, 0.0
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0.0, 0.0, 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

# ==========================================
# 3. NLI Clustering Logic
# ==========================================

class SemanticClusterer:
    def __init__(self, model_name, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Loading NLI model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Dynamic Label Mapping
        # Different models have different mappings (e.g., Entailment can be 1 or 2)
        self.entailment_idx = -1
        
        # Try finding "entailment" (case-insensitive) in id2label
        if hasattr(self.model.config, "id2label"):
            for idx, label in self.model.config.id2label.items():
                if "entailment" in str(label).lower():
                    self.entailment_idx = int(idx)
                    break
        
        # Fallback/Default if not found (Warn user)
        if self.entailment_idx == -1:
            print("⚠️ WARNING: Could not find 'entailment' label in model config. Defaulting to index 2 (common for MNLI).")
            print(f"Labels found: {self.model.config.id2label}")
            self.entailment_idx = 2 
        else:
            print(f"✅ Found Entailment label at index: {self.entailment_idx} ({self.model.config.id2label[self.entailment_idx]})")
    def predict_entailment(self, premise, hypothesis):
        inputs = self.tokenizer(
            premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)
            
        # Check if Entailment prob is highest or greater than threshold?
        # Standard approach: argmax == entailment_idx
        prediction = torch.argmax(logits, dim=-1).item()
        return prediction == self.entailment_idx

    def are_equivalent(self, text1, text2):
        if text1 == text2:
            return True
        # Bidirectional entailment
        return self.predict_entailment(text1, text2) and self.predict_entailment(text2, text1)

    def cluster_answers(self, answers_with_probs):
        """
        answers_with_probs: list of tuples (answer_text, probability)
        Returns: list of clusters, where each cluster is a dict with {'answers': [], 'prob_sum': float}
        """
        # Sort by probability descending to pick "canonical" representative for efficiency if needed
        # but for clustering we just iterate.
        
        clusters = [] # List of {'text': representative_text, 'prob': total_prob, 'members': [texts]}
        
        for ans_text, ans_prob in answers_with_probs:
            matched = False
            for cluster in clusters:
                # Check equivalence with the cluster representative
                if self.are_equivalent(ans_text, cluster['text']):
                    cluster['prob'] += ans_prob
                    cluster['members'].append(ans_text)
                    matched = True
                    break
            
            if not matched:
                clusters.append({
                    'text': ans_text,
                    'prob': ans_prob,
                    'members': [ans_text]
                })
                
        return clusters

# ==========================================
# 4. Tinker Generation & Metrics
# ==========================================

def calculate_sequence_prob(token_ids, logprobs):
    """
    Estimate P(sequence) from token logprobs.
    Tinker returns logprobs for each token.
    P(seq) = exp(sum(logprobs))
    """
    # Sum of log probabilities = log(Product of probabilities)
    # We want P(seq), so we exponentiate.
    # Note: Using lengthy sequences might result in underflow if we multiplied probs,
    # but summing logprobs is stable.
    total_logprob = sum(logprobs)
    return np.exp(total_logprob)

def main():
    print("🚀 Connecting to Tinker for Semantic Entropy Eval...")
    if not os.environ.get("TINKER_API_KEY"):
        print("⚠️ TINKER_API_KEY not found.")
        return

    service = tinker.ServiceClient()
    sampling_client = service.create_sampling_client(model_path=ADAPTER_PATH)
    
    # Get tokenizer for decoding
    training_client = service.create_lora_training_client(base_model=BASE_MODEL, rank=16)
    tokenizer = training_client.get_tokenizer()

    print("init nli...")
    clusterer = SemanticClusterer(NLI_MODEL_NAME)

    print(f"\nLoading Dataset ({DATASET_NAME})...")
    dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split=SPLIT)
    if TOTAL_EVAL_SAMPLES:
        dataset = dataset.select(range(min(len(dataset), TOTAL_EVAL_SAMPLES)))
    
    results = []
    
    print(f"Starting eval loop for {len(dataset)} questions...")
    
    for i, item in tqdm(enumerate(dataset), total=len(dataset)):
        question = item['question']
        gold_answers = item['answer']
        if not isinstance(gold_answers, list):
             gold_answers = [gold_answers]

        # 1. Prepare Prompt
        messages = [
            {"role": "system", "content": "You are a concise encyclopedia. Answer the question directly with a short phrase or entity name. Do not explain."},
            {"role": "user", "content": question}
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        model_input = ttypes.ModelInput.from_ints(input_ids)
        
        # 2. Generate N Samples
        sampling_params = ttypes.SamplingParams(
            max_tokens=32, 
            temperature=0.7, # High temp for diversity
            top_p=0.9
        )
        
        try:
            # We request N samples in one call (Tinker supports this efficiencly)
            result = sampling_client.sample(model_input, NUM_SAMPLES_PER_QUESTION, sampling_params).result()
            
            samples = []
            
            for seq in result.sequences:
                # Decode Text
                text = tokenizer.decode(seq.tokens)
                # Cleanup
                if "assistant\n" in text: text = text.split("assistant\n")[-1]
                text = text.replace("<|im_end|>", "").strip()
                # Basic normalization
                text = text.split("\n")[0].strip()
                if "." in text: text = text.split(".")[0].strip()
                
                # Get Probability
                # seq.logprobs is List[float] (if requested? Tinker usually returns it by default in TokensWithLogprobs?)
                # Wait, check Tinker types. TokensWithLogprobs usually has `logprobs` or `maybe_logprobs`.
                # Assuming `seq.logprobs` exists and is populated.
                # If not available, we assume uniform distribution (1/N) as fallback, but that defeats purpose.
                # Let's verify `seq` has logprobs.
                
                # In tinker/types.py or docs: TokensWithLogprobs has `tokens: List[int]` and `logprobs: Optional[List[float]]`
                if hasattr(seq, 'logprobs') and seq.logprobs:
                    prob = calculate_sequence_prob(seq.tokens, seq.logprobs)
                else:
                    # Fallback to uniform if API doesn't return logprobs (should confirm)
                    prob = 1.0 / NUM_SAMPLES_PER_QUESTION
                
                samples.append((text, prob))
                
            # Normalize probabilities to sum to 1 (across the N samples)
            total_prob = sum(p for t, p in samples)
            if total_prob > 0:
                samples = [(t, p/total_prob) for t, p in samples]
            
            # 3. Cluster Semantically
            clusters = clusterer.cluster_answers(samples)
            
            # 4. Calculate Entropy
            # SE = - sum( P(C) * log(P(C)) )
            entropy = 0.0
            
            # Best Cluster Logic for Accuracy
            best_cluster = None
            max_cluster_prob = -1.0
            
            for c in clusters:
                p_c = c['prob']
                
                if p_c > max_cluster_prob:
                    max_cluster_prob = p_c
                    best_cluster = c
                
                if p_c > 0:
                    entropy -= p_c * np.log(p_c)

            # 5. Calculate Accuracy (EM/F1) on Best Cluster
            # Use the representative text from the best cluster
            best_answer = best_cluster['text'] if best_cluster else ""
            
            sample_em = max([float(exact_match_score(best_answer, gold)) for gold in gold_answers])
            sample_f1 = max([f1_score(best_answer, gold)[0] for gold in gold_answers])
                    
            # Store result
            results.append({
                "question": question,
                "gold_answers": gold_answers,
                "prediction": best_answer,
                "entropy": entropy,
                "metrics": {
                    "exact_match": sample_em,
                    "f1_score": sample_f1
                },
                "num_clusters": len(clusters),
                "clusters": [{"text": c['text'], "prob": c['prob']} for c in clusters],
                "samples": [s[0] for s in samples]
            })
            
            # print(f"Q: {question[:30]}... | SE: {entropy:.4f} | Clusters: {len(clusters)}")

        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue

    # Stats
    avg_se = np.mean([r['entropy'] for r in results]) if results else 0
    avg_em = np.mean([r['metrics']['exact_match'] for r in results]) * 100 if results else 0
    avg_f1 = np.mean([r['metrics']['f1_score'] for r in results]) * 100 if results else 0
    
    print(f"\nResults ({len(results)} samples):")
    print(f"  Avg Semantic Entropy: {avg_se:.4f}")
    print(f"  Avg Exact Match:      {avg_em:.2f}%")
    print(f"  Avg F1 Score:         {avg_f1:.2f}%")
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
