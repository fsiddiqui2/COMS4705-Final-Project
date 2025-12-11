
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

NLI_MODEL = "microsoft/deberta-large-mnli"

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {NLI_MODEL} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device)

    text1 = "Annie Morton"
    text2 = "I don't know"

    # Forward
    premise = text1
    hypothesis = text2
    input_text = f"{premise} [SEP] {hypothesis}"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        # MNLI: 0=Contradiction, 1=Neutral, 2=Entailment (Standard mapping for Deberta?)
        # Actually checking mapping... Deberta usually: 0: Contradiction, 1: Neutral, 2: Entailment
        entail_prob = probs[0][2].item()
    
    print(f"'{text1}' -> '{text2}': {entail_prob:.4f}")

    # Backward
    premise = text2
    hypothesis = text1
    input_text = f"{premise} [SEP] {hypothesis}"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        entail_prob_back = probs[0][2].item()

    print(f"'{text2}' -> '{text1}': {entail_prob_back:.4f}")

    if entail_prob > 0.5 and entail_prob_back > 0.5:
        print("BUG REPRODUCED: Model considers them equivalent!")
    else:
        print("Behaved correctly (No equivalence).")

if __name__ == "__main__":
    main()
