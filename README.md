# Knowing When You Don't Know
### Separating Access Failures from Knowledge Gaps for Closed-Book Hallucination Mitigation

Final project for COMS 4705 (Natural Language Processing), Columbia University, Fall 2025.
**Farhaan Siddiqui, Ahmed Jaber, Josh Hegstad.**

Paper: [`paper/report.pdf`](paper/report.pdf) (source in [`paper/main.tex`](paper/main.tex)).

## Summary

Small language models answering closed-book questions hallucinate confidently because they can't tell
facts they have from facts they don't. We split that failure into two causes and attack each:

- **Access failures** (the fact is in the weights but isn't retrieved): Chain-of-Thought distillation
  from a 235B teacher, with rationales phrased as recall from the model's own parameters rather than
  citations of a passage. **+5.4 EM** on closed-book HotPotQA (18.9 → 24.3), reaching ~92% of
  PaLM-62B's closed-book EM with a 7B model.
- **Knowledge gaps** (the fact isn't there): map the model's *knowledge boundary* by running the
  distilled model closed-book over 40k training questions, relabel the ones it gets wrong as
  "I don't know", and fine-tune on an 80/20 answer/abstain mix. **97.5% abstention precision** at a
  46% abstention rate; the fraction of questions answered incorrectly falls from **81% → 33%** with
  overall EM unchanged. A verifier-reward RL stage (RLVF) then recovers 5 points of coverage at
  97.8% precision.
- **Uncertainty estimation**: after fine-tuning, RAUQ and Semantic Entropy separate correct from
  incorrect answers at AUROC 0.86, vs. chance on the base model.

All experiments use **Qwen2.5-7B-Instruct** with LoRA/QLoRA adapters on the **HotPotQA** distractor
split, evaluated closed-book on the 7,405-question validation set.

## Who did what

| Area | Owner | Code |
|---|---|---|
| Teacher data generation (Qwen3-235B via Tinker), CoT distillation, abstention dataset + SFT, baseline SFT, vLLM/RAUQ evaluation pipeline, analysis | Farhaan Siddiqui | `finetune-cot/`, `finetune-cot-closed/`, `finetune_abstention/`, `baseline_*/`, `teacher_*/`, `results/`, `analysis/`, `run_evals.sh` |
| Reinforcement Learning from Verifier Feedback (RLVF), abstention classifier | Ahmed Jaber | `reward_learning/` |
| Semantic Entropy, Qwen3-8B re-implementation at 80k scale on Tinker, SE rejection curves | Josh Hegstad | `tinker/`, `SE/`, `se_*.png`, `zJosh Analysis/`, `zGENERAL SCRIPTS/` |

## Pipeline

```
HotPotQA (distractor, train)
        │
        ├─ finetune-cot/teacher_hotpot_tqdm_full.py      Qwen3-235B teacher writes "internalized-knowledge"
        │                                                 rationales given the supporting paragraphs → Dataset B
        │                                                 (teacher_closedbook/ = same teacher with no context;
        │                                                  teacher_openbook/ = citation-style variant, unused in paper)
        │
        ├─ baseline_finetune/qwen-finetune.py            Qwen2.5-SFT   (10k Q→A pairs, Dataset A1)
        ├─ finetune-cot/qwen-finetune-cot.py             Qwen2.5-FCoT  (Dataset B, LoRA)
        │        │
        │        └─ finetune-cot/eval_train.py           FCoT run closed-book over 40k train questions
        │             └─ finetune-cot/abstention_data.py relabel wrong answers → "I don't know"
        │                  └─ finetune-cot/balance_data.py  80/20 answer/abstain mix → Dataset C
        │
        ├─ finetune_abstention/qwen-abstain2.py          Qwen2.5-Abstain (Dataset C, QLoRA)
        │        └─ reward_learning/distractor_rlhf.py   Qwen2.5-RLVF (init from Abstain, verifier reward)
        │
        └─ evaluation (closed-book, greedy, HotPotQA validation)
             ├─ */eval_vllm*.py                          EM / F1 / abstention rate / selective EM
             ├─ */eval_rauq.py, run_evals.sh             RAUQ uncertainty + rejection curves
             ├─ tinker/eval_semantic_entropy*.py         Semantic Entropy (M=10 samples, NLI clustering)
             └─ analysis/                                precision/recall vs. reference models, confusion
                                                          matrices, question-type breakdowns, notebook
```

## Repository layout

| Path | Contents |
|---|---|
| `paper/` | Final report (PDF + LaTeX source) |
| `baseline_vanilla/` | Qwen2.5-7B-Instruct out of the box: eval scripts + results |
| `baseline_finetune/` | Plain SFT baseline: training, eval, results |
| `finetune-cot/` | CoT distillation (teacher generation → HF dataset → LoRA SFT → eval); also the abstention-dataset construction scripts |
| `finetune-cot-closed/` | Same recipe trained on closed-book teacher traces (no context given to the teacher) |
| `finetune_abstention/` | Abstention SFT (`qwen-abstain2.py`), eval, and per-run results for the 90/10 and 80/20 mixes in CoT and short-answer target formats |
| `teacher_closedbook/`, `teacher_openbook/` | Teacher-generation scripts for the closed-book and citation-style variants (data is on the Hub, see below) |
| `reward_learning/` | RLVF: reward/loss definitions, training script, abstention classifier, logs and metrics |
| `tinker/` | Josh's Tinker-based pipeline (Qwen3-8B distillation + abstention at 80k scale) and all Semantic Entropy evaluation |
| `results/` | Consolidated per-example results and metrics for the main-table models |
| `analysis/` | `analysis.ipynb`, `comprehensive_analysis.py`, `breakdown_metrics.py`; abstention precision/recall CSVs, confusion matrices, type/level breakdowns, SFT-vs-CoT win/loss lists |
| `run_evals.sh` | Runs the RAUQ evaluation for base / abstention / CoT models |

Each training folder also has `save_model.py` (merge adapter and push to the Hub) and a `scrap/` subfolder
with earlier runs kept for reference.

## Models and datasets on the Hugging Face Hub

Datasets (all derived from HotPotQA distractor/train):

- `fsiddiqui2/hotpot-qa-cot-reasoning` — Dataset B, 10k internalized-knowledge rationales (teacher given supporting paragraphs)
- `fsiddiqui2/hotpot-qa-closedbook-cot-reasoning` — closed-book teacher rationales (no context)
- `fsiddiqui2/hotpotqa-abstention-90-10`, `fsiddiqui2/hotpotqa-abstention-70-30` — abstention mixes; the 80/20 mix used in the paper is Dataset C

Models (Qwen2.5-7B-Instruct + merged adapter):

- `fsiddiqui2/Qwen2.5-7B-Instruct-HotpotQA-Finetuned-Distractor-10000` — Qwen2.5-SFT
- `fsiddiqui2/Qwen2.5-7B-Instruct-HotpotQA-CoT-Finetuned-10000` — Qwen2.5-FCoT
- `fsiddiqui2/Qwen2.5-7B-Instruct-HotpotQA-Abstention-10000-80-20` — Qwen2.5-Abstain

## Reproducing

```bash
pip install -r requirements.txt          # torch, transformers, peft, bitsandbytes, datasets, accelerate
pip install vllm                         # for the *_vllm eval scripts
export TINKER_API_KEY=...                # only for teacher generation and tinker/

# 1. Teacher rationales → Hub            finetune-cot/teacher_hotpot_tqdm_full.py, then upload_data_to_hf.py
# 2. CoT distillation                     finetune-cot/qwen-finetune-cot.py  → save_model.py
# 3. Knowledge boundary + Dataset C       finetune-cot/eval_train.py → abstention_data.py → balance_data.py
# 4. Abstention SFT                       finetune_abstention/qwen-abstain2.py (TRAINING_MODE="SHORT")
# 5. RLVF                                 reward_learning/run_experiments.sh
# 6. Evaluate                             */eval_vllm*.py, run_evals.sh, tinker/eval_semantic_entropy_abstention.py
```

Training was done on single A100/H100-class GPUs (QLoRA fits in ~24 GB). Scripts take their
model/dataset IDs from constants at the top of each file.

## Main result (HotPotQA validation, closed-book, N = 7,405)

| Model | EM | F1 | Abstain rate | Selective EM | Abstention precision* |
|---|---|---|---|---|---|
| Qwen2.5-Instruct | 18.86 | 27.02 | – | – | – |
| Qwen2.5-SFT | 21.94 | 31.41 | – | – | – |
| Qwen2.5-FCoT | **24.31** | **33.87** | – | – | – |
| Qwen2.5-Abstain | 19.47 | 25.78 | 46.3% | 37.98 | 97.49% |
| Qwen2.5-RLVF | 20.37 | 26.24 | 41.2% | 34.62 | **97.84%** |

\*fraction of abstentions on questions the base model answers incorrectly. Full tables, rejection
curves and selective-prediction metrics (AUROC, AURC, ECE) are in the paper.
