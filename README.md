# StanceEval-2026 Track 1: Unseen-Target Arabic Stance Detection

Code for the ArabicNLP 2026 paper *rabia_naz at StanceEval-2026: An Ensemble of
Fine-Tuned Open LLMs for Unseen-Target Arabic Stance Detection*.

The system scored 0.7564 Favg2 on StanceEval-2026 Track 1, where the test target
does not appear in training. The prediction is a vote over three QLoRA fine-tuned
models, Qwen2.5-7B, ALLaM-7B, and a sarcasm-aware Qwen2.5-7B, adapted to the test
target with pseudo-labels accepted only where independent models agree.

## Scripts

| File | Purpose |
|:---|:---|
| `01_qwen_selftrained.py` | Trains the self-trained Qwen2.5-7B |
| `02_allam.py` | Trains ALLaM-7B |
| `03_sarcasm_and_vote.py` | Trains the sarcasm-aware model, builds the final vote and submission file |

Each script runs on a single Kaggle T4 in about three hours and installs its own
dependencies.

## Configuration

| Parameter | Value |
|:---|:---|
| Adapter | LoRA, r=16, alpha=32 |
| Quantization | 4-bit NF4 |
| Learning rate | 2e-4 |
| Effective batch size | 16 |
| Seed | 42 |
| Base models | `Qwen/Qwen2.5-7B-Instruct`, `ALLaM-AI/ALLaM-7B-Instruct-preview` |
