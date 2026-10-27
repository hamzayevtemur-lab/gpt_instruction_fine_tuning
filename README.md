# InstructGPT-2: Instruction Fine-Tuning from Scratch in PyTorch

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)

An end-to-end, reproducible implementation of instruction fine-tuning for a 124M parameter GPT-2 causal language model built from first principles in pure PyTorch.

This project implements:
1. **From-Scratch GPT-2 Transformer**: Custom Multi-Head Causal Attention with `scaled_dot_product_attention`, LayerNorm, GELU, residual connections, and weight-tied LM head.
2. **HuggingFace Weight Ingestion**: Direct weight porting from OpenAI GPT-2 checkpoints with Conv1D $\rightarrow$ Linear tensor transformations.
3. **Parameter-Efficient Selective Fine-Tuning (PEFT)**: Freezing bottom blocks and token embeddings to train only **11.4% (14M)** of the parameters.
4. **Data Engineering & Dynamic Collation**: Token-boundary aligned response-only loss masking (`-100` targets) and per-batch dynamic padding.
5. **Evaluation Benchmark**: Side-by-side comparison of Pretrained Base GPT-2 vs. Fine-Tuned GPT-2 scored with Gemini 1.5 Flash LLM-as-a-Judge.

---

## Architecture Overview

```mermaid
graph TD
    InputTokens[Token IDs] --> WTE[Word Token Embedding (wte)]
    InputTokens --> WPE[Word Position Embedding (wpe)]
    WTE & WPE --> Sum[Embeddings Sum + Dropout]
    
    subgraph "12x Transformer Blocks (Layer 1-10 Frozen, 11-12 Trainable)"
        Sum --> LN1[LayerNorm 1]
        LN1 --> MHA[Multi-Head Causal Attention]
        MHA --> Add1[Residual Add]
        Add1 --> LN2[LayerNorm 2]
        LN2 --> MLP[MLP: c_fc -> GELU -> c_proj]
        MLP --> Add2[Residual Add]
    end
    
    Add2 --> LNF[Final LayerNorm (ln_f)]
    LNF --> Head[LM Head (Weight Tied with wte)]
    Head --> Logits[Logits: batch, seq_len, 50257]
```

---

## Directory Structure

```text
InstructGPT2/
├── gpt/                    # Core Model Architecture
│   ├── __init__.py         # Exports GPT2Model, GPTConfig
│   ├── config.py           # Model Hyperparameters (768 dim, 12 heads, 12 layers)
│   ├── attention.py        # Multi-Head Causal Attention
│   └── model.py            # Complete GPT2Model & Weight Tying
├── data/                   # Data directory (created by create_splits.py)
│   ├── train.json          # 9,497 training samples
│   ├── val.json            # 500 validation samples
│   └── test.json           # 500 held-out test samples
├── tests/                  # Automated Correctness Tests
│   └── test_dataset.py     # Token boundary, response masking & collate tests
├── docs/                   # Documentation & Methodology
├── results/                # Evaluation & Benchmark Outputs
├── config.py               # Training & Fine-Tuning Hyperparameters
├── create_splits.py        # Deduplicated, zero-overlap reproducible split script
├── dataset.py              # AlpacaDataset with dynamic padding custom_collate_fn
├── formatter.py            # Alpaca prompt template formatting
├── load_weights.py         # HuggingFace weight loader & Conv1D transpositions
├── train.py                # Training loop with best checkpointing & resume support
├── generate.py             # Autoregressive generation CLI (fine-tuned & baseline)
├── evaluate.py             # Greedy generation over held-out test split
├── judge.py                # Automated Gemini 1.5 Flash LLM-as-a-Judge benchmark
└── requirements.txt        # Project dependencies
```

---

## Key Engineering Features

### 1. Response-Only Loss Masking (Verified Alignment)
Autoregressive instruction models should only be penalized for predicting the **response**, not the prompt. We compute cross-entropy strictly on response tokens:
$$\mathcal{L} = -\frac{1}{N_{\text{response}}} \sum_{t \in \text{response}} \log P(x_{t+1} \mid x_{\le t})$$
All prompt and padding tokens are masked with label `-100`. Unit tests in `tests/test_dataset.py` verify that:
- Header tokens align 100% with full prompt token prefixes (preventing BPE merge errors).
- Truncation safeguards safely discard samples whose responses are cut off, eliminating empty targets.

### 2. Dynamic Batch Collation
Rather than pre-padding every sequence to a fixed `MAX_LENGTH = 256`, `custom_collate_fn` dynamically computes the maximum sequence length within each batch:
- **Average sample**: ~88 tokens
- Saves **~35-50% GPU memory & compute** compared to static padding.

### 3. Parameter-Efficient Selective Fine-Tuning
| Parameter Group | Total Params | Trainable? | Note |
| :--- | :---: | :---: | :--- |
| **Token Embeddings (`wte`)** | 38,597,376 | ❄️ Frozen | Base model already represents English vocabulary |
| **Position Embeddings (`wpe`)** | 786,432 | ❄️ Frozen | Preserves sequence positional encodings |
| **Blocks 0 to 9 (10 layers)** | 70,878,720 | ❄️ Frozen | Core linguistic representations kept intact |
| **Blocks 10 & 11 (Top 2 layers)** | 14,175,744 | 🔥 **Trainable** | Aligned for instruction following |
| **Final LayerNorm (`ln_f`)** | 1,536 | 🔥 **Trainable** | Normalized feature scale |
| **Output Head (`lm_head`)** | 0 (tied) | ❄️ Frozen | Tied to `wte` |
| **Total** | **124,439,808** | **14,177,280 (11.4%)** | **88.6% parameters frozen** |

---

## Quickstart & Installation

```bash
# Clone the repository
git clone https://github.com/hamzayevtemur-lab/gpt_instruction_fine_tuning.git
cd gpt_instruction_fine_tuning

# Install dependencies
pip install -r requirements.txt
```

### 1. Run Unit Tests
```bash
python3 -c "import tests.test_dataset as t; [f() for f in [t.test_token_boundary_alignment, t.test_response_masking_values, t.test_eos_token_present_in_labels, t.test_truncation_filters_empty_responses, t.test_dynamic_collate_fn]]; print('All tests passed!')"
```

### 2. Prepare Data Splits
```bash
python3 create_splits.py
```

### 3. Training
```bash
# Train from scratch (auto-downloads GPT-2 weights and saves best & latest checkpoints)
python3 train.py

# Resume interrupted training
python3 train.py --resume
```

### 4. Interactive Generation
```bash
# Test fine-tuned checkpoint
python3 generate.py --checkpoint checkpoints/instruct_gpt2_best.pt

# Test un-finetuned base GPT-2 baseline for comparison
python3 generate.py --use-pretrained
```

---

## Evaluation: Baseline vs. Fine-Tuned (LLM-as-a-Judge)

We benchmark both models on the identical 500-sample held-out test split (`data/test.json`) using greedy decoding (`argmax`), and score with **Gemini 1.5 Flash** on a 0–100 scale:

```bash
# 1. Generate responses from Base GPT-2 (Baseline)
python3 evaluate.py --use-pretrained --num-samples 100 --output results/eval_baseline.json

# 2. Generate responses from Fine-Tuned GPT-2
python3 evaluate.py --checkpoint checkpoints/instruct_gpt2_best.pt --num-samples 100 --output results/eval_finetuned.json

# 3. Grade responses with Gemini Judge
export GEMINI_API_KEY="your_api_key_here"
python3 judge.py
```

---

## License
MIT License. See [LICENSE](LICENSE) for details.
