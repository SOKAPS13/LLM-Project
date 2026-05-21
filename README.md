# LLM-Project : Video Fingerprinting via Encrypted Network Traffic

A GPT-based system that identifies which video a user is watching by analysing patterns in encrypted HTTPS traffic - no packet contents required.

---

## Overview

Modern video streaming is delivered over encrypted HTTPS, making content inspection impossible. However, the timing and volume patterns of traffic - how many bytes arrive in each 100ms window - leave a distinctive fingerprint for each video. This project trains a custom GPT (TrafficGPT) to learn those fingerprints and classify 100 videos with high accuracy.

### Pipeline

```
Raw .log files
      │
      ▼
[1] pre-processing.py   — Parse packets → 600-window features → K-means tokenisation
      │
      ▼
[2] pre-train.py        — Train TrafficGPT (causal LM) on token sequences
      │
      ▼
[3] fine-tuning.py      — Add classification head, two-phase fine-tune
      │
      ▼
[4] evaluate.py         — Top-1/5, confusion matrix, baseline comparison
      │
      ▼
[5] GPT-2.py            — Alternative: fine-tune HuggingFace GPT-2 on the same task
```

---

## Model Architecture — TrafficGPT

| Hyperparameter | Value |
|---|---|
| Vocab size | 260 (256 K-means + 4 special tokens) |
| d_model | 256 |
| Layers | 6 |
| Attention heads | 8 |
| Feed-forward dim | 1024 |
| Sequence length | 602 (BOS + 600 windows + EOS) |
| Dropout | 0.2 |

Defined in `model.py` and shared across pre-training and fine-tuning.

---

## Tokenisation

Each 60-second traffic capture is split into **600 × 100ms windows**. Six features are extracted per window:

- `bytes_in`, `bytes_out`
- `pkts_in`, `pkts_out`
- `avg_size_in`
- `gap_before_burst_ms`

A **K-means codebook (K=256)** maps each window to a single integer token, producing a sequence analogous to BPE tokens in NLP. Silent windows (no traffic) get a dedicated SILENT token.

---

## Training

### Pre-training (`pre-train.py`)
Causal language modelling (next-token prediction) on all 1000 traffic sequences across 100 video classes.

- Stratified split: 8 train / 1 val / 1 test per class
- Optimiser: AdamW with cosine LR decay and warmup
- 150 epochs, batch size 64

### Fine-tuning (`fine-tuning.py`)
Two-phase transfer learning:

**Phase 1 - Frozen backbone (20 epochs)**  
Only the classification head is trained (LR = 1e-3). Fast convergence.

**Phase 2 - Full fine-tune (60 epochs)**  
Backbone unfrozen with differential learning rates (head: 1e-4, backbone: 1e-5) to preserve pre-trained representations.

---

## GPT-2 Variant (`GPT-2.py`)

An alternative approach using HuggingFace `GPT2ForSequenceClassification`. Traffic windows are mapped to English words (e.g. *silence*, *trickle*, *burst*, *surge*) based on byte volume and inter-arrival time, then fed to GPT-2's existing vocabulary. Three-phase fine-tuning: frozen → top-4 blocks → full.

---

## Evaluation (`evaluate.py`)

| Metric | Value |
|---|---|
| Top-1 accuracy | reported per run |
| Top-5 accuracy | reported per run |
| Macro F1 | reported per run |
| Baselines | k-NN (k=1,3,5) and LinearSVC on token histograms |

Confusion matrix saved to `evaluation/confusion_matrix.npy`.

---

## File Structure

```
LLM-Project/
├── model.py            # TrafficGPT architecture + Config
├── pre-processing.py   # Feature extraction + K-means tokenisation
├── pre-train.py        # Causal LM pre-training
├── fine-tuning.py      # Classification head + two-phase fine-tune
├── evaluate.py         # Metrics, baselines, error analysis
├── GPT-2.py            # HuggingFace GPT-2 fine-tuning variant
└── README.md
```

---

## Requirements

```bash
pip install torch numpy pandas scikit-learn transformers
```

---

## Usage

```bash
# 1 — Preprocess raw logs
python pre-processing.py

# 2 — Pre-train TrafficGPT
python pre-train.py

# 3 — Fine-tune for classification
python fine-tuning.py

# 4 — Evaluate
python evaluate.py

# Optional — GPT-2 variant
python GPT-2.py
```

To run a smoke test before committing compute:
```bash
python pre-train.py test
python fine-tuning.py test
```

---

## Data

Place raw `.log` files under a folder organised as:
```
dataset/
├── 1/   ← video class 1
│   ├── capture_1.log
│   ├── capture_2.log
│   └── ...
├── 2/
│   └── ...
...
└── 100/
```

Each `.log` line: `timestamp_ns, direction (s/r), size_bytes`

---

## Citation / Project Context

Built as a course project on applying large language model techniques to network traffic analysis. Demonstrates that transformer architectures trained on tokenised time-series can outperform classical baselines (k-NN, SVM) on encrypted traffic fingerprinting.
