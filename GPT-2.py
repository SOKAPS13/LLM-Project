"""
True GPT-2 Fine-tuning for Video Traffic Fingerprinting
=========================================================
Uses GPT2ForSequenceClassification from Hugging Face.
Original GPT-2 embeddings and transformer weights are kept.
Traffic windows are converted to English words GPT-2 already knows.

Word mapping uses BOTH bytes_in and inter_arrival_ms (15 words):
  silence / whisper / murmur / rustle /
  trickle / drip / drops /
  stream  / flow  / current /
  flood   / surge / wave /
  burst   / peak  / spike
"""

import os, sys, json, pickle, math, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (GPT2Tokenizer, GPT2ForSequenceClassification,
                          get_cosine_schedule_with_warmup)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_DIR = "/home2/somka34/somiya/LLM/data"
CKPT_DIR = "/home2/somka34/somiya/LLM/checkpoints"


# ── Config ─────────────────────────────────────────────────────────────────────
class FTConfig:
    num_classes     : int   = 100
    max_length      : int   = 700
    label_smoothing : float = 0.1
    p1_epochs       : int   = 15
    p1_lr           : float = 1e-3
    p2_epochs       : int   = 40
    p2_lr_head      : float = 1e-4
    p2_lr_backbone  : float = 1e-5
    p2_unfreeze     : int   = 4
    p3_epochs       : int   = 60
    p3_lr_head      : float = 5e-5
    p3_lr_backbone  : float = 5e-6
    batch_size      : int   = 16
    weight_decay    : float = 0.01
    grad_clip       : float = 1.0
    warmup_steps    : int   = 50
    log_every       : int   = 5
    device          : str   = "cuda" if torch.cuda.is_available() else "cpu"

CFG = FTConfig()


# ── Rich vocabulary (bytes_in × inter_arrival → 15 English words) ─────────────
RICH_VOCAB = {
    (0, 0): "silence",
    (1, 0): "whisper",  (1, 1): "murmur",   (1, 2): "rustle",
    (2, 0): "trickle",  (2, 1): "drip",     (2, 2): "drops",
    (3, 0): "stream",   (3, 1): "flow",      (3, 2): "current",
    (4, 0): "flood",    (4, 1): "surge",     (4, 2): "wave",
    (5, 0): "burst",    (5, 1): "peak",      (5, 2): "spike",
}
BYTES_BINS  = [0, 500, 10_000, 100_000, 500_000]   # overwritten by calibrate_bins()
TIMING_BINS = [0.5, 2.0]                             # overwritten by calibrate_bins()

def bytes_inter_to_word(bytes_in: float, inter_ms: float) -> str:
    b_lvl = 0
    for i, t in enumerate(BYTES_BINS):
        if bytes_in > t: b_lvl = i + 1
    b_lvl = min(b_lvl, 5)
    if b_lvl == 0:
        return "silence"
    if inter_ms <= TIMING_BINS[0]:   s_lvl = 2
    elif inter_ms <= TIMING_BINS[1]: s_lvl = 1
    else:                            s_lvl = 0
    return RICH_VOCAB.get((b_lvl, s_lvl), "stream")

def calibrate_bins(data_dir: str, vocab_pkl: dict):
    """
    Compute data-driven BYTES_BINS and TIMING_BINS from the actual centroid
    distribution so that all 15 vocabulary words get roughly equal frequency.
    Updates the module-level BYTES_BINS and TIMING_BINS lists in-place.
    """
    from collections import Counter

    X         = np.load(f"{data_dir}/X_tokens.npy")
    km        = vocab_pkl["kmeans"]
    scaler    = vocab_pkl["scaler"]
    silent_id = vocab_pkl.get("silent", vocab_pkl["k"] - 1)

    # Collect all non-silent token IDs from positions 1..600
    tokens_flat = X[:, 1:601].flatten().astype(int)
    ns_tokens   = tokens_flat[tokens_flat != silent_id]

    # Unique token IDs weighted by occurrence (avoids re-computing per token)
    unique_toks, counts = np.unique(ns_tokens, return_counts=True)
    originals  = scaler.inverse_transform(km.cluster_centers_[unique_toks])
    bytes_vals = np.maximum(0.0, originals[:, 0])
    inter_vals = np.maximum(0.0, originals[:, 5])

    def weighted_pct(vals, weights, pcts):
        sorter     = np.argsort(vals)
        v_s, w_s   = vals[sorter], weights[sorter].astype(float)
        cdf        = np.cumsum(w_s); cdf /= cdf[-1]
        return np.interp(np.array(pcts) / 100.0, cdf, v_s).tolist()

    # bytes: keep threshold[0]=0 (zero boundary); thresholds 1-4 split the
    # non-zero distribution into 5 equal-frequency bands → levels 1–5
    nz = bytes_vals > 0
    b_thresh = (weighted_pct(bytes_vals[nz], counts[nz], [20, 40, 60, 80])
                if nz.sum() >= 4 else [100.0, 1_000.0, 10_000.0, 100_000.0])
    BYTES_BINS[:]  = [0.0] + b_thresh

    # timing: split into 3 equal-frequency bands → s_lvl 0, 1, 2
    TIMING_BINS[:] = weighted_pct(inter_vals, counts, [33, 67])

    # Report expected word distribution given new bins
    word_counts: Counter = Counter()
    for tok, cnt in zip(unique_toks, counts):
        b = max(0.0, float(originals[unique_toks == tok][0, 0]))
        t = max(0.0, float(originals[unique_toks == tok][0, 5]))
        word_counts[bytes_inter_to_word(b, t)] += int(cnt)
    total = sum(word_counts.values())
    print("  Expected word distribution after calibration:")
    for w, c in sorted(word_counts.items(), key=lambda x: -x[1]):
        print(f"    {w:12s}: {c:7d}  ({c / total * 100:5.1f}%)")
    print(f"  Unique words reachable: {len(word_counts)} / 15")


def vq_to_rich_text(token_seq: np.ndarray, km, scaler, silent_id: int) -> str:
    """
    Convert a 602-token VQ sequence to a 600-word English string.
    Uses both bytes_in (orig[0]) and inter_arrival_ms (orig[5]).
    """
    words = []
    for tok in token_seq[1:601]:
        tok = int(tok)
        if tok == silent_id:
            words.append("silence")
        else:
            centroid = km.cluster_centers_[tok]
            orig     = scaler.inverse_transform(centroid.reshape(1, -1))[0]
            bytes_in = max(0.0, float(orig[0]))
            inter_ms = max(0.0, float(orig[5]))
            words.append(bytes_inter_to_word(bytes_in, inter_ms))
    return " ".join(words)


# ── Dataset ────────────────────────────────────────────────────────────────────
class TextTrafficDataset(Dataset):
    def __init__(self, data_dir, tokenizer, vocab_pkl, max_length=700):
        X      = np.load(f"{data_dir}/X_tokens.npy")
        y      = np.load(f"{data_dir}/y_labels.npy")
        km     = vocab_pkl["kmeans"]
        scaler = vocab_pkl["scaler"]
        silent = vocab_pkl.get("silent", vocab_pkl["k"] - 1)

        self.labels = torch.tensor(y, dtype=torch.long)
        input_ids, masks = [], []
        print(f"  Tokenising {len(X)} sequences …")
        for i, seq in enumerate(X):
            text = vq_to_rich_text(seq, km, scaler, silent)
            enc  = tokenizer(text, max_length=max_length,
                             truncation=True, padding="max_length",
                             return_tensors="pt")
            input_ids.append(enc["input_ids"].squeeze(0))
            masks.append(enc["attention_mask"].squeeze(0))
            if (i+1) % 200 == 0:
                print(f"    {i+1}/{len(X)}")
        self.input_ids = torch.stack(input_ids)
        self.masks     = torch.stack(masks)
        print(f"  Done. Shape: {self.input_ids.shape}")

    def __len__(self):        return len(self.labels)
    def __getitem__(self, i): return self.input_ids[i], self.masks[i], self.labels[i]


# ── Freeze / unfreeze helpers ──────────────────────────────────────────────────
def freeze_backbone(model):
    for name, p in model.named_parameters():
        if "score" not in name:
            p.requires_grad = False

def unfreeze_top_layers(model, n=4):
    for block in model.transformer.h[-n:]:
        for p in block.parameters():
            p.requires_grad = True

def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True

def n_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Evaluation ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, label_smoothing=0.0):
    model.eval()
    logits_all, labels_all = [], []
    for ids, mask, y in loader:
        out = model(input_ids=ids.to(device), attention_mask=mask.to(device))
        logits_all.append(out.logits.cpu())
        labels_all.append(y)
    model.train()
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    loss   = nn.CrossEntropyLoss(label_smoothing=label_smoothing)(logits, labels).item()
    top1   = (logits.argmax(1) == labels).float().mean().item() * 100
    top5   = (logits.topk(5,1).indices == labels.unsqueeze(1)).any(1).float().mean().item() * 100
    return {"loss": loss, "top1": top1, "top5": top5,
            "logits": logits, "labels": labels}


# ── Training phase ─────────────────────────────────────────────────────────────
def run_phase(model, train_loader, val_loader, device,
              n_epochs, lr_head, lr_backbone, cfg, tag):

    head_params     = [p for n,p in model.named_parameters()
                       if p.requires_grad and "score" in n]
    backbone_params = [p for n,p in model.named_parameters()
                       if p.requires_grad and "score" not in n]

    param_groups = [{"params": head_params, "lr": lr_head}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})

    optimizer   = torch.optim.AdamW(param_groups,
                                    weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    total_steps = n_epochs * len(train_loader)
    scheduler   = get_cosine_schedule_with_warmup(
                      optimizer, cfg.warmup_steps, total_steps)
    criterion   = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    history     = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    best_val, best_state = 0.0, None

    print(f"\n── {tag} ──")
    print(f"   Trainable : {n_trainable(model):,}")
    if backbone_params:
        print(f"   LR  head={lr_head:.0e}  backbone={lr_backbone:.0e}")
    else:
        print(f"   LR  head={lr_head:.0e}  (backbone frozen)")

    for epoch in range(1, n_epochs+1):
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        t0 = time.time()
        for ids, mask, y in train_loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step(); scheduler.step()
            ep_loss    += loss.item()
            ep_correct += (logits.argmax(1)==y).sum().item()
            ep_total   += len(y)

        train_loss = ep_loss / len(train_loader)
        train_acc  = ep_correct / ep_total * 100
        val        = evaluate(model, val_loader, device, cfg.label_smoothing)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["top1"])

        if val["top1"] > best_val:
            best_val   = val["top1"]
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}

        if epoch % cfg.log_every == 0 or epoch == 1:
            print(f"   Epoch {epoch:3d}/{n_epochs} | "
                  f"train loss {train_loss:.4f} acc {train_acc:5.1f}% | "
                  f"val loss {val['loss']:.4f} acc {val['top1']:5.1f}% "
                  f"top5 {val['top5']:5.1f}% | {time.time()-t0:.1f}s")

    print(f"   Best val acc: {best_val:.1f}%")
    return best_state, history, best_val


# ── Main ───────────────────────────────────────────────────────────────────────
def main(cfg=CFG):
    os.makedirs(CKPT_DIR, exist_ok=True)
    device = torch.device(cfg.device)
    print(f"Device : {device}\n")

    with open(f"{DATA_DIR}/vocab.pkl", "rb") as f:
        vocab_pkl = pickle.load(f)
    silent = vocab_pkl.get("silent", vocab_pkl["k"] - 1)

    print("Calibrating vocabulary bins …")
    calibrate_bins(DATA_DIR, vocab_pkl)
    print(f"  BYTES_BINS  : {[round(v, 2) for v in BYTES_BINS]}")
    print(f"  TIMING_BINS : {[round(v, 4) for v in TIMING_BINS]}\n")

    print("Loading GPT-2 tokenizer …")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Show example conversion (now with calibrated bins)
    sample  = np.load(f"{DATA_DIR}/X_tokens.npy")[0]
    text_ex = vq_to_rich_text(sample, vocab_pkl["kmeans"],
                               vocab_pkl["scaler"], silent)
    enc_len = len(tokenizer(text_ex)["input_ids"])
    print(f"Example (first 20 words): {' '.join(text_ex.split()[:20])} …")
    print(f"BPE token count: {enc_len}  (GPT-2 limit=1024)  ✓")
    print(f"Unique words used: {len(set(text_ex.split()))}\n")

    print("Building dataset …")
    full_ds = TextTrafficDataset(DATA_DIR, tokenizer, vocab_pkl, cfg.max_length)

    train_idx = np.load(f"{CKPT_DIR}/split_train_idx.npy")
    val_idx   = np.load(f"{CKPT_DIR}/split_val_idx.npy")
    test_idx  = np.load(f"{CKPT_DIR}/split_test_idx.npy")

    train_loader = DataLoader(Subset(full_ds, train_idx),
                              batch_size=cfg.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(Subset(full_ds, val_idx),
                              batch_size=cfg.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(Subset(full_ds, test_idx),
                              batch_size=cfg.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"\nSplit: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

    print("\nLoading GPT2ForSequenceClassification …")
    model = GPT2ForSequenceClassification.from_pretrained(
                "gpt2", num_labels=cfg.num_classes)
    model.config.pad_token_id = tokenizer.eos_token_id
    model = model.to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total/1e6:.1f}M  (original GPT-2 embeddings KEPT)\n")

    all_history = {}

    # Phase 1: frozen GPT-2
    freeze_backbone(model)
    p1_state, p1_hist, _ = run_phase(
        model, train_loader, val_loader, device,
        cfg.p1_epochs, cfg.p1_lr, 0.0, cfg,
        "Phase 1 — frozen GPT-2, classification head only")
    all_history["phase1"] = p1_hist
    model.load_state_dict(p1_state)

    # Phase 2: top 4 blocks
    unfreeze_top_layers(model, cfg.p2_unfreeze)
    p2_state, p2_hist, _ = run_phase(
        model, train_loader, val_loader, device,
        cfg.p2_epochs, cfg.p2_lr_head, cfg.p2_lr_backbone, cfg,
        f"Phase 2 — top {cfg.p2_unfreeze} transformer blocks + head")
    all_history["phase2"] = p2_hist
    model.load_state_dict(p2_state)

    # Phase 3: full fine-tune
    unfreeze_all(model)
    p3_state, p3_hist, _ = run_phase(
        model, train_loader, val_loader, device,
        cfg.p3_epochs, cfg.p3_lr_head, cfg.p3_lr_backbone, cfg,
        "Phase 3 — full fine-tune")
    all_history["phase3"] = p3_hist
    model.load_state_dict(p3_state)

    # Test evaluation
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION  (true GPT-2 fine-tuning)")
    print("=" * 60)
    test   = evaluate(model, test_loader, device)
    preds  = test["logits"].argmax(1).numpy()
    labels = test["labels"].numpy()

    print(f"Top-1 accuracy : {test['top1']:.2f}%")
    print(f"Top-5 accuracy : {test['top5']:.2f}%")
    print(f"Test loss      : {test['loss']:.4f}")
    print(f"Classes correct: {(preds==labels).sum()} / {cfg.num_classes}")

    model.save_pretrained(f"{CKPT_DIR}/gpt2_traffic")
    tokenizer.save_pretrained(f"{CKPT_DIR}/gpt2_traffic")
    with open(f"{CKPT_DIR}/gpt2_history.json", "w") as f:
        json.dump(all_history, f, indent=2)
    print(f"\nSaved → {CKPT_DIR}/gpt2_traffic/")
    return model, test


if __name__ == "__main__":
    main()