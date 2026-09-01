"""
GPT-2 Fine-tuning for Video Traffic Fingerprinting
====================================================
Loads pre-prepared X_tokens.npy / y_labels.npy (from prepare_data.py).
Integer token IDs (0-17) are mapped to English words that GPT-2 already
knows, then BPE-tokenised and fed to GPT2ForSequenceClassification (117M).

Pipeline:
  prepare_data.py  (already done)
  → GPT-2.py                     ← you are here
"""

import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import (
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    get_cosine_schedule_with_warmup,
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class FTConfig:
    data_dir: str = "./data"
    ckpt_dir: str = "./checkpoints"
    num_classes: int = 100
    # Allow slightly richer token strings without truncating as aggressively.
    max_length: int = 800
    label_smoothing: float = 0.1
    subset_fraction: float = 1.0   # <1.0 → quick-test mode

    # Phase 1: frozen GPT-2, classification head only
    p1_epochs: int = 5
    p1_lr: float = 1e-3

    # Phase 2: unfreeze top-N transformer blocks + head
    p2_epochs: int = 25
    p2_lr_head: float = 1e-4
    p2_lr_backbone: float = 1e-5
    p2_unfreeze: int = 4

    # Phase 3: full fine-tune
    p3_epochs: int = 150
    p3_lr_head: float = 5e-5
    p3_lr_backbone: float = 5e-6

    batch_size: int = 16
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 50
    log_every_epochs: int = 5
    device: str = ("cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available()
                   else "cpu")


CFG = FTConfig()

RICH_VOCAB = {
    (0, 0): "silence",
    (1, 0): "whisper", (1, 1): "murmur", (1, 2): "rustle",
    (2, 0): "trickle", (2, 1): "drip", (2, 2): "drops",
    (3, 0): "stream", (3, 1): "flow", (3, 2): "current",
    (4, 0): "flood", (4, 1): "surge", (4, 2): "wave",
    (5, 0): "burst", (5, 1): "peak", (5, 2): "spike",
}
BYTES_BINS = [0.0, 500.0, 10_000.0, 100_000.0, 500_000.0]
TIMING_BINS = [0.5, 2.0]


def bytes_inter_to_word(bytes_in: float, inter_ms: float) -> str:
    b_lvl = 0
    for i, t in enumerate(BYTES_BINS):
        if bytes_in > t:
            b_lvl = i + 1
    b_lvl = min(b_lvl, 5)
    if b_lvl == 0:
        return "silence"
    if inter_ms <= TIMING_BINS[0]:
        s_lvl = 2
    elif inter_ms <= TIMING_BINS[1]:
        s_lvl = 1
    else:
        s_lvl = 0
    return RICH_VOCAB.get((b_lvl, s_lvl), "stream")


def calibrate_bins(data_dir: str, vocab_pkl: dict):
    X = np.load(f"{data_dir}/X_tokens.npy")
    tokens_flat = X[:, 1:601].flatten().astype(int)
    special = vocab_pkl.get("special", {}) if isinstance(vocab_pkl, dict) else {}
    special_ids = {int(v) for v in special.values()} if isinstance(special, dict) else set()
    ns_tokens = tokens_flat[~np.isin(tokens_flat, list(special_ids))] if special_ids else tokens_flat
    if len(ns_tokens) == 0:
        print("  No non-special tokens found; skipping vocabulary summary.")
        return

    unique_toks, counts = np.unique(ns_tokens, return_counts=True)

    word_counts: Counter = Counter()
    id2word_mapping = normalize_id2word(vocab_pkl)
    for tok, cnt in zip(unique_toks, counts):
        word_counts[id2word_mapping.get(int(tok), f"token_{int(tok):03d}")] += int(cnt)
    total = sum(word_counts.values())
    print("  Token-word distribution:")
    for w, c in sorted(word_counts.items(), key=lambda x: -x[1]):
        print(f"    {w:12s}: {c:7d}  ({c / total * 100:5.1f}%)")
    print(f"  Unique words reachable: {len(word_counts)} / {len(id2word_mapping)}")


def vq_to_rich_text(token_seq: np.ndarray, km, scaler, silent_id: int) -> str:
    words = []
    for tok in token_seq[1:601]:
        tok = int(tok)
        if tok == silent_id:
            words.append("silence")
        else:
            if km is not None and scaler is not None:
                centroid = km.cluster_centers_[tok]
                orig = scaler.inverse_transform(centroid.reshape(1, -1))[0]
                bytes_in = max(0.0, float(orig[0]))
                inter_ms = max(0.0, float(orig[5]))
            else:
                bytes_in = float(tok)
                inter_ms = float(tok)
            words.append(bytes_inter_to_word(bytes_in, inter_ms))
    return " ".join(words)


def build_default_id2word(num_tokens=512):
    """Fallback mapping for vocab IDs when no vocabulary file is available."""
    return {i: f"token_{i:03d}" for i in range(num_tokens)}


def normalize_id2word(vocab, fallback_num_tokens=512):
    """
    Normalize the vocabulary mapping into a dict of token_id -> text token.

    Prefers the new tokenizer's id2word/id2token outputs, but also supports
    older hard-coded mappings and adds special tokens if present.
    """
    if vocab is None:
        return build_default_id2word(fallback_num_tokens)

    if "id2word" in vocab and isinstance(vocab["id2word"], dict):
        mapping = {int(k): str(v) for k, v in vocab["id2word"].items()}
    elif "id2token" in vocab and isinstance(vocab["id2token"], dict):
        mapping = {int(k): str(v) for k, v in vocab["id2token"].items()}
    else:
        mapping = {}

    special = vocab.get("special", {})
    if isinstance(special, dict):
        for name, code in special.items():
            mapping.setdefault(int(code), name.lower())

    if not mapping:
        return build_default_id2word(fallback_num_tokens)

    return mapping


def token_to_text(token_id, id2word):
    """Convert a token id to a text token that GPT-2 can process."""
    token_id = int(token_id)
    word = id2word.get(token_id)
    if word is None:
        return f"token_{token_id:03d}"
    return str(word).replace(" ", "_")


class NumpyTrafficDataset(Dataset):
    """
    Converts integer token sequences (X_tokens.npy rows) to GPT-2 BPE tensors.
    Uses the id2word mapping from the tokenizer vocabulary.
    """
    def __init__(self, X, y, tokenizer, max_length, vocab=None):
        self.labels = torch.tensor(y, dtype=torch.long)
        input_ids_all, mask_all = [], []
        n = len(X)
        print(f"  BPE-tokenising {n:,} sequences ...")
        t0 = time.time()
        if vocab is None:
            raise ValueError("vocab must be provided for the word mapping")

        # Use id2word directly from vocab — works for both k-means and DT tokenizers
        id2word = normalize_id2word(vocab)
        pad_id = vocab.get("special", {}).get("PAD", -1)

        for i, seq in enumerate(X):
            # Convert integer token IDs to English words using id2word mapping
            words = []
            for tok in seq:
                tok = int(tok)
                if tok == pad_id:
                    break  # Stop at padding
                word = id2word.get(tok, f"tok{tok:03d}")
                words.append(str(word))
            text = " ".join(words)
            enc = tokenizer(text, max_length=max_length, truncation=True,
                            padding="max_length", return_tensors="pt")
            input_ids_all.append(enc["input_ids"].squeeze(0))
            mask_all.append(enc["attention_mask"].squeeze(0))
            if (i + 1) % 20_000 == 0:
                print(f"    {i+1:,}/{n:,}  ({time.time()-t0:.0f}s)")
        self.input_ids = torch.stack(input_ids_all)
        self.masks = torch.stack(mask_all)
        print(f"  Done in {time.time()-t0:.0f}s.  Shape: {self.input_ids.shape}")

    def __len__(self):        return len(self.labels)
    def __getitem__(self, i): return self.input_ids[i], self.masks[i], self.labels[i]








def _removed_stream_json_array(path, chunk_size=1024 * 1024):  # kept for reference, not used
    """
    Stream objects from a top-level JSON array without loading all data in memory.
    Expects file format: [ {...}, {...}, ... ]
    """
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buffer = ""
        pos = 0
        started = False

        while True:
            if pos >= len(buffer):
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                buffer = ""
                pos = 0
                buffer += chunk

            if not started:
                while True:
                    while pos < len(buffer) and buffer[pos].isspace():
                        pos += 1
                    if pos < len(buffer):
                        break
                    chunk = f.read(chunk_size)
                    if not chunk:
                        raise ValueError("Unexpected EOF before JSON array start")
                    buffer += chunk
                if buffer[pos] != "[":
                    raise ValueError("Expected top-level JSON array '['")
                pos += 1
                started = True

            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1

                if pos >= len(buffer):
                    chunk = f.read(chunk_size)
                    if not chunk:
                        return
                    buffer = buffer[pos:] + chunk
                    pos = 0
                    continue

                if buffer[pos] == ",":
                    pos += 1
                    continue

                if buffer[pos] == "]":
                    return

                try:
                    obj, end = decoder.raw_decode(buffer, pos)
                    pos = end
                    yield obj
                except json.JSONDecodeError:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        raise
                    buffer = buffer[pos:] + chunk
                    pos = 0


def _removed_count_stream_stats(path, cfg: FTConfig):  # kept for reference, not used
    stats_path = os.path.join(CKPT_DIR, "stream_stats.json")
    os.makedirs(CKPT_DIR, exist_ok=True)

    expected_meta = {
        "size": os.path.getsize(path),
        "sample_fraction": cfg.sample_fraction,
        "split": [cfg.split_train, cfg.split_val, cfg.split_test],
    }

    if os.path.exists(stats_path):
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("meta") == expected_meta:
                return cached
        except Exception:
            pass

    counts = {"train": 0, "val": 0, "test": 0, "sampled": 0, "seen": 0}
    label_set = set()

    t0 = time.time()
    for i, record in enumerate(stream_json_array(path)):
        counts["seen"] += 1
        if not keep_record(i, cfg.sample_fraction):
            continue
        counts["sampled"] += 1
        split = split_for_record(i, cfg)
        counts[split] += 1
        label_set.add(int(record["label"]))

        if (i + 1) % 100000 == 0:
            dt = time.time() - t0
            print(f"  Stats pass: seen {i+1:,}, sampled {counts['sampled']:,} ({dt:.1f}s)")

    stats = {
        "meta": expected_meta,
        "counts": counts,
        "num_labels_seen": len(label_set),
        "max_label_seen": max(label_set) if label_set else -1,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


class _RemovedStreamingTextTrafficDataset:  # not used — replaced by NumpyTrafficDataset
    def __init__(self, json_path, tokenizer, split, cfg: FTConfig):
        super().__init__()
        self.json_path = json_path
        self.tokenizer = tokenizer
        self.split = split
        self.cfg = cfg

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        for i, record in enumerate(stream_json_array(self.json_path)):
            if i % num_workers != worker_id:
                continue

            if not keep_record(i, self.cfg.sample_fraction):
                continue
            if split_for_record(i, self.cfg) != self.split:
                continue

            tokens_list = record["tokens"]
            label = int(record["label"])
            text = tokens_to_rich_text(tokens_list)
            enc = self.tokenizer(
                text,
                max_length=self.cfg.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            yield (
                enc["input_ids"].squeeze(0),
                enc["attention_mask"].squeeze(0),
                label,
            )


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


@torch.no_grad()
def evaluate(model, loader, device, max_steps=None, label_smoothing=0.0):
    model.eval()
    logits_all, labels_all = [], []

    for step, batch in enumerate(loader, start=1):
        ids, mask, y = batch
        out = model(input_ids=ids.to(device), attention_mask=mask.to(device))
        logits_all.append(out.logits.cpu())
        labels_all.append(y)
        if max_steps is not None and step >= max_steps:
            break

    model.train()
    if not logits_all:
        return {"loss": math.nan, "top1": 0.0, "top5": 0.0, "logits": torch.empty(0), "labels": torch.empty(0)}

    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)(logits, labels).item()
    top1 = (logits.argmax(1) == labels).float().mean().item() * 100
    top5 = (logits.topk(5, 1).indices == labels.unsqueeze(1)).any(1).float().mean().item() * 100
    return {"loss": loss, "top1": top1, "top5": top5, "logits": logits, "labels": labels}


def run_phase(
    model,
    train_loader,
    val_loader,
    device,
    n_epochs,
    lr_head,
    lr_backbone,
    cfg,
    tag,
    train_steps_per_epoch,
    eval_steps=None,
):
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "score" in n]
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and "score" not in n]

    param_groups = [{"params": head_params, "lr": lr_head}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})

    steps_per_epoch = train_steps_per_epoch

    total_steps = max(1, n_epochs * steps_per_epoch)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, best_state = 0.0, None

    print(f"\n-- {tag} --")
    print(f"   Trainable : {n_trainable(model):,}")
    if backbone_params:
        print(f"   LR head={lr_head:.0e} backbone={lr_backbone:.0e}")
    else:
        print(f"   LR head={lr_head:.0e} (backbone frozen)")

    for epoch in range(1, n_epochs + 1):
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        t0 = time.time()

        for step, batch in enumerate(train_loader, start=1):
            ids, mask, y = batch
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            ep_loss += loss.item()
            ep_correct += (logits.argmax(1) == y).sum().item()
            ep_total += len(y)

            if step >= steps_per_epoch:
                break

        if ep_total == 0:
            print("   No training batches produced. Check sample_fraction and split settings.")
            break

        train_loss = ep_loss / max(1, min(step, steps_per_epoch))
        train_acc = ep_correct / ep_total * 100

        val = evaluate(
            model,
            val_loader,
            device,
            max_steps=eval_steps,
            label_smoothing=cfg.label_smoothing,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["top1"])

        if val["top1"] > best_val:
            best_val = val["top1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % cfg.log_every_epochs == 0 or epoch == 1:
            print(
                f"   Epoch {epoch:2d}/{n_epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:5.1f}% | "
                f"val loss {val['loss']:.4f} acc {val['top1']:5.1f}% top5 {val['top5']:5.1f}% | "
                f"{time.time() - t0:.1f}s"
            )

    print(f"   Best val acc: {best_val:.1f}%")
    return best_state, history, best_val


def main(cfg=CFG):
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    device = torch.device(cfg.device)
    print(f"Device: {device}")

    print("Loading GPT-2 tokenizer ...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load pre-prepared numpy arrays (from tokenize_v3_dt.py or prepare_data.py)
    X = np.load(f"{cfg.data_dir}/X_tokens.npy")
    y = np.load(f"{cfg.data_dir}/y_labels.npy")   # (N,)

    # Load vocabulary mapping from vocab.pkl if produced by the tokenizer.
    import pickle as _pkl
    _vocab_path = os.path.join(cfg.data_dir, "vocab.pkl")
    vocab = None
    if os.path.exists(_vocab_path):
        with open(_vocab_path, "rb") as _f:
            vocab = _pkl.load(_f)
        print(f"Loaded vocabulary mapping from vocab.pkl  ({len(vocab.get('id2word', {}))} tokens)")
    else:
        print("No vocab.pkl found; using fallback token names")

    if vocab is not None:
        calibrate_bins(cfg.data_dir, vocab)

    # Load split indices — check data/ first (tokenize_v3_dt.py), then checkpoints/
    def _load_split(name):
        for d in [cfg.data_dir, cfg.ckpt_dir]:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return np.load(p)
        raise FileNotFoundError(f"{name} not found in {cfg.data_dir}/ or {cfg.ckpt_dir}/")

    train_idx = _load_split("split_train_idx.npy")
    val_idx   = _load_split("split_val_idx.npy")
    test_idx  = _load_split("split_test_idx.npy")

    if cfg.subset_fraction < 1.0:
        rng    = np.random.default_rng(SEED)
        n_keep = max(cfg.num_classes * 10, int(len(train_idx) * cfg.subset_fraction))
        train_idx = train_idx[rng.choice(len(train_idx), size=n_keep, replace=False)]
        print(f"Quick-test: {len(train_idx):,} training samples ({cfg.subset_fraction:.0%})")

    print(f"\nDataset: {len(train_idx):,} train / {len(val_idx):,} val / {len(test_idx):,} test")

    # Pre-tokenise with GPT-2 BPE (one-time cost, stored in RAM)
    print(f"\nPre-tokenising with GPT-2 BPE (max_length={cfg.max_length}) ...")
    train_ds = NumpyTrafficDataset(X[train_idx], y[train_idx], tokenizer, cfg.max_length, vocab=vocab)
    val_ds   = NumpyTrafficDataset(X[val_idx],   y[val_idx],   tokenizer, cfg.max_length, vocab=vocab)
    test_ds  = NumpyTrafficDataset(X[test_idx],  y[test_idx],  tokenizer, cfg.max_length, vocab=vocab)

    _pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=4, pin_memory=_pin)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=4, pin_memory=_pin)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                              num_workers=4, pin_memory=_pin)

    steps_per_epoch = len(train_loader)

    print(f"\nLoading GPT2ForSequenceClassification (117M params) ...")
    model = GPT2ForSequenceClassification.from_pretrained(
        "gpt2",
        num_labels=cfg.num_classes,
        resid_pdrop=0.2,   # was 0.1 — more dropout to reduce overfitting
        embd_pdrop=0.2,
        attn_pdrop=0.2,
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    model = model.to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total/1e6:.1f}M\n")

    all_history = {}

    # Phase 1: frozen backbone, head only
    freeze_backbone(model)
    p1_state, p1_hist, _ = run_phase(
        model, train_loader, val_loader, device,
        cfg.p1_epochs, cfg.p1_lr, 0.0, cfg,
        "Phase 1: frozen GPT-2, head only",
        train_steps_per_epoch=steps_per_epoch,
    )
    all_history["phase1"] = p1_hist
    if p1_state is not None:
        model.load_state_dict(p1_state)

    # Phase 2: unfreeze top-N transformer blocks
    unfreeze_top_layers(model, cfg.p2_unfreeze)
    p2_state, p2_hist, _ = run_phase(
        model, train_loader, val_loader, device,
        cfg.p2_epochs, cfg.p2_lr_head, cfg.p2_lr_backbone, cfg,
        f"Phase 2: top {cfg.p2_unfreeze} blocks + head",
        train_steps_per_epoch=steps_per_epoch,
    )
    all_history["phase2"] = p2_hist
    if p2_state is not None:
        model.load_state_dict(p2_state)

    # Phase 3: full fine-tune
    unfreeze_all(model)
    p3_state, p3_hist, _ = run_phase(
        model, train_loader, val_loader, device,
        cfg.p3_epochs, cfg.p3_lr_head, cfg.p3_lr_backbone, cfg,
        "Phase 3: full fine-tune",
        train_steps_per_epoch=steps_per_epoch,
    )
    all_history["phase3"] = p3_hist
    if p3_state is not None:
        model.load_state_dict(p3_state)

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION  (GPT-2 fine-tuning)")
    print("=" * 60)
    test   = evaluate(model, test_loader, device, label_smoothing=0.0)
    preds  = test["logits"].argmax(1).numpy()
    labels = test["labels"].numpy()
    print(f"Top-1 accuracy : {test['top1']:.2f}%")
    print(f"Top-5 accuracy : {test['top5']:.2f}%")
    print(f"Test loss      : {test['loss']:.4f}")
    print(f"Correct        : {(preds == labels).sum()} / {len(labels)}")

    model.save_pretrained(f"{cfg.ckpt_dir}/gpt2_traffic")
    tokenizer.save_pretrained(f"{cfg.ckpt_dir}/gpt2_traffic")
    with open(f"{cfg.ckpt_dir}/gpt2_history.json", "w") as f:
        json.dump(all_history, f, indent=2)
    print(f"\nSaved → {cfg.ckpt_dir}/gpt2_traffic/")
    return model, test


if __name__ == "__main__":
    import argparse
    base = Path(__file__).parent
    CFG.data_dir = str(base / "data")
    CFG.ckpt_dir = str(base / "checkpoints")

    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=float, default=1.0,
                        help="Fraction of training data (0<f<=1.0); <1.0 = quick-test")
    args = parser.parse_args()
    if 0.0 < args.subset < 1.0:
        CFG.subset_fraction = args.subset
        CFG.p1_epochs = min(CFG.p1_epochs, 2)
        CFG.p2_epochs = min(CFG.p2_epochs, 5)
        CFG.p3_epochs = min(CFG.p3_epochs, 5)
        print(f"Quick-test mode: subset={args.subset:.0%}, epochs 2+5+5")
    main(CFG)