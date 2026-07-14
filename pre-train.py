import os, math, json, pickle, time, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from collections import defaultdict

from model import TrafficGPT, Config

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

CFG = Config()


# ── Stratified split ──────────────────────────────────────────────────────────
def stratified_split(y: np.ndarray, train_k: int, val_k: int, test_k: int, seed: int = SEED):
    rng = np.random.default_rng(seed)
    class_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_indices[int(label)].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for label, indices in sorted(class_indices.items()):
        n = len(indices)
        assert n == train_k + val_k + test_k, \
            f"Class {label} has {n} samples, expected {train_k+val_k+test_k}"
        perm     = rng.permutation(n)
        shuffled = [indices[i] for i in perm]
        train_idx.extend(shuffled[:train_k])
        val_idx.extend(shuffled[train_k:train_k + val_k])
        test_idx.extend(shuffled[train_k + val_k:])

    return train_idx, val_idx, test_idx


# ── Dataset ───────────────────────────────────────────────────────────────────
class TrafficDataset(Dataset):
    """
    Each item: (input_ids, labels)
      input_ids = seq[:-1]   shape (601,)
      labels    = seq[1:]    shape (601,)
    PAD tokens in labels → -100 (ignored by CrossEntropyLoss).
    """
    def __init__(self, data_dir: str, vocab_cfg: dict):
        X = np.load(f"{data_dir}/X_tokens.npy")   # (1000, 602)
        y = np.load(f"{data_dir}/y_labels.npy")   # (1000,)
        self.seqs   = torch.tensor(X, dtype=torch.long)
        self.labels = torch.tensor(y, dtype=torch.long)
        self.PAD    = vocab_cfg["special"]["PAD"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq       = self.seqs[idx]
        input_ids = seq[:-1]
        labels    = seq[1:].clone()
        labels[labels == self.PAD] = -100
        return input_ids, labels


# ── LR schedule ───────────────────────────────────────────────────────────────
def get_lr(step, cfg, steps_per_epoch):
    total = cfg.epochs * steps_per_epoch
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


# ── Train / eval helpers ──────────────────────────────────────────────────────
def compute_loss(model, batch, device):
    """
    Returns (loss, n_tokens):
      loss     — mean CE over the non-ignored target tokens in this batch
      n_tokens — number of non-ignored (!= -100) target tokens

    n_tokens lets callers form a token-weighted average across batches, so a
    short trailing batch is not weighted equally with a full one. This matters
    here because best-checkpoint selection rides on the val loss and the val
    set is tiny (val_k=1 → 100 sequences).
    """
    inp, tgt = batch
    inp, tgt = inp.to(device), tgt.to(device)
    logits   = model(inp)
    B, T, V  = logits.shape
    loss = nn.CrossEntropyLoss(ignore_index=-100)(
        logits.view(B * T, V), tgt.view(B * T))
    n_tokens = int((tgt != -100).sum().item())
    return loss, n_tokens


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in loader:
        loss, n_tokens = compute_loss(model, batch, device)
        total_loss   += loss.item() * n_tokens
        total_tokens += n_tokens
    model.train()
    return total_loss / max(1, total_tokens)


def cfg_to_dict(cfg) -> dict:
    """Full config snapshot: class-level defaults + any instance overrides.
    (Config declares its fields as class attributes, so cfg.__dict__ alone
    would miss the architecture — d_model, n_layers, etc.)"""
    keys = set(vars(type(cfg))) | set(vars(cfg))
    return {k: getattr(cfg, k) for k in keys
            if not k.startswith("__") and not callable(getattr(cfg, k))}


# ── Main training loop ────────────────────────────────────────────────────────
def train(cfg: Config = CFG):
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device(cfg.device)
    print(f"Device      : {device}")
    print(f"Data dir    : {cfg.data_dir}")

    # Load vocab
    with open(f"{cfg.data_dir}/vocab.pkl", "rb") as f:
        vocab_cfg = pickle.load(f)
    cfg.vocab_size = vocab_cfg["vocab_size"]

    # Full dataset
    full_ds = TrafficDataset(cfg.data_dir, vocab_cfg)
    y_all   = full_ds.labels.numpy()

    # Split indices are computed in pre-processing.py — BEFORE the scaler/K-means
    # codebook is fit — so that fitting only ever sees train traces. We load
    # (not recompute) them here so training uses the exact same split.
    train_idx = np.load(f"{cfg.data_dir}/split_train_idx.npy").tolist()
    val_idx   = np.load(f"{cfg.data_dir}/split_val_idx.npy").tolist()
    test_idx  = np.load(f"{cfg.data_dir}/split_test_idx.npy").tolist()
    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)
    test_ds  = Subset(full_ds, test_idx)

    print(f"\nDataset     : {len(full_ds)} total sequences, {len(np.unique(y_all))} classes")
    print(f"Split       : {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")
    print(f"  = {cfg.train_k} train + {cfg.val_k} val + {cfg.test_k} test per class")

    # Save split indices for Part 3 reproducibility
    np.save(f"{cfg.output_dir}/split_train_idx.npy", np.array(train_idx))
    np.save(f"{cfg.output_dir}/split_val_idx.npy",   np.array(val_idx))
    np.save(f"{cfg.output_dir}/split_test_idx.npy",  np.array(test_idx))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = TrafficGPT(cfg).to(device)
    print(f"\nModel       : {model.count_params():,} parameters")
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"n_heads={cfg.n_heads}, d_ff={cfg.d_ff}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr,
        weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val, global_step = float("inf"), 0
    steps_per_epoch = len(train_loader)

    print(f"\nPre-training for {cfg.epochs} epochs "
          f"({steps_per_epoch} steps/epoch) …\n")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        ep_loss, ep_tokens = 0.0, 0
        t0 = time.time()

        for batch in train_loader:
            lr = get_lr(global_step, cfg, steps_per_epoch)
            for pg in optimizer.param_groups: pg["lr"] = lr

            optimizer.zero_grad()
            loss, n_tokens = compute_loss(model, batch, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            # Token-weighted accumulation: matches the eval convention so that
            # the reported train/val losses are on the same scale.
            ep_loss   += loss.item() * n_tokens
            ep_tokens += n_tokens
            global_step += 1

        train_loss = ep_loss / max(1, ep_tokens)
        val_loss   = evaluate(model, val_loader, device)
        elapsed    = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr)

        if epoch % cfg.log_every == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{cfg.epochs} | "
                  f"train {train_loss:.4f} (ppl {math.exp(min(train_loss,20)):.1f}) | "
                  f"val {val_loss:.4f} (ppl {math.exp(min(val_loss,20)):.1f}) | "
                  f"lr {lr:.2e} | {elapsed:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss, "cfg": cfg_to_dict(cfg),
            }, f"{cfg.output_dir}/best_pretrain.pt")

        if epoch % cfg.save_every == 0:
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_loss": val_loss, "cfg": cfg_to_dict(cfg),
            }, f"{cfg.output_dir}/pretrain_epoch{epoch}.pt")

    with open(f"{cfg.output_dir}/pretrain_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val loss: {best_val:.4f} "
          f"(ppl {math.exp(min(best_val,20)):.1f})")
    print(f"Checkpoint: {cfg.output_dir}/best_pretrain.pt")
    return model, history


# ── Smoke test ────────────────────────────────────────────────────────────────
def smoke_test():
    print("=" * 60)
    print("PRE-TRAIN — SMOKE TEST  (1000 files / 100 classes)")
    print("=" * 60)
    cfg   = Config()
    model = TrafficGPT(cfg)
    print(f"vocab_size={cfg.vocab_size}  d_model={cfg.d_model}  "
          f"n_layers={cfg.n_layers}  n_heads={cfg.n_heads}  d_ff={cfg.d_ff}")
    print(f"Parameters : {model.count_params():,}")

    # Stratified split check
    y_dummy = np.repeat(np.arange(100), 10)
    ti, vi, tsi = stratified_split(y_dummy, 8, 1, 1)
    print(f"\nStratified split: train={len(ti)} / val={len(vi)} / test={len(tsi)}  ✓")
    assert not (set(ti) & set(vi)) and not (set(vi) & set(tsi)) \
        and not (set(ti) & set(tsi)), "Split indices overlap!"
    print(f"No overlap between splits  ✓")

    # Forward pass
    B, T   = 8, 601
    dummy  = torch.randint(0, cfg.vocab_size, (B, T))
    logits = model(dummy)
    hidden = model.get_hidden_states(dummy)
    print(f"\nForward pass  : {tuple(dummy.shape)} → {tuple(logits.shape)}  ✓")
    print(f"Hidden states : {tuple(hidden.shape)}  ✓")

    loss = nn.CrossEntropyLoss()(logits.view(-1, cfg.vocab_size),
                                  torch.randint(0, cfg.vocab_size, (B*T,)))
    print(f"Random loss   : {loss.item():.4f}  "
          f"(expected ≈ {math.log(cfg.vocab_size):.2f})  ✓")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_DIR = "/home2/somka34/somiya/LLM/data"
    CKPT_DIR = "/home2/somka34/somiya/LLM/checkpoints"

    if len(sys.argv) == 2 and sys.argv[1] == "test":
        smoke_test()
    else:
        CFG.data_dir   = DATA_DIR
        CFG.output_dir = CKPT_DIR
        train(CFG)
