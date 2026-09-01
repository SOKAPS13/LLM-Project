import os, math, json, pickle, time, sys, logging, traceback
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from collections import defaultdict

from model_gmm import TrafficGPT, Config


def setup_logging(log_dir: str):
    """Write INFO+ to training.log and ERROR+ to error.log, and mirror both to stdout."""
    os.makedirs(log_dir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Full training log
    fh = logging.FileHandler(f"{log_dir}/training.log", mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Errors only
    eh = logging.FileHandler(f"{log_dir}/error.log", mode="a")
    eh.setLevel(logging.ERROR)
    eh.setFormatter(fmt)
    root.addHandler(eh)

    return logging.getLogger(__name__)

log = logging.getLogger(__name__)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

CFG = Config()


def _valid_saved_split(train_idx, val_idx, test_idx, n_total):
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    if len(all_idx) == 0:
        return False
    if np.any(all_idx < 0) or np.any(all_idx >= n_total):
        return False
    if len(np.unique(all_idx)) != len(all_idx):
        return False
    return True


def _load_existing_split(data_dir: str, n_total: int):
    train_p = Path(data_dir) / "split_train_idx.npy"
    val_p = Path(data_dir) / "split_val_idx.npy"
    test_p = Path(data_dir) / "split_test_idx.npy"
    if not (train_p.exists() and val_p.exists() and test_p.exists()):
        return None

    train_idx = np.load(train_p)
    val_idx = np.load(val_p)
    test_idx = np.load(test_p)
    if _valid_saved_split(train_idx, val_idx, test_idx, n_total):
        return train_idx, val_idx, test_idx
    return None


def configure_for_mac(cfg: Config):
    """Lower training cost for Apple Silicon / laptop runs without changing the model."""
    if str(cfg.device) == "mps" or str(cfg.device) == "cpu":
        cfg.epochs = 10
        cfg.batch_size = 16
        cfg.warmup_steps = 100
        cfg.log_every = 2
        cfg.save_every = 5


# ── Stratified split ──────────────────────────────────────────────────────────
def stratified_split(y: np.ndarray, train_k: int, val_k: int, test_k: int, seed: int = SEED):
    rng = np.random.default_rng(seed)
    class_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_indices[int(label)].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for label, indices in sorted(class_indices.items()):
        n = len(indices)
        perm = rng.permutation(n)
        shuffled = [indices[i] for i in perm]
        t_k = min(train_k, n)
        v_k = min(val_k, n - t_k)
        e_k = min(test_k, n - t_k - v_k)
        train_idx.extend(shuffled[:t_k])
        val_idx.extend(shuffled[t_k : t_k + v_k])
        test_idx.extend(shuffled[t_k + v_k : t_k + v_k + e_k])

    return train_idx, val_idx, test_idx


# ── Dataset ───────────────────────────────────────────────────────────────────
class TrafficDataset(Dataset):
    """
    Each item: (input_ids, labels)
      input_ids = seq[:-1]   shape (601,)
      labels    = seq[1:]    shape (601,)
    PAD tokens in labels → -100 (ignored by CrossEntropyLoss).

    Uses memory-mapped numpy arrays so the full file is never loaded into RAM.
    Pass subset_fraction (0 < f <= 1.0) for a stratified random subset.
    """
    def __init__(self, data_dir: str, vocab_cfg: dict, subset_fraction: float = 1.0):
        # mmap_mode='r' keeps data on disk; only requested rows are read
        X = np.load(f"{data_dir}/X_tokens.npy", mmap_mode='r')   # (N, 602)
        y = np.load(f"{data_dir}/y_labels.npy", mmap_mode='r')   # (N,)

        if subset_fraction < 1.0:
            rng = np.random.default_rng(SEED)
            classes, counts = np.unique(y, return_counts=True)
            keep = []
            for cls, cnt in zip(classes, counts):
                idx = np.where(y == cls)[0]
                n_keep = max(1, int(cnt * subset_fraction))
                keep.append(rng.choice(idx, size=n_keep, replace=False))
            keep = np.sort(np.concatenate(keep))
            # Copy only the chosen rows into memory (much smaller)
            X = np.array(X[keep])
            y = np.array(y[keep])

        self.seqs   = torch.tensor(np.array(X), dtype=torch.long)
        self.labels = torch.tensor(np.array(y), dtype=torch.long)
        self.PAD    = vocab_cfg["special"]["PAD"]
        self.seq_len = int(self.seqs.shape[1])

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
    inp, tgt = batch
    inp, tgt = inp.to(device), tgt.to(device)
    logits   = model(inp)
    B, T, V  = logits.shape
    return nn.CrossEntropyLoss(ignore_index=-100)(
        logits.view(B * T, V), tgt.view(B * T))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        total += compute_loss(model, batch, device).item(); n += 1
    model.train()
    return total / max(1, n)


# ── Main training loop ────────────────────────────────────────────────────────
def train(cfg: Config = CFG):
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device(cfg.device)
    log.info(f"Device      : {device}")
    log.info(f"Data dir    : {cfg.data_dir}")

    # Load vocab
    with open(f"{cfg.data_dir}/vocab.pkl", "rb") as f:
        vocab_cfg = pickle.load(f)
    cfg.vocab_size = vocab_cfg["vocab_size"]

    # prepare_data.py already subsampled the data — use all of it here.
    # To use less data, re-run: python prepare_data.py --subset 0.1
    full_ds = TrafficDataset(cfg.data_dir, vocab_cfg, subset_fraction=1.0)
    y_all   = full_ds.labels.numpy()
    cfg.max_seq_len = full_ds.seq_len

    existing_split = _load_existing_split(cfg.data_dir, len(full_ds))
    if existing_split is not None:
        train_idx, val_idx, test_idx = existing_split
        log.info("Split       : reusing tokenizer-provided split from data directory")
    else:
        train_idx, val_idx, test_idx = stratified_split(
            y_all, cfg.train_k, cfg.val_k, cfg.test_k)
        log.info("Split       : no saved tokenizer split found, generated a new stratified split")

    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)
    test_ds  = Subset(full_ds, test_idx)

    log.info(f"Dataset     : {len(full_ds)} total sequences, {len(np.unique(y_all))} classes")
    log.info(f"Split       : {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")
    log.info(f"Sequence len: {cfg.max_seq_len} tokens")
    log.info(f"Vocab size  : {cfg.vocab_size}")

    # Save split indices for Part 3 reproducibility
    np.save(f"{cfg.output_dir}/split_train_idx.npy", np.array(train_idx))
    np.save(f"{cfg.output_dir}/split_val_idx.npy",   np.array(val_idx))
    np.save(f"{cfg.output_dir}/split_test_idx.npy",  np.array(test_idx))

    # On MPS (Apple Silicon) pin_memory must be False; num_workers=0 avoids
    # multiprocessing issues with the Metal backend
    is_mps = str(device) == "mps"
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=0 if is_mps else 4,
                              pin_memory=not is_mps)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False,
                              num_workers=0 if is_mps else 4,
                              pin_memory=not is_mps)

    # Model
    model = TrafficGPT(cfg).to(device)
    if torch.cuda.device_count() > 1:
        log.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    log.info(f"Model       : {raw_model.count_params():,} parameters")
    log.info(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
             f"n_heads={cfg.n_heads}, d_ff={cfg.d_ff}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr,
        weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val, global_step = float("inf"), 0
    steps_per_epoch = len(train_loader)

    log.info(f"Pre-training for {cfg.epochs} epochs ({steps_per_epoch} steps/epoch) …")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        ep_loss, n = 0.0, 0
        t0 = time.time()

        for batch in train_loader:
            lr = get_lr(global_step, cfg, steps_per_epoch)
            for pg in optimizer.param_groups: pg["lr"] = lr

            optimizer.zero_grad()
            loss = compute_loss(model, batch, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            ep_loss += loss.item(); n += 1; global_step += 1

            # Log progress every 100 steps so you can see it's alive
            if n % 100 == 0:
                log.info(f"  Epoch {epoch}/{cfg.epochs}  step {n}/{steps_per_epoch}"
                         f"  loss {ep_loss/n:.4f}  "
                         f"elapsed {time.time()-t0:.0f}s")

        train_loss = ep_loss / n
        val_loss   = evaluate(model, val_loader, device)
        elapsed    = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr)

        if epoch % cfg.log_every == 0 or epoch == 1:
            log.info(f"Epoch {epoch:4d}/{cfg.epochs} | "
                     f"train {train_loss:.4f} (ppl {math.exp(min(train_loss,20)):.1f}) | "
                     f"val {val_loss:.4f} (ppl {math.exp(min(val_loss,20)):.1f}) | "
                     f"lr {lr:.2e} | {elapsed:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                "epoch": epoch, "model_state": raw_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss, "cfg": cfg.__dict__,
            }, f"{cfg.output_dir}/best_pretrain.pt")

        if epoch % cfg.save_every == 0:
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                "epoch": epoch, "model_state": raw_model.state_dict(),
                "val_loss": val_loss, "cfg": cfg.__dict__,
            }, f"{cfg.output_dir}/pretrain_epoch{epoch}.pt")

    with open(f"{cfg.output_dir}/pretrain_history.json", "w") as f:
        json.dump(history, f, indent=2)

    log.info(f"Done. Best val loss: {best_val:.4f} "
             f"(ppl {math.exp(min(best_val,20)):.1f})")
    log.info(f"Checkpoint: {cfg.output_dir}/best_pretrain.pt")
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
    assert not set(ti) & set(vi) & set(tsi), "Split indices overlap!"
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
    BASE_DIR = Path(__file__).parent
    CFG.data_dir = str(BASE_DIR / "data")
    CFG.output_dir = str(BASE_DIR / "checkpoints")
    configure_for_mac(CFG)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=CFG.data_dir,
                        help="directory containing X_tokens.npy, y_labels.npy, and vocab.pkl")
    parser.add_argument("--output_dir", type=str, default=CFG.output_dir,
                        help="directory to save checkpoints and split copies")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override number of pre-training epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="override pre-training batch size")
    parser.add_argument("test", nargs="?", default=None,
                        help="pass 'test' to run the smoke test")
    args = parser.parse_args()

    CFG.data_dir = args.data_dir
    CFG.output_dir = args.output_dir
    if args.epochs is not None:
        CFG.epochs = args.epochs
    if args.batch_size is not None:
        CFG.batch_size = args.batch_size

    log = setup_logging(CFG.output_dir)
    log.info("=" * 60)
    log.info("Training started")
    log.info("=" * 60)

    if args.test == "test":
        smoke_test()
    else:
        try:
            train(CFG)
        except Exception:
            log.error("Training crashed:\n" + traceback.format_exc())
            sys.exit(1)