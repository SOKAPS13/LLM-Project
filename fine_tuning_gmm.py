import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from model_gmm import Config as PretrainConfig, TrafficGPT


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


class FTConfig:
    data_dir: str = "./data"
    ckpt_dir: str = "./checkpoints"
    pretrain_ckpt: str = "./checkpoints/best_pretrain.pt"
    num_classes: int = 100
    subset_fraction: float = 1.0  # <1.0 → quick-test mode
    pool: str = "last"   # last token at position 600 (pre-training boundary)
    head_dropout: float = 0.3
    label_smoothing: float = 0.1
    p1_epochs: int = 15
    p1_lr: float = 1e-3
    p1_batch_size: int = 256
    p2_epochs: int = 100
    p2_lr_head: float = 5e-4
    p2_lr_backbone: float = 5e-5
    p2_batch_size: int = 256
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 200
    log_every: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


FTCFG = FTConfig()


def configure_for_mac(cfg: FTConfig):
    """Reduce fine-tuning cost for Mac runs without changing the model."""
    if str(cfg.device) in ("mps", "cpu"):
        cfg.p1_epochs = 5
        cfg.p2_epochs = 10
        cfg.p1_batch_size = 8
        cfg.p2_batch_size = 8
        cfg.log_every = 1


class ClassificationDataset(Dataset):
    def __init__(self, data_dir):
        x = np.load(f"{data_dir}/X_tokens.npy", mmap_mode="r")
        y = np.load(f"{data_dir}/y_labels.npy", mmap_mode="r")
        self.x = torch.tensor(np.array(x), dtype=torch.long)
        self.y = torch.tensor(np.array(y), dtype=torch.long)
        self.seq_len = int(self.x.shape[1])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class AttentionPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.scorer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, mask):
        scores = self.scorer(hidden).squeeze(-1)
        valid_rows = mask.any(dim=1)
        safe_mask = mask.clone()
        safe_mask[~valid_rows, -1] = True
        weights = torch.softmax(scores.masked_fill(~safe_mask, float("-inf")), dim=1)
        return (weights.unsqueeze(-1) * hidden).sum(dim=1)


class TrafficClassifier(nn.Module):
    def __init__(self, backbone, num_classes, pool="mean", head_dropout=0.3,
                 pad_token_id=0, bos_token_id=None, eos_token_id=None,
                 silent_token_id=None):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.silent_token_id = silent_token_id
        hidden = backbone.cfg.d_model
        self.attention_pool = AttentionPool(hidden)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(head_dropout),
            nn.Linear(hidden, num_classes),
        )

    def _masked_mean(self, hidden, token_ids, mask):
        mask = mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1)
        pooled = (hidden * mask).sum(dim=1) / denom
        fallback = hidden[:, -1, :]
        has_any = mask.squeeze(-1).any(dim=1, keepdim=True)
        return torch.where(has_any, pooled, fallback)

    def forward(self, input_ids):
        backbone_input = input_ids[:, :-1]
        h = self.backbone.get_hidden_states(backbone_input)

        if self.pool == "last":
            nonpad = backbone_input != self.pad_token_id
            last_idx = nonpad.sum(dim=1).clamp(min=1) - 1
            rep = h[torch.arange(h.size(0), device=h.device), last_idx]
        else:
            content_mask = backbone_input != self.pad_token_id
            if self.bos_token_id is not None:
                content_mask &= backbone_input != self.bos_token_id
            if self.eos_token_id is not None:
                content_mask &= backbone_input != self.eos_token_id
            if self.pool == "attention":
                if self.silent_token_id is not None:
                    content_mask &= backbone_input != self.silent_token_id
                rep = self.attention_pool(h, content_mask)
            elif self.pool == "mean_nonpad":
                rep = self._masked_mean(h, backbone_input, content_mask)
            else:  # mean
                rep = self._masked_mean(h, backbone_input, content_mask)

        return self.classifier(rep)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_lr(step, base_lr, warmup_steps, total_steps):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def evaluate(model, loader, device, label_smoothing=0.0):
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        all_logits.append(model(x.to(device)).cpu())
        all_labels.append(y)
    model.train()

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)(logits, labels).item()
    top1 = (logits.argmax(1) == labels).float().mean().item() * 100
    top5 = (logits.topk(5, dim=1).indices == labels.unsqueeze(1)).any(1).float().mean().item() * 100
    return {"loss": loss, "top1": top1, "top5": top5, "logits": logits, "labels": labels}


def _unwrap(model):
    """Return the underlying module whether or not DataParallel is used."""
    return model.module if isinstance(model, nn.DataParallel) else model


def run_phase(model, train_loader, val_loader, device, n_epochs, lr_head, lr_backbone, cfg, tag):
    raw = _unwrap(model)
    head_params = list(raw.classifier.parameters()) + list(raw.attention_pool.parameters())
    backbone_params = [p for p in raw.backbone.parameters() if p.requires_grad]

    param_groups = [{"params": head_params, "lr": lr_head}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    total_steps = max(1, n_epochs * len(train_loader))
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, best_state, step = 0.0, None, 0

    print(f"\n-- {tag} --")
    print(f"   Trainable params: {_unwrap(model).trainable_params():,}")
    if backbone_params:
        print(f"   LR head={lr_head:.0e} backbone={lr_backbone:.0e}")
    else:
        print(f"   LR head={lr_head:.0e}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            new_lr = get_lr(step, lr_head, cfg.warmup_steps, total_steps)
            optimizer.param_groups[0]["lr"] = new_lr
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]["lr"] = new_lr * (lr_backbone / lr_head)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            ep_loss += loss.item()
            ep_correct += (logits.argmax(1) == y).sum().item()
            ep_total += len(y)
            step += 1

        train_loss = ep_loss / max(1, len(train_loader))
        train_acc = ep_correct / max(1, ep_total) * 100
        val = evaluate(model, val_loader, device, cfg.label_smoothing)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["top1"])

        if val["top1"] > best_val:
            best_val = val["top1"]
            best_state = {k: v.cpu().clone() for k, v in _unwrap(model).state_dict().items()}

        if epoch % cfg.log_every == 0 or epoch == 1:
            print(
                f"   Epoch {epoch:3d}/{n_epochs} | train loss {train_loss:.4f} acc {train_acc:5.1f}% | "
                f"val loss {val['loss']:.4f} acc {val['top1']:5.1f}% top5 {val['top5']:5.1f}% | {time.time() - t0:.1f}s"
            )

    print(f"   Best val acc: {best_val:.1f}%")
    return best_state, history, best_val


def fine_tune(cfg=FTCFG):
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    device = torch.device(cfg.device)
    is_mps = str(device) == "mps"
    print(f"Device: {device}")

    with open(f"{cfg.data_dir}/vocab.pkl", "rb") as f:
        vocab_cfg = pickle.load(f)
    pad_token_id = vocab_cfg["special"]["PAD"]
    bos_token_id = vocab_cfg["special"].get("BOS")
    eos_token_id = vocab_cfg["special"].get("EOS")
    silent_token_id = vocab_cfg.get("silent")

    ckpt = torch.load(cfg.pretrain_ckpt, map_location="cpu", weights_only=False)
    pt_cfg = PretrainConfig()
    for key, value in ckpt["cfg"].items():
        if hasattr(pt_cfg, key):
            setattr(pt_cfg, key, value)

    backbone = TrafficGPT(pt_cfg)
    backbone.load_state_dict(ckpt["model_state"])
    print(f"Loaded backbone (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f})")

    model = TrafficClassifier(
        backbone,
        cfg.num_classes,
        pool=cfg.pool,
        head_dropout=cfg.head_dropout,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        silent_token_id=silent_token_id,
    ).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    full_ds = ClassificationDataset(cfg.data_dir)
    train_split_path = Path(cfg.data_dir) / "split_train_idx.npy"
    val_split_path = Path(cfg.data_dir) / "split_val_idx.npy"
    test_split_path = Path(cfg.data_dir) / "split_test_idx.npy"
    if not (train_split_path.exists() and val_split_path.exists() and test_split_path.exists()):
        train_split_path = Path(cfg.ckpt_dir) / "split_train_idx.npy"
        val_split_path = Path(cfg.ckpt_dir) / "split_val_idx.npy"
        test_split_path = Path(cfg.ckpt_dir) / "split_test_idx.npy"

    train_idx = np.load(train_split_path)
    val_idx = np.load(val_split_path)
    test_idx = np.load(test_split_path)

    if cfg.subset_fraction < 1.0:
        rng = np.random.default_rng(SEED)
        n_full = len(train_idx)
        n_keep = max(cfg.num_classes * 10, int(n_full * cfg.subset_fraction))
        keep = rng.choice(n_full, size=n_keep, replace=False)
        train_idx = train_idx[keep]
        print(f"Quick-test mode: {len(train_idx):,} / {n_full:,} training samples "
              f"({cfg.subset_fraction:.0%})")

    loader_workers = 0 if is_mps else 4
    loader_pin_memory = not is_mps

    train_loader = DataLoader(
        Subset(full_ds, train_idx),
        batch_size=cfg.p1_batch_size,
        shuffle=True,
        num_workers=loader_workers,
        pin_memory=loader_pin_memory,
    )
    val_loader = DataLoader(
        Subset(full_ds, val_idx),
        batch_size=cfg.p1_batch_size,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=loader_pin_memory,
    )
    test_loader = DataLoader(
        Subset(full_ds, test_idx),
        batch_size=cfg.p1_batch_size,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=loader_pin_memory,
    )

    print(f"Dataset: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test\n")

    all_history = {}

    _unwrap(model).freeze_backbone()
    p1_state, p1_hist, _ = run_phase(model, train_loader, val_loader, device, cfg.p1_epochs, cfg.p1_lr, 0.0, cfg, "Phase 1")
    all_history["phase1"] = p1_hist
    if p1_state is not None:
        _unwrap(model).load_state_dict(p1_state)

    _unwrap(model).unfreeze_backbone()
    p2_state, p2_hist, _ = run_phase(model, train_loader, val_loader, device, cfg.p2_epochs, cfg.p2_lr_head, cfg.p2_lr_backbone, cfg, "Phase 2")
    all_history["phase2"] = p2_hist
    if p2_state is not None:
        _unwrap(model).load_state_dict(p2_state)

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    test = evaluate(model, test_loader, device)
    preds = test["logits"].argmax(1).numpy()
    labels = test["labels"].numpy()

    print(f"Top-1 accuracy : {test['top1']:.2f}%")
    print(f"Top-5 accuracy : {test['top5']:.2f}%")
    print(f"Test loss      : {test['loss']:.4f}")
    print(f"Classes correct: {(preds == labels).sum()} / {cfg.num_classes}")

    torch.save(
        {
            "model_state": _unwrap(model).state_dict(),
            "cfg": {
                "num_classes": cfg.num_classes,
                "pool": cfg.pool,
                "head_dropout": cfg.head_dropout,
                "pad_token_id": pad_token_id,
            },
            "test_top1": test["top1"],
            "test_top5": test["top5"],
        },
        f"{cfg.ckpt_dir}/best_finetune.pt",
    )

    with open(f"{cfg.ckpt_dir}/finetune_history.json", "w", encoding="utf-8") as f:
        json.dump(all_history, f, indent=2)

    print(f"\nSaved: best_finetune.pt -> {cfg.ckpt_dir}/")
    return model, test


def smoke_test():
    print("=" * 60)
    print("FINE-TUNING - SMOKE TEST")
    print("=" * 60)
    cfg = PretrainConfig()
    backbone = TrafficGPT(cfg)

    for pool in ("last", "mean", "mean_nonpad", "attention"):
        model = TrafficClassifier(backbone, 100, pool=pool, pad_token_id=18)
        dummy = torch.randint(0, cfg.vocab_size, (4, 602))
        logits = model(dummy)
        print(f"pool='{pool}': input {tuple(dummy.shape)} -> {tuple(logits.shape)}")
    print("Smoke test passed  ✓")


if __name__ == "__main__":
    import argparse

    base_dir = Path(__file__).parent
    FTCFG.data_dir = str(base_dir / "data")
    FTCFG.ckpt_dir = str(base_dir / "checkpoints")
    FTCFG.pretrain_ckpt = str(base_dir / "checkpoints" / "best_pretrain.pt")
    configure_for_mac(FTCFG)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=FTCFG.data_dir,
                        help="directory containing tokenized arrays and vocab.pkl")
    parser.add_argument("--ckpt_dir", type=str, default=FTCFG.ckpt_dir,
                        help="directory to save fine-tuning checkpoints")
    parser.add_argument("--pretrain_ckpt", type=str, default=FTCFG.pretrain_ckpt,
                        help="path to the pre-training checkpoint")
    parser.add_argument("--pool", type=str, default=FTCFG.pool,
                        choices=["last", "mean", "mean_nonpad", "attention"],
                        help="pooling strategy for sequence classification")
    parser.add_argument("--subset", type=float, default=1.0,
                        help="fraction of training data to use (0<f<=1.0)")
    parser.add_argument("--test", action="store_true", help="run smoke test only")
    args = parser.parse_args()

    FTCFG.data_dir = args.data_dir
    FTCFG.ckpt_dir = args.ckpt_dir
    FTCFG.pretrain_ckpt = args.pretrain_ckpt
    FTCFG.pool = args.pool

    if args.test:
        smoke_test()
    else:
        if 0.0 < args.subset < 1.0:
            FTCFG.subset_fraction = args.subset
            # Fewer epochs for quick tests — enough to see if learning happens
            FTCFG.p1_epochs = min(FTCFG.p1_epochs, 5)
            FTCFG.p2_epochs = min(FTCFG.p2_epochs, 15)
            print(f"Quick-test mode: subset={args.subset:.0%}, "
                  f"p1_epochs={FTCFG.p1_epochs}, p2_epochs={FTCFG.p2_epochs}")
        fine_tune(FTCFG)
