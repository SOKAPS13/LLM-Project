"""
Part 3 — Fine-tune for Video Classification
Video Fingerprinting via Encrypted Network Traffic

Takes the pre-trained TrafficGPT backbone from Part 2 and adds a
classification head to identify which of the 100 videos a traffic
sequence belongs to.

Architecture:
  Backbone  : pre-trained TrafficGPT  (frozen in Phase 1, unfrozen in Phase 2)
  Pooling   : learned attention pooling over content windows (pos 1..600),
              silent windows masked out of the attention softmax
  Head      : LayerNorm(256) → Linear(256 → 256) → GELU → Dropout(0.3) → Linear(256 → 100)

Two-phase fine-tuning:
  Phase 1 — frozen backbone (20 epochs, lr=1e-3)
    Train only the classification head.
    The backbone's representations are used as-is.
    Fast convergence, sets a strong baseline.

  Phase 2 — full fine-tune (100 epochs, backbone lr=1e-5, head lr=1e-4)
    Unfreeze the entire backbone with a 10× lower LR than the head.
    Allows the backbone to adapt its representations to the classification task
    without destroying the pre-trained knowledge (differential learning rates).

Evaluation metrics (on 100 held-out test sequences):
  - Top-1 accuracy   (primary metric)
  - Top-5 accuracy   (is the correct video in the top-5 predictions?)
  - Per-class accuracy
  - Confusion matrix (saved to disk)
"""

import os, sys, json, pickle, math, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).parent))
from model import TrafficGPT, Config as PretrainConfig

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Fine-tune config ──────────────────────────────────────────────────────────
class FTConfig:
    data_dir       : str   = "./data"
    ckpt_dir       : str   = "./checkpoints"
    pretrain_ckpt  : str   = "./checkpoints/best_pretrain.pt"
    num_classes    : int   = 100
    pool           : str   = "attention"  # "attention" | "mean" | "last"
    head_dropout   : float = 0.3
    label_smoothing: float = 0.1
    # phase 1 — frozen backbone
    p1_epochs      : int   = 20
    p1_lr          : float = 1e-3
    p1_batch_size  : int   = 32
    # phase 2 — full fine-tune
    p2_epochs      : int   = 100       # ← increased from 60
    p2_lr_head     : float = 1e-4
    p2_lr_backbone : float = 1e-5
    p2_batch_size  : int   = 32
    weight_decay   : float = 0.01
    grad_clip      : float = 1.0
    warmup_steps   : int   = 50
    log_every      : int   = 5
    device         : str   = "cuda" if torch.cuda.is_available() else "cpu"

FTCFG = FTConfig()


# ── Dataset ───────────────────────────────────────────────────────────────────
class ClassificationDataset(Dataset):
    def __init__(self, data_dir):
        X = np.load(f"{data_dir}/X_tokens.npy")
        y = np.load(f"{data_dir}/y_labels.npy")
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):        return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── Model ─────────────────────────────────────────────────────────────────────
# This definition MUST stay identical to the one in evaluate.py so that training
# and evaluation agree and best_finetune.pt is reproducible.
class AttentionPool(nn.Module):
    """Learned attention pooling: a per-position score → softmax → weighted sum."""
    def __init__(self, d_model):
        super().__init__()
        self.scorer = nn.Linear(d_model, 1, bias=False)

    def forward(self, content_h, valid_mask=None):
        scores = self.scorer(content_h).squeeze(-1)             # (B, T)
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, float("-inf"))
        w = torch.softmax(scores, dim=1).unsqueeze(-1)          # (B, T, 1)
        return (w * content_h).sum(dim=1)                       # (B, d)


class TrafficClassifier(nn.Module):

    def __init__(self, backbone, num_classes, pool="attention",
                 head_dropout=0.3, silent_token_id=255):
        super().__init__()
        self.backbone        = backbone
        self.pool            = pool
        self.silent_token_id = silent_token_id
        d = backbone.cfg.d_model
        self.attn_pool  = AttentionPool(d)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d),            # .0
            nn.Linear(d, d),            # .1
            nn.GELU(),                  # .2
            nn.Dropout(head_dropout),   # .3
            nn.Linear(d, num_classes),  # .4
        )

    def forward(self, input_ids):

        h = self.backbone.get_hidden_states(input_ids)   # (B, 602, d)
        content_h = h[:, 1:601, :]                        # (B, 600, d)  skip BOS/EOS

        if self.pool == "attention":
            # Attention over content tokens, silent windows masked out of the softmax.
            mask = (input_ids[:, 1:601] != self.silent_token_id)     # (B, 600) bool
            rep  = self.attn_pool(content_h, mask)                   # (B, d)
        elif self.pool == "mean":
            rep = content_h.mean(dim=1)
        else:                                             # "last" / "eos"
            rep = h[:, -1, :]

        return self.classifier(rep)

    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = True

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── LR schedule ───────────────────────────────────────────────────────────────
def get_lr(step, base_lr, warmup_steps, total_steps):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, label_smoothing=0.0):
    model.eval()
    all_logits, all_labels = [], []
    for X, y in loader:
        all_logits.append(model(X.to(device)).cpu())
        all_labels.append(y)
    model.train()

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    loss   = nn.CrossEntropyLoss(label_smoothing=label_smoothing)(logits, labels).item()
    top1   = (logits.argmax(1) == labels).float().mean().item() * 100
    top5   = (logits.topk(5, dim=1).indices == labels.unsqueeze(1)).any(1).float().mean().item() * 100
    return {"loss": loss, "top1": top1, "top5": top5,
            "logits": logits, "labels": labels}


# ── Training phase ────────────────────────────────────────────────────────────
def run_phase(model, train_loader, val_loader, device,
              n_epochs, lr_head, lr_backbone, cfg, tag):

    head_params     = list(model.classifier.parameters()) + list(model.attn_pool.parameters())
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]

    param_groups = [{"params": head_params, "lr": lr_head}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})

    optimizer   = torch.optim.AdamW(param_groups,
                                    weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    total_steps = n_epochs * len(train_loader)
    history     = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, best_state, step = 0.0, None, 0
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    print(f"\n── {tag} ({'frozen' if not backbone_params else 'full fine-tune'}) ──")
    print(f"   Trainable params  : {model.trainable_params():,}")
    print(f"   Epochs            : {n_epochs}")
    print(f"   Label smoothing   : {cfg.label_smoothing}")
    print(f"   Pooling           : {model.pool}")
    if backbone_params:
        print(f"   LR  head={lr_head:.0e}  backbone={lr_backbone:.0e}")
    else:
        print(f"   LR  head={lr_head:.0e}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        t0 = time.time()

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            new_lr = get_lr(step, lr_head, cfg.warmup_steps, total_steps)
            optimizer.param_groups[0]["lr"] = new_lr
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]["lr"] = new_lr * (lr_backbone / lr_head)

            optimizer.zero_grad()
            logits = model(X)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            ep_loss    += loss.item()
            ep_correct += (logits.argmax(1) == y).sum().item()
            ep_total   += len(y)
            step       += 1

        train_loss = ep_loss / len(train_loader)
        train_acc  = ep_correct / ep_total * 100
        val        = evaluate(model, val_loader, device, cfg.label_smoothing)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["top1"])

        if val["top1"] > best_val:
            best_val  = val["top1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % cfg.log_every == 0 or epoch == 1:
            print(f"   Epoch {epoch:3d}/{n_epochs} | "
                  f"train loss {train_loss:.4f} acc {train_acc:5.1f}% | "
                  f"val loss {val['loss']:.4f} acc {val['top1']:5.1f}% "
                  f"top5 {val['top5']:5.1f}% | {time.time()-t0:.1f}s")

    print(f"   Best val acc: {best_val:.1f}%")
    return best_state, history, best_val


# ── Main fine-tuning pipeline ─────────────────────────────────────────────────
def fine_tune(cfg=FTCFG):
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    device = torch.device(cfg.device)
    print(f"Device: {device}")

    # Load vocab — get silent token ID
    with open(f"{cfg.data_dir}/vocab.pkl", "rb") as f:
        vocab_cfg = pickle.load(f)
    silent_token_id = vocab_cfg.get("silent", vocab_cfg["k"] - 1)
    print(f"Silent token ID: {silent_token_id}")

    # Load pre-trained backbone
    ckpt   = torch.load(cfg.pretrain_ckpt, map_location="cpu", weights_only=False)
    pt_cfg = PretrainConfig()
    for k, v in ckpt["cfg"].items():
        if hasattr(pt_cfg, k): setattr(pt_cfg, k, v)

    backbone = TrafficGPT(pt_cfg)
    backbone.load_state_dict(ckpt["model_state"])
    print(f"Loaded backbone  (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f})")

    model = TrafficClassifier(
        backbone, cfg.num_classes,
        pool            = cfg.pool,
        head_dropout    = cfg.head_dropout,
        silent_token_id = silent_token_id,
    ).to(device)

    # Splits
    full_ds    = ClassificationDataset(cfg.data_dir)
    train_idx  = np.load(f"{cfg.ckpt_dir}/split_train_idx.npy")
    val_idx    = np.load(f"{cfg.ckpt_dir}/split_val_idx.npy")
    test_idx   = np.load(f"{cfg.ckpt_dir}/split_test_idx.npy")

    train_loader = DataLoader(Subset(full_ds, train_idx),
                              batch_size=cfg.p1_batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(Subset(full_ds, val_idx),
                              batch_size=cfg.p1_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(Subset(full_ds, test_idx),
                              batch_size=cfg.p1_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Dataset: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test\n")

    all_history = {}

    # Phase 1 — frozen backbone
    model.freeze_backbone()
    p1_state, p1_hist, _ = run_phase(
        model, train_loader, val_loader, device,
        cfg.p1_epochs, cfg.p1_lr, 0.0, cfg, "Phase 1")
    all_history["phase1"] = p1_hist
    model.load_state_dict(p1_state)

    # Phase 2 — full fine-tune
    model.unfreeze_backbone()
    p2_state, p2_hist, p2_best = run_phase(
        model, train_loader, val_loader, device,
        cfg.p2_epochs, cfg.p2_lr_head, cfg.p2_lr_backbone, cfg, "Phase 2")
    all_history["phase2"] = p2_hist
    model.load_state_dict(p2_state)

    # ── Test set evaluation ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    test = evaluate(model, test_loader, device)
    preds  = test["logits"].argmax(1).numpy()
    labels = test["labels"].numpy()

    print(f"Top-1 accuracy : {test['top1']:.2f}%")
    print(f"Top-5 accuracy : {test['top5']:.2f}%")
    print(f"Test loss      : {test['loss']:.4f}")

    correct = (preds == labels).sum()
    print(f"Classes correct: {correct} / {cfg.num_classes}")

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    conf = confusion_matrix(labels, preds, labels=list(range(cfg.num_classes)))
    np.save(f"{cfg.ckpt_dir}/confusion_matrix.npy", conf)

    # Save
    torch.save({
        "model_state": model.state_dict(),
        "cfg": {
            "num_classes":   cfg.num_classes,
            "pool":          cfg.pool,
            "head_dropout":  cfg.head_dropout,
            "silent_token_id": silent_token_id,
        },
        "test_top1": test["top1"],
        "test_top5": test["top5"],
    }, f"{cfg.ckpt_dir}/best_finetune.pt")

    with open(f"{cfg.ckpt_dir}/finetune_history.json", "w") as f:
        json.dump(all_history, f, indent=2)

    print(f"\nSaved: best_finetune.pt  →  {cfg.ckpt_dir}/")
    return model, test


# ── Smoke test ────────────────────────────────────────────────────────────────
def smoke_test():
    print("=" * 60)
    print("FINE-TUNING — SMOKE TEST")
    print("=" * 60)
    cfg      = PretrainConfig()
    backbone = TrafficGPT(cfg)

    for pool in ("attention", "mean"):
        model  = TrafficClassifier(backbone, 100, pool=pool, silent_token_id=255)
        dummy  = torch.randint(0, 260, (4, 602))
        logits = model(dummy)
        print(f"pool='{pool}': input {tuple(dummy.shape)} → {tuple(logits.shape)}  ✓")

    model.freeze_backbone()
    print(f"Frozen   trainable: {model.trainable_params():,}")
    model.unfreeze_backbone()
    print(f"Unfrozen trainable: {model.trainable_params():,}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_DIR = "/home2/somka34/somiya/LLM/data"
    CKPT_DIR = "/home2/somka34/somiya/LLM/checkpoints"

    if len(sys.argv) == 2 and sys.argv[1] == "test":
        smoke_test()
    else:
        FTCFG.data_dir      = DATA_DIR
        FTCFG.ckpt_dir      = CKPT_DIR
        FTCFG.pretrain_ckpt = f"{CKPT_DIR}/best_pretrain.pt"
        fine_tune(FTCFG)
