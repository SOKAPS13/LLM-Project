from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_TOKENIZED_DIR = Path("processed/packet_tokenization_vABP_offset0_8_1_1_gap5_phrase1024_len768")
DEFAULT_OUTPUT_DIR = Path("checkpoints/gpt2_abp_offset0_8_1_1_C")

PAD_TOKEN = "PAD"
UNK_TOKEN = "UNK"
BOS_TOKEN = "BOS"
EOS_TOKEN = "EOS"


@dataclass
class GPT2ABPConfig:
    vocab_size: int
    max_seq_len: int = 768
    num_classes: int = 100
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.2
    head_dropout: float = 0.3
    pad_id: int = 0


@dataclass
class TrainConfig:
    device: str = "cuda"
    pretrain_epochs: int = 50
    pretrain_batch_size: int = 16
    pretrain_lr: float = 3e-4
    pretrain_weight_decay: float = 0.1
    pretrain_warmup_steps: int = 200
    pretrain_early_stop_patience: int = 20
    pretrain_early_stop_min_delta: float = 0.0
    finetune_p1_epochs: int = 20
    finetune_p1_batch_size: int = 16
    finetune_p1_lr: float = 1e-3
    finetune_p2_epochs: int = 50
    finetune_p2_batch_size: int = 16
    finetune_p2_lr_head: float = 1e-4
    finetune_p2_lr_backbone: float = 1e-5
    finetune_weight_decay: float = 0.01
    finetune_warmup_steps: int = 50
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 5
    early_stop_patience: int = 20
    early_stop_metric: str = "top1"


def load_vocab(tokenized_dir: Path) -> dict[str, int]:
    vocab_file = tokenized_dir / "vocab.json"
    if not vocab_file.exists():
        raise FileNotFoundError(f"Missing vocab file: {vocab_file}")
    with vocab_file.open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    for token in [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]:
        if token not in vocab:
            raise ValueError(f"vocab.json is missing required token: {token}")
    return {str(key): int(value) for key, value in vocab.items()}


def load_split(tokenized_dir: Path, split: str) -> pd.DataFrame:
    import pandas as pd

    parquet_file = tokenized_dir / f"{split}.parquet"
    jsonl_file = tokenized_dir / f"{split}.jsonl"
    if parquet_file.exists():
        df = pd.read_parquet(parquet_file)
    elif jsonl_file.exists():
        df = pd.read_json(jsonl_file, orient="records", lines=True)
    else:
        raise FileNotFoundError(f"Missing {split}.parquet or {split}.jsonl in {tokenized_dir}")

    required = {"tokens", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{split} split is missing columns: {sorted(missing)}")
    return df


def pad_or_truncate(tokens: list[int], max_seq_len: int, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    ids = [int(token) for token in tokens[:max_seq_len]]
    mask = [1] * len(ids)
    if len(ids) < max_seq_len:
        pad_count = max_seq_len - len(ids)
        ids.extend([pad_id] * pad_count)
        mask.extend([0] * pad_count)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


class ABPSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_seq_len: int, pad_id: int):
        self.tokens = df["tokens"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        input_ids, attention_mask = pad_or_truncate(
            self.tokens[index],
            self.max_seq_len,
            self.pad_id,
        )
        return input_ids, attention_mask, torch.tensor(self.labels[index], dtype=torch.long)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPT2ABPConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, width = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        key_mask = attention_mask[:, None, None, :]
        scores = scores.masked_fill(~causal_mask[None, None, :, :], torch.finfo(scores.dtype).min)
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        y = weights @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, width)
        return self.resid_dropout(self.proj(y))


class GPT2Block(nn.Module):
    def __init__(self, cfg: GPT2ABPConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Backbone(nn.Module):
    def __init__(self, cfg: GPT2ABPConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([GPT2Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.cfg.max_seq_len}")
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, attention_mask)
        return self.ln_f(x)


class GPT2ForABP(nn.Module):
    def __init__(self, cfg: GPT2ABPConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = GPT2Backbone(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.backbone.token_embedding.weight
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.d_model, cfg.num_classes),
        )

    def forward_lm(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids, attention_mask)
        return self.lm_head(hidden)

    def forward_classifier(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids, attention_mask)
        lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
        return self.classifier(pooled)

    def trainable_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)


def get_lr(step: int, base_lr: float, warmup_steps: int, total_steps: int) -> float:
    if total_steps <= 0:
        return base_lr
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate_lm(model: GPT2ForABP, loader: DataLoader, device: torch.device, pad_id: int) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for input_ids, attention_mask, _ in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        logits = model.forward_lm(input_ids[:, :-1], attention_mask[:, :-1])
        labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=pad_id)
        total_loss += loss.item()
        total_batches += 1
    loss = total_loss / max(1, total_batches)
    return {"loss": loss, "ppl": math.exp(min(loss, 20.0))}


@torch.no_grad()
def evaluate_classifier(
    model: GPT2ForABP,
    loader: DataLoader,
    device: torch.device,
    label_smoothing: float,
) -> dict[str, float]:
    model.eval()
    logits_all = []
    labels_all = []
    for input_ids, attention_mask, labels in loader:
        logits = model.forward_classifier(input_ids.to(device), attention_mask.to(device))
        logits_all.append(logits.cpu())
        labels_all.append(labels)
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)(logits, labels).item()
    top1 = (logits.argmax(dim=1) == labels).float().mean().item() * 100
    topk = min(5, logits.size(1))
    top5 = (logits.topk(topk, dim=1).indices == labels.unsqueeze(1)).any(dim=1).float().mean().item() * 100
    return {"loss": loss, "top1": top1, "top5": top5}


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def pretrain(
    model: GPT2ForABP,
    train_ds: Dataset,
    val_ds: Dataset,
    output_dir: Path,
    cfg: TrainConfig,
    pad_id: int,
    num_workers: int,
) -> dict:
    train_loader = make_loader(train_ds, cfg.pretrain_batch_size, True, num_workers)
    val_loader = make_loader(val_ds, cfg.pretrain_batch_size, False, num_workers)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.pretrain_lr,
        weight_decay=cfg.pretrain_weight_decay,
        betas=(0.9, 0.95),
    )
    total_steps = cfg.pretrain_epochs * len(train_loader)
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = {"train_loss": [], "val_loss": [], "val_ppl": [], "lr": []}
    step = 0

    print("\n-- GPT-2 ABP pretraining --")
    print(f"   Trainable params: {model.trainable_params():,}")
    print(
        "   Early stopping  : "
        f"metric=val_loss, patience={cfg.pretrain_early_stop_patience}, "
        f"min_delta={cfg.pretrain_early_stop_min_delta}"
    )
    for epoch in range(1, cfg.pretrain_epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for input_ids, attention_mask, _ in train_loader:
            current_lr = get_lr(step, cfg.pretrain_lr, cfg.pretrain_warmup_steps, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model.forward_lm(input_ids[:, :-1], attention_mask[:, :-1])
            labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=pad_id)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_loss += loss.item()
            step += 1

        train_loss /= max(1, len(train_loader))
        val = evaluate_lm(model, val_loader, device, pad_id)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_ppl"].append(val["ppl"])
        history["lr"].append(current_lr)

        if val["loss"] < best_val - cfg.pretrain_early_stop_min_delta:
            best_val = val["loss"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": asdict(model.cfg),
                    "train_config": asdict(cfg),
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                    "epoch": epoch,
                },
                output_dir / "best_pretrain.pt",
            )
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % cfg.log_every == 0:
            print(
                f"   Epoch {epoch:3d}/{cfg.pretrain_epochs} | "
                f"train loss {train_loss:.4f} | val loss {val['loss']:.4f} "
                f"ppl {val['ppl']:.1f} | stale {stale_epochs:2d}/"
                f"{cfg.pretrain_early_stop_patience} | {time.time() - t0:.1f}s"
            )

        if stale_epochs >= cfg.pretrain_early_stop_patience:
            print(
                f"   Early stop at epoch {epoch}; "
                f"best val loss={best_val:.4f} at epoch {best_epoch}"
            )
            break

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val
    history["stopped_epoch"] = len(history["train_loss"])
    with (output_dir / "pretrain_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"   Best pretrain val loss: {best_val:.4f} at epoch {best_epoch}")
    return history


def set_backbone_trainable(model: GPT2ForABP, trainable: bool) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = trainable
    for param in model.lm_head.parameters():
        param.requires_grad = trainable
    for param in model.classifier.parameters():
        param.requires_grad = True


def run_finetune_phase(
    model: GPT2ForABP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int,
    lr_head: float,
    lr_backbone: float,
    cfg: TrainConfig,
    tag: str,
) -> tuple[dict[str, torch.Tensor], dict, float]:
    head_params = [param for param in model.classifier.parameters() if param.requires_grad]
    backbone_params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and not name.startswith("classifier.")
    ]
    param_groups = [{"params": head_params, "lr": lr_head}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.finetune_weight_decay, betas=(0.9, 0.95))
    total_steps = n_epochs * len(train_loader)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    best_score = -1.0
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    step = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_top1": [], "val_top5": []}

    print(f"\n-- {tag} --")
    print(f"   Trainable params: {model.trainable_params():,}")
    for epoch in range(1, n_epochs + 1):
        model.train()
        t0 = time.time()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        for input_ids, attention_mask, labels in train_loader:
            current_lr = get_lr(step, lr_head, cfg.finetune_warmup_steps, total_steps)
            optimizer.param_groups[0]["lr"] = current_lr
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]["lr"] = current_lr * (lr_backbone / lr_head)

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            logits = model.forward_classifier(input_ids, attention_mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            ep_loss += loss.item()
            ep_correct += (logits.argmax(dim=1) == labels).sum().item()
            ep_total += labels.size(0)
            step += 1

        train_loss = ep_loss / max(1, len(train_loader))
        train_acc = ep_correct / max(1, ep_total) * 100
        val = evaluate_classifier(model, val_loader, device, cfg.label_smoothing)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val["loss"])
        history["val_top1"].append(val["top1"])
        history["val_top5"].append(val["top5"])

        score = val["top5"] if cfg.early_stop_metric == "top5" else val["top1"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % cfg.log_every == 0:
            print(
                f"   Epoch {epoch:3d}/{n_epochs} | train loss {train_loss:.4f} "
                f"acc {train_acc:5.1f}% | val loss {val['loss']:.4f} "
                f"top1 {val['top1']:5.1f}% top5 {val['top5']:5.1f}% | "
                f"stale {stale_epochs:2d}/{cfg.early_stop_patience} | {time.time() - t0:.1f}s"
            )

        if stale_epochs >= cfg.early_stop_patience:
            print(f"   Early stop at epoch {epoch}; best epoch={best_epoch}, score={best_score:.1f}%")
            break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history["best_epoch"] = best_epoch
    history["best_score"] = best_score
    return best_state, history, best_score


def finetune(
    model: GPT2ForABP,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    output_dir: Path,
    cfg: TrainConfig,
    pretrain_ckpt: Path | None,
    num_workers: int,
) -> dict:
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    if pretrain_ckpt is not None:
        checkpoint = torch.load(pretrain_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"], strict=False)
    model.to(device)

    p1_train_loader = make_loader(train_ds, cfg.finetune_p1_batch_size, True, num_workers)
    p1_val_loader = make_loader(val_ds, cfg.finetune_p1_batch_size, False, num_workers)
    p2_train_loader = make_loader(train_ds, cfg.finetune_p2_batch_size, True, num_workers)
    p2_val_loader = make_loader(val_ds, cfg.finetune_p2_batch_size, False, num_workers)
    test_loader = make_loader(test_ds, cfg.finetune_p2_batch_size, False, num_workers)

    histories = {}
    set_backbone_trainable(model, False)
    p1_state, p1_history, _ = run_finetune_phase(
        model,
        p1_train_loader,
        p1_val_loader,
        device,
        cfg.finetune_p1_epochs,
        cfg.finetune_p1_lr,
        0.0,
        cfg,
        "Phase 1: frozen GPT-2 backbone, classifier head only",
    )
    model.load_state_dict(p1_state)
    histories["phase1"] = p1_history

    set_backbone_trainable(model, True)
    p2_state, p2_history, _ = run_finetune_phase(
        model,
        p2_train_loader,
        p2_val_loader,
        device,
        cfg.finetune_p2_epochs,
        cfg.finetune_p2_lr_head,
        cfg.finetune_p2_lr_backbone,
        cfg,
        "Phase 2: full GPT-2 fine-tune",
    )
    model.load_state_dict(p2_state)
    histories["phase2"] = p2_history

    test = evaluate_classifier(model, test_loader, device, cfg.label_smoothing)
    summary = {
        "test_top1": test["top1"],
        "test_top5": test["top5"],
        "test_loss": test["loss"],
        "best_phase1_epoch": histories["phase1"].get("best_epoch"),
        "best_phase1_score": histories["phase1"].get("best_score"),
        "best_phase2_epoch": histories["phase2"].get("best_epoch"),
        "best_phase2_score": histories["phase2"].get("best_score"),
        "early_stop_metric": cfg.early_stop_metric,
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model.cfg),
            "train_config": asdict(cfg),
            "test": test,
            "summary": summary,
        },
        output_dir / "best_finetune.pt",
    )
    metrics = {"history": histories, "test": test, "summary": summary}
    with (output_dir / "finetune_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n-- GPT-2 ABP test --")
    print(f"   Top-1: {test['top1']:.2f}% | Top-5: {test['top5']:.2f}% | Loss: {test['loss']:.4f}")
    print(f"   Saved metrics: {output_dir / 'summary_metrics.json'}")
    return metrics


def validate_tokenized_dir(tokenized_dir: Path) -> None:
    missing = [
        name
        for name in ["train.parquet", "val.parquet", "test.parquet", "vocab.json"]
        if not (tokenized_dir / name).exists()
    ]
    jsonl_missing = [
        name.replace(".parquet", ".jsonl")
        for name in missing
        if name.endswith(".parquet") and not (tokenized_dir / name.replace(".parquet", ".jsonl")).exists()
    ]
    if "vocab.json" in missing or jsonl_missing:
        raise FileNotFoundError(
            f"Tokenized directory is incomplete: {tokenized_dir}\n"
            f"Missing: {', '.join(missing)}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized-dir", default=str(DEFAULT_TOKENIZED_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stage", choices=["pretrain", "finetune", "all"], default="all")
    parser.add_argument("--pretrain-ckpt", default=None)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--head-dropout", type=float, default=0.3)

    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--pretrain-batch-size", type=int, default=16)
    parser.add_argument("--pretrain-lr", type=float, default=3e-4)
    parser.add_argument("--pretrain-weight-decay", type=float, default=0.1)
    parser.add_argument("--pretrain-warmup-steps", type=int, default=200)
    parser.add_argument("--pretrain-early-stop-patience", type=int, default=20)
    parser.add_argument("--pretrain-early-stop-min-delta", type=float, default=0.0)

    parser.add_argument("--p1-epochs", type=int, default=20)
    parser.add_argument("--p1-batch-size", type=int, default=16)
    parser.add_argument("--p1-lr", type=float, default=1e-3)
    parser.add_argument("--p2-epochs", type=int, default=50)
    parser.add_argument("--p2-batch-size", type=int, default=16)
    parser.add_argument("--p2-lr-head", type=float, default=1e-4)
    parser.add_argument("--p2-lr-backbone", type=float, default=1e-5)

    parser.add_argument("--finetune-weight-decay", type=float, default=0.01)
    parser.add_argument("--finetune-warmup-steps", type=int, default=50)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--early-stop-metric", choices=["top1", "top5"], default="top1")
    return parser.parse_args()


def main(args) -> None:
    tokenized_dir = Path(args.tokenized_dir)
    output_dir = Path(args.output_dir)
    validate_tokenized_dir(tokenized_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab(tokenized_dir)
    pad_id = vocab[PAD_TOKEN]
    model_cfg = GPT2ABPConfig(
        vocab_size=max(vocab.values()) + 1,
        max_seq_len=args.max_seq_len,
        num_classes=args.num_classes,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        pad_id=pad_id,
    )
    train_cfg = TrainConfig(
        device=args.device,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_batch_size=args.pretrain_batch_size,
        pretrain_lr=args.pretrain_lr,
        pretrain_weight_decay=args.pretrain_weight_decay,
        pretrain_warmup_steps=args.pretrain_warmup_steps,
        pretrain_early_stop_patience=args.pretrain_early_stop_patience,
        pretrain_early_stop_min_delta=args.pretrain_early_stop_min_delta,
        finetune_p1_epochs=args.p1_epochs,
        finetune_p1_batch_size=args.p1_batch_size,
        finetune_p1_lr=args.p1_lr,
        finetune_p2_epochs=args.p2_epochs,
        finetune_p2_batch_size=args.p2_batch_size,
        finetune_p2_lr_head=args.p2_lr_head,
        finetune_p2_lr_backbone=args.p2_lr_backbone,
        finetune_weight_decay=args.finetune_weight_decay,
        finetune_warmup_steps=args.finetune_warmup_steps,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        early_stop_patience=args.early_stop_patience,
        early_stop_metric=args.early_stop_metric,
    )

    train_df = load_split(tokenized_dir, "train")
    val_df = load_split(tokenized_dir, "val")
    test_df = load_split(tokenized_dir, "test")
    train_ds = ABPSequenceDataset(train_df, args.max_seq_len, pad_id)
    val_ds = ABPSequenceDataset(val_df, args.max_seq_len, pad_id)
    test_ds = ABPSequenceDataset(test_df, args.max_seq_len, pad_id)

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "tokenized_dir": str(tokenized_dir),
                "model_config": asdict(model_cfg),
                "train_config": asdict(train_cfg),
                "rows": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
            },
            f,
            indent=2,
        )

    model = GPT2ForABP(model_cfg)
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Tokenized dir: {tokenized_dir}")
    print(f"Output dir   : {output_dir}")
    print(f"Rows         : {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")
    print(f"Vocab size   : {model_cfg.vocab_size}")
    print(f"GPT-2 params : {total_params:,}")

    if args.stage in {"pretrain", "all"}:
        pretrain(model, train_ds, val_ds, output_dir, train_cfg, pad_id, args.num_workers)

    if args.stage in {"finetune", "all"}:
        pretrain_ckpt = Path(args.pretrain_ckpt) if args.pretrain_ckpt else output_dir / "best_pretrain.pt"
        if args.stage == "finetune" and not pretrain_ckpt.exists():
            raise FileNotFoundError(f"Pretrain checkpoint not found: {pretrain_ckpt}")
        if args.stage == "all":
            pretrain_ckpt = output_dir / "best_pretrain.pt"
        finetune(model, train_ds, val_ds, test_ds, output_dir, train_cfg, pretrain_ckpt, args.num_workers)


if __name__ == "__main__":
    main(parse_args())
