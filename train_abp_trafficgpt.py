from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn

import trafficgpt_tokenizationC as base


DEFAULT_TOKENIZED_DIR = Path("processed/packet_tokenization_vABP_offset0_8_1_1_gap5_phrase1024_len768")
DEFAULT_OUTPUT_DIR = Path("checkpoints/trafficgpt_abp_offset0_8_1_1_C_small_earlystop")


def _jsonable(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def save_summary_metrics(output_dir: Path, tokenized_dir: Path, args) -> None:
    checkpoint_file = output_dir / "best_finetune.pt"
    summary_file = output_dir / "summary_metrics.json"
    metrics = {
        "tokenized_dir": str(tokenized_dir),
        "output_dir": str(output_dir),
        "stage": args.stage,
        "max_seq_len": args.max_seq_len,
        "early_stop_metric": args.early_stop_metric,
        "early_stop_patience": args.early_stop_patience,
    }

    if not checkpoint_file.exists():
        metrics["warning"] = f"Missing checkpoint: {checkpoint_file}"
    else:
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        checkpoint = _jsonable(checkpoint)
        if "test_top1" in checkpoint:
            metrics["test_top1"] = checkpoint["test_top1"]
        if "test_top5" in checkpoint:
            metrics["test_top5"] = checkpoint["test_top5"]
        if "test" in checkpoint and isinstance(checkpoint["test"], dict):
            test = checkpoint["test"]
            if "top1" in test:
                metrics["test_top1"] = test["top1"]
            if "top5" in test:
                metrics["test_top5"] = test["top5"]
            if "loss" in test:
                metrics["test_loss"] = test["loss"]
        if "best_val_top1" in checkpoint:
            metrics["best_val_top1"] = checkpoint["best_val_top1"]
        if "best_val_top5" in checkpoint:
            metrics["best_val_top5"] = checkpoint["best_val_top5"]
        if "test_top5" not in metrics:
            metrics["warning"] = "Checkpoint did not contain test_top5."

    output_dir.mkdir(parents=True, exist_ok=True)
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved summary metrics to: {summary_file}")


def validate_tokenized_dir(tokenized_dir: Path) -> None:
    required = ["train.parquet", "val.parquet", "test.parquet", "vocab.json"]
    missing = [name for name in required if not (tokenized_dir / name).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"Tokenized directory is not ready: {tokenized_dir}\n"
            f"Missing: {missing_text}\n\n"
            "For offset 0 ABP C, create it with:\n"
            "  python \"New Meothd\\make_offset0_split.py\"\n"
            "  python \"New Meothd\\abp_tokenization.py\" "
            "--index-file metadata/trace_index_offset0_8_1_1.csv "
            "--burst-gap-ms 5 --max-phrases 1024 --max-seq-len 768 "
            "--output-dir processed/packet_tokenization_vABP_offset0_8_1_1_gap5_phrase1024_len768"
        )


def run_ft_phase_earlystop(
    model,
    train_loader,
    val_loader,
    device,
    n_epochs,
    lr_head,
    lr_backbone,
    cfg,
    tag,
):
    head_params = list(model.classifier.parameters())
    backbone_params = [param for param in model.backbone.parameters() if param.requires_grad]
    param_groups = [{"params": head_params, "lr": lr_head}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    total_steps = n_epochs * len(train_loader)
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_top5": [],
        "lr_head": [],
    }
    best_score = -1.0
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    step = 0
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    patience = getattr(cfg, "early_stop_patience", 20)
    min_delta = getattr(cfg, "early_stop_min_delta", 0.0)
    metric = getattr(cfg, "early_stop_metric", "top1")

    print(f"\n-- {tag} --")
    print(f"   Trainable params: {model.trainable_params():,}")
    print(f"   Early stopping  : metric={metric}, patience={patience}, min_delta={min_delta}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        t0 = time.time()
        current_lr = lr_head

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            current_lr = base.get_ft_lr(step, lr_head, cfg.warmup_steps, total_steps)
            optimizer.param_groups[0]["lr"] = current_lr
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]["lr"] = current_lr * (lr_backbone / lr_head)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            ep_loss += loss.item()
            ep_correct += (logits.argmax(1) == labels).sum().item()
            ep_total += len(labels)
            step += 1

        train_loss = ep_loss / len(train_loader)
        train_acc = ep_correct / ep_total * 100
        val = base.evaluate_classifier(model, val_loader, device, cfg.label_smoothing)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["top1"])
        history["val_top5"].append(val["top5"])
        history["lr_head"].append(current_lr)

        score = val["top5"] if metric == "top5" else val["top1"]
        if score > best_score + min_delta:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1

        if epoch % cfg.log_every == 0 or epoch == 1:
            print(
                f"   Epoch {epoch:3d}/{n_epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:5.1f}% | "
                f"val loss {val['loss']:.4f} acc {val['top1']:5.1f}% "
                f"top5 {val['top5']:5.1f}% | stale {stale_epochs:2d}/{patience} | "
                f"{time.time() - t0:.1f}s"
            )

        if stale_epochs >= patience:
            print(f"   Early stop at epoch {epoch}; best {metric}={best_score:.1f}% at epoch {best_epoch}")
            break

    if best_state is None:
        best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
    history["best_epoch"] = best_epoch
    history["best_metric"] = metric
    history["best_score"] = best_score
    history["stopped_epoch"] = len(history["train_loss"])
    print(f"   Best val {metric}: {best_score:.1f}% at epoch {best_epoch}")
    return best_state, history, best_score


def build_pretrain_config(args) -> base.CConfig:
    cfg = base.CConfig()
    cfg.max_seq_len = args.max_seq_len
    cfg.epochs = args.pretrain_epochs
    cfg.batch_size = args.pretrain_batch_size
    cfg.lr = args.pretrain_lr
    cfg.weight_decay = args.pretrain_weight_decay
    cfg.grad_clip = args.grad_clip
    cfg.warmup_steps = args.pretrain_warmup_steps
    cfg.log_every = args.log_every
    cfg.save_every = args.save_every
    cfg.d_model = args.d_model
    cfg.n_layers = args.n_layers
    cfg.n_heads = args.n_heads
    cfg.d_ff = args.d_ff
    cfg.dropout = args.dropout
    cfg.device = args.device
    return cfg


def build_finetune_config(args) -> base.CFTConfig:
    cfg = base.CFTConfig()
    cfg.num_classes = args.num_classes
    cfg.head_dropout = args.head_dropout
    cfg.label_smoothing = args.label_smoothing
    cfg.p1_epochs = args.p1_epochs
    cfg.p1_lr = args.p1_lr
    cfg.p1_batch_size = args.p1_batch_size
    cfg.p2_epochs = args.p2_epochs
    cfg.p2_lr_head = args.p2_lr_head
    cfg.p2_lr_backbone = args.p2_lr_backbone
    cfg.p2_batch_size = args.p2_batch_size
    cfg.weight_decay = args.finetune_weight_decay
    cfg.grad_clip = args.grad_clip
    cfg.warmup_steps = args.finetune_warmup_steps
    cfg.log_every = args.log_every
    cfg.device = args.device
    cfg.early_stop_patience = args.early_stop_patience
    cfg.early_stop_min_delta = args.early_stop_min_delta
    cfg.early_stop_metric = args.early_stop_metric
    return cfg


def main(args) -> None:
    base.run_ft_phase = run_ft_phase_earlystop

    tokenized_dir = Path(args.tokenized_dir)
    output_dir = Path(args.output_dir)
    validate_tokenized_dir(tokenized_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pretrain_cfg = build_pretrain_config(args)
    ft_cfg = build_finetune_config(args)

    if args.stage in {"pretrain", "all"}:
        base.pretrain(tokenized_dir, output_dir, pretrain_cfg)

    if args.stage in {"finetune", "all"}:
        ckpt = Path(args.pretrain_ckpt) if args.pretrain_ckpt else output_dir / "best_pretrain.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Pretrain checkpoint not found: {ckpt}")
        base.finetune(tokenized_dir, output_dir, ckpt, ft_cfg)
        save_summary_metrics(output_dir, tokenized_dir, args)
    elif args.stage == "summarize":
        save_summary_metrics(output_dir, tokenized_dir, args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized-dir", default=str(DEFAULT_TOKENIZED_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stage", choices=["pretrain", "finetune", "all", "summarize"], default="all")
    parser.add_argument("--pretrain-ckpt", default=None)

    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seq-len", type=int, default=768)

    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.25)

    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--pretrain-batch-size", type=int, default=16)
    parser.add_argument("--pretrain-lr", type=float, default=3e-4)
    parser.add_argument("--pretrain-weight-decay", type=float, default=0.1)
    parser.add_argument("--pretrain-warmup-steps", type=int, default=200)

    parser.add_argument("--p1-epochs", type=int, default=50)
    parser.add_argument("--p1-batch-size", type=int, default=16)
    parser.add_argument("--p1-lr", type=float, default=1e-3)

    parser.add_argument("--p2-epochs", type=int, default=100)
    parser.add_argument("--p2-batch-size", type=int, default=16)
    parser.add_argument("--p2-lr-head", type=float, default=1e-4)
    parser.add_argument("--p2-lr-backbone", type=float, default=1e-5)

    parser.add_argument("--finetune-weight-decay", type=float, default=0.02)
    parser.add_argument("--finetune-warmup-steps", type=int, default=50)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--head-dropout", type=float, default=0.35)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=25)

    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--early-stop-metric", choices=["top1", "top5"], default="top1")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
