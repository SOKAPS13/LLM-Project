from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


INDEX_FILE = Path("metadata/trace_index_with_split_larger.csv")
OUTPUT_DIR = Path("processed/packet_tokenization_vABP_ack_burst_phrase")

TRACE_DURATION_NS = 60_000_000_000
BURST_GAP_MS = 10.0
MAX_SEQ_LEN = 512
MAX_PHRASES = 512
MIN_PHRASE_COUNT = 3
QUANTILE_LEVELS = [0.2, 0.4, 0.6, 0.8]

PAD_TOKEN = "PAD"
UNK_TOKEN = "UNK"
BOS_TOKEN = "BOS"
EOS_TOKEN = "EOS"


def normalize_dir(value: str) -> str:
    value = str(value).strip().lower()
    if value == "s":
        return "S"
    if value == "r":
        return "R"
    return "U"


def parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def log_value(value: float) -> float:
    return math.log10(max(float(value), 0.0) + 1.0)


def iter_packets(file_path: Path):
    previous_epoch_ms = None
    with file_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                ts = int(row[0])
                direction = normalize_dir(row[1])
                length = int(row[2])
            except ValueError:
                continue
            if ts < 0:
                continue
            if ts >= TRACE_DURATION_NS:
                break
            if length < 0:
                continue

            epoch_ms = parse_optional_int(row[3]) if len(row) > 3 else None
            tcp_seq = parse_optional_int(row[4]) if len(row) > 4 else None
            tcp_ack = parse_optional_int(row[5]) if len(row) > 5 else None
            epoch_delta_ms = None
            if epoch_ms is not None and previous_epoch_ms is not None:
                epoch_delta_ms = max(0, epoch_ms - previous_epoch_ms)
            if epoch_ms is not None:
                previous_epoch_ms = epoch_ms

            yield {
                "ts": ts,
                "direction": direction,
                "length": length,
                "epoch_delta_ms": epoch_delta_ms,
                "tcp_seq": tcp_seq,
                "tcp_ack": tcp_ack,
            }


def start_burst(pkt: dict) -> dict:
    return {
        "direction": pkt["direction"],
        "start_ns": pkt["ts"],
        "end_ns": pkt["ts"],
        "total_bytes": int(pkt["length"]),
        "packet_count": 1,
        "first_seq": pkt["tcp_seq"],
        "last_seq": pkt["tcp_seq"],
        "first_ack": pkt["tcp_ack"],
        "last_ack": pkt["tcp_ack"],
        "epoch_delta_sum_ms": int(pkt["epoch_delta_ms"] or 0),
    }


def update_burst(burst: dict, pkt: dict) -> None:
    burst["end_ns"] = pkt["ts"]
    burst["total_bytes"] += int(pkt["length"])
    burst["packet_count"] += 1
    if pkt["tcp_seq"] is not None:
        if burst["first_seq"] is None:
            burst["first_seq"] = pkt["tcp_seq"]
        burst["last_seq"] = pkt["tcp_seq"]
    if pkt["tcp_ack"] is not None:
        if burst["first_ack"] is None:
            burst["first_ack"] = pkt["tcp_ack"]
        burst["last_ack"] = pkt["tcp_ack"]
    burst["epoch_delta_sum_ms"] += int(pkt["epoch_delta_ms"] or 0)


def finalize_bursts(bursts: list[dict]) -> list[dict]:
    prev_end = None
    for burst in bursts:
        duration_ms = max(0.0, (burst["end_ns"] - burst["start_ns"]) / 1_000_000.0)
        if prev_end is None:
            gap_ms = None
        else:
            gap_ms = max(0.0, (burst["start_ns"] - prev_end) / 1_000_000.0)
        seq_progress = 0
        if burst["first_seq"] is not None and burst["last_seq"] is not None:
            seq_progress = max(0, int(burst["last_seq"]) - int(burst["first_seq"]))
        ack_progress = 0
        if burst["first_ack"] is not None and burst["last_ack"] is not None:
            ack_progress = max(0, int(burst["last_ack"]) - int(burst["first_ack"]))

        burst["duration_ms"] = duration_ms
        burst["gap_ms"] = gap_ms
        burst["avg_len"] = burst["total_bytes"] / max(1, burst["packet_count"])
        burst["seq_progress"] = seq_progress
        burst["ack_progress"] = ack_progress
        burst["position_bin"] = position_bin(burst["start_ns"])
        prev_end = burst["end_ns"]
    return bursts


def parse_bursts(file_path: Path, burst_gap_ms: float = BURST_GAP_MS) -> list[dict]:
    gap_ns = int(burst_gap_ms * 1_000_000)
    bursts = []
    current = None
    prev_packet_ts = None

    for pkt in iter_packets(file_path):
        starts_new = (
            current is None
            or pkt["direction"] != current["direction"]
            or (prev_packet_ts is not None and pkt["ts"] - prev_packet_ts > gap_ns)
        )
        if starts_new:
            if current is not None:
                bursts.append(current)
            current = start_burst(pkt)
        else:
            update_burst(current, pkt)
        prev_packet_ts = pkt["ts"]

    if current is not None:
        bursts.append(current)
    return finalize_bursts(bursts)


def position_bin(time_relative_ns: int) -> str:
    ratio = min(max(float(time_relative_ns) / TRACE_DURATION_NS, 0.0), 0.999999)
    return f"POS_Q{int(ratio * 5) + 1}"


def transformed_feature(name: str, value: float | None) -> float:
    if value is None:
        return 0.0
    if name in {
        "total_bytes",
        "packet_count",
        "duration_ms",
        "gap_ms",
        "seq_progress",
        "ack_progress",
        "epoch_delta_sum_ms",
    }:
        return log_value(float(value))
    return float(value)


def collect_training_values(index_df: pd.DataFrame, burst_gap_ms: float) -> dict[str, np.ndarray]:
    values = {
        "total_bytes": [],
        "packet_count": [],
        "duration_ms": [],
        "gap_ms": [],
        "avg_len": [],
        "seq_progress": [],
        "ack_progress": [],
        "epoch_delta_sum_ms": [],
    }
    train_df = index_df[index_df["split"] == "train"].copy()

    for n, (_, row) in enumerate(train_df.iterrows(), start=1):
        file_path = Path(row["file_path"])
        if not file_path.exists():
            print(f"Warning: missing file, skip threshold collection: {file_path}")
            continue
        for burst in parse_bursts(file_path, burst_gap_ms):
            for key in values:
                if key == "gap_ms" and burst[key] is None:
                    continue
                values[key].append(transformed_feature(key, burst[key]))
        if n % 200 == 0:
            print(f"Collected ABP values from {n} train traces...")

    output = {}
    for key, vals in values.items():
        if not vals:
            raise ValueError(f"No training values collected for {key}")
        output[key] = np.asarray(vals, dtype=np.float32)
    return output


def compute_thresholds(values: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {
        key: np.quantile(arr, QUANTILE_LEVELS).astype(float).tolist()
        for key, arr in values.items()
    }


def qbin(value: float, thresholds: list[float], prefix: str) -> str:
    if value <= thresholds[0]:
        return f"{prefix}_Q1"
    if value <= thresholds[1]:
        return f"{prefix}_Q2"
    if value <= thresholds[2]:
        return f"{prefix}_Q3"
    if value <= thresholds[3]:
        return f"{prefix}_Q4"
    return f"{prefix}_Q5"


def progress_bin(value: float, thresholds: list[float], prefix: str) -> str:
    if value <= 0:
        return f"{prefix}_SAME"
    return qbin(value, thresholds, prefix)


def gap_bin(value: float | None, thresholds: list[float]) -> str:
    if value is None:
        return "GAP_START"
    return qbin(transformed_feature("gap_ms", value), thresholds, "GAP")


def burst_to_base_token(burst: dict, thresholds: dict[str, list[float]]) -> str:
    direction = burst["direction"] if burst["direction"] in {"S", "R"} else "U"
    vol = qbin(
        transformed_feature("total_bytes", burst["total_bytes"]),
        thresholds["total_bytes"],
        "VOL",
    )
    pkts = qbin(
        transformed_feature("packet_count", burst["packet_count"]),
        thresholds["packet_count"],
        "PKT",
    )
    dur = qbin(
        transformed_feature("duration_ms", burst["duration_ms"]),
        thresholds["duration_ms"],
        "DUR",
    )
    gap = gap_bin(burst["gap_ms"], thresholds["gap_ms"])
    avg = qbin(
        transformed_feature("avg_len", burst["avg_len"]),
        thresholds["avg_len"],
        "AVG",
    )
    seq = progress_bin(
        transformed_feature("seq_progress", burst["seq_progress"]),
        thresholds["seq_progress"],
        "SEQ",
    )
    ack = progress_bin(
        transformed_feature("ack_progress", burst["ack_progress"]),
        thresholds["ack_progress"],
        "ACK",
    )
    epoch = progress_bin(
        transformed_feature("epoch_delta_sum_ms", burst["epoch_delta_sum_ms"]),
        thresholds["epoch_delta_sum_ms"],
        "EPOCH",
    )
    pos = burst["position_bin"]
    return f"B_{direction}_{vol}_{pkts}_{dur}_{gap}_{avg}_{seq}_{ack}_{epoch}_{pos}"


def base_tokens_for_trace(
    file_path: Path,
    thresholds: dict[str, list[float]],
    burst_gap_ms: float,
) -> list[str]:
    return [
        burst_to_base_token(burst, thresholds)
        for burst in parse_bursts(file_path, burst_gap_ms)
    ]


def collect_training_base_sequences(
    index_df: pd.DataFrame,
    thresholds: dict[str, list[float]],
    burst_gap_ms: float,
) -> list[list[str]]:
    sequences = []
    train_df = index_df[index_df["split"] == "train"].copy()
    for n, (_, row) in enumerate(train_df.iterrows(), start=1):
        file_path = Path(row["file_path"])
        if not file_path.exists():
            continue
        sequences.append(base_tokens_for_trace(file_path, thresholds, burst_gap_ms))
        if n % 200 == 0:
            print(f"Collected ABP base tokens from {n} train traces...")
    return sequences


def learn_phrase_map(
    train_sequences: list[list[str]],
    max_phrases: int,
    min_phrase_count: int,
) -> dict[tuple[str, str], str]:
    pair_counts = Counter()
    for seq in train_sequences:
        pair_counts.update(zip(seq, seq[1:]))

    phrase_map = {}
    for (left, right), count in pair_counts.most_common(max_phrases):
        if count < min_phrase_count:
            break
        phrase_map[(left, right)] = f"PH_{len(phrase_map):04d}"
    return phrase_map


def apply_phrase_merge(tokens: list[str], phrase_map: dict[tuple[str, str], str]) -> list[str]:
    merged = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) in phrase_map:
            merged.append(phrase_map[(tokens[i], tokens[i + 1])])
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


def build_vocab(
    train_sequences: list[list[str]],
    phrase_map: dict[tuple[str, str], str],
) -> dict[str, int]:
    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
        BOS_TOKEN: 2,
        EOS_TOKEN: 3,
    }
    token_counts = Counter()
    for seq in train_sequences:
        token_counts.update(seq)
        token_counts.update(apply_phrase_merge(seq, phrase_map))

    for token, _ in token_counts.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)
    for phrase in phrase_map.values():
        if phrase not in vocab:
            vocab[phrase] = len(vocab)
    return vocab


def tokenize_trace(
    file_path: Path,
    thresholds: dict[str, list[float]],
    phrase_map: dict[tuple[str, str], str],
    vocab: dict[str, int],
    burst_gap_ms: float,
    max_seq_len: int,
) -> tuple[list[int], int, int]:
    base_tokens = base_tokens_for_trace(file_path, thresholds, burst_gap_ms)
    merged_tokens = apply_phrase_merge(base_tokens, phrase_map)
    keep = max(0, max_seq_len - 2)
    merged_tokens = merged_tokens[:keep]
    token_ids = [vocab[BOS_TOKEN]]
    token_ids.extend(vocab.get(token, vocab[UNK_TOKEN]) for token in merged_tokens)
    token_ids.append(vocab[EOS_TOKEN])
    return token_ids, len(base_tokens), len(merged_tokens)


def write_split_parquet(
    index_df: pd.DataFrame,
    split: str,
    thresholds: dict[str, list[float]],
    phrase_map: dict[tuple[str, str], str],
    vocab: dict[str, int],
    output_dir: Path,
    burst_gap_ms: float,
    max_seq_len: int,
) -> None:
    rows = []
    split_df = index_df[index_df["split"] == split].copy()
    for _, row in split_df.iterrows():
        file_path = Path(row["file_path"])
        if not file_path.exists():
            print(f"Warning: missing file, skip tokenization: {file_path}")
            continue
        token_ids, num_bursts, num_after_merge = tokenize_trace(
            file_path,
            thresholds,
            phrase_map,
            vocab,
            burst_gap_ms,
            max_seq_len,
        )
        rows.append(
            {
                "trace_id": row["trace_id"],
                "label": int(row["label"]),
                "split": split,
                "tokens": token_ids,
                "num_tokens": len(token_ids),
                "num_bursts_raw": int(num_bursts),
                "num_phrase_tokens": int(num_after_merge),
            }
        )
        if len(rows) % 200 == 0:
            print(f"{split}: tokenized {len(rows)} traces...")

    df = pd.DataFrame(rows)
    output_file = output_dir / f"{split}.parquet"
    try:
        df.to_parquet(output_file, index=False)
        print(f"Saved {split}: {output_file} | rows={len(df)}")
    except ImportError:
        fallback_file = output_dir / f"{split}.jsonl"
        df.to_json(fallback_file, orient="records", lines=True)
        print(
            f"Parquet engine unavailable; saved {split} JSONL fallback: "
            f"{fallback_file} | rows={len(df)}"
        )


def save_artifacts(
    output_dir: Path,
    thresholds: dict[str, list[float]],
    phrase_map: dict[tuple[str, str], str],
    vocab: dict[str, int],
    args,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    phrase_pairs = [
        {"phrase": phrase, "left": left, "right": right}
        for (left, right), phrase in phrase_map.items()
    ]
    config = {
        "version": "packet_tokenization_vABP_ack_burst_phrase",
        "description": (
            "ACK-aware Burst Phrase tokenization. Packets are grouped into "
            "directional bursts; burst volume/timing/TCP progress features are "
            "quantile-binned; frequent adjacent burst pairs from the training "
            "split are merged into phrase tokens."
        ),
        "trace_duration_ns": TRACE_DURATION_NS,
        "burst_gap_ms": args.burst_gap_ms,
        "max_seq_len": args.max_seq_len,
        "max_phrases": args.max_phrases,
        "min_phrase_count": args.min_phrase_count,
        "quantile_levels": QUANTILE_LEVELS,
        "thresholds": thresholds,
        "num_phrases": len(phrase_map),
        "phrase_pairs": phrase_pairs,
        "feature_transform": {
            "total_bytes": "log10(x + 1)",
            "packet_count": "log10(x + 1)",
            "duration_ms": "log10(x + 1)",
            "gap_ms": "log10(x + 1)",
            "avg_len": "raw",
            "seq_progress": "log10(x + 1), with SAME for zero",
            "ack_progress": "log10(x + 1), with SAME for zero",
            "epoch_delta_sum_ms": "log10(x + 1), with SAME for zero",
        },
        "token_format": (
            "B_{S/R/U}_VOL_Q*_PKT_Q*_DUR_Q*_GAP_{START/Q*}_AVG_Q*"
            "_SEQ_{SAME/Q*}_ACK_{SAME/Q*}_EPOCH_{SAME/Q*}_POS_Q*"
        ),
        "special_tokens": {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
            BOS_TOKEN: 2,
            EOS_TOKEN: 3,
        },
        "vocab_size": len(vocab),
    }
    with (output_dir / "bin_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved ABP vocab/config to: {output_dir}")


def main(args) -> None:
    index_file = Path(args.index_file)
    output_dir = Path(args.output_dir)
    if not index_file.exists():
        raise FileNotFoundError(f"Index file does not exist: {index_file}")

    index_df = pd.read_csv(index_file)
    required = {"trace_id", "file_path", "label", "split"}
    missing = required - set(index_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if args.limit_traces is not None:
        index_df = index_df.groupby("split", group_keys=False).head(args.limit_traces)
        print(f"Smoke mode: using up to {args.limit_traces} traces per split")

    print("Collecting ABP training thresholds...")
    values = collect_training_values(index_df, args.burst_gap_ms)
    thresholds = compute_thresholds(values)

    print("Collecting training base-token sequences...")
    train_sequences = collect_training_base_sequences(index_df, thresholds, args.burst_gap_ms)
    print("Learning frequent burst phrases...")
    phrase_map = learn_phrase_map(train_sequences, args.max_phrases, args.min_phrase_count)
    vocab = build_vocab(train_sequences, phrase_map)
    save_artifacts(output_dir, thresholds, phrase_map, vocab, args)

    print(f"Vocab size: {len(vocab)}")
    print(f"Phrase tokens: {len(phrase_map)}")
    print("Tokenizing splits...")
    for split in ["train", "val", "test"]:
        write_split_parquet(
            index_df,
            split,
            thresholds,
            phrase_map,
            vocab,
            output_dir,
            args.burst_gap_ms,
            args.max_seq_len,
        )
    print("ABP tokenization completed.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-file", default=str(INDEX_FILE))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--burst-gap-ms", type=float, default=BURST_GAP_MS)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--max-phrases", type=int, default=MAX_PHRASES)
    parser.add_argument("--min-phrase-count", type=int, default=MIN_PHRASE_COUNT)
    parser.add_argument("--limit-traces", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
