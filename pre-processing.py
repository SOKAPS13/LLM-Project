"""
Part 1 — Data Preprocessing Pipeline
Video Fingerprinting via Encrypted Network Traffic

Tokenization design (VQ / K-means codebook):
  - Each 100ms window has 6 raw features → normalise → K-means cluster
  - Each window maps to its nearest centroid ID  →  ONE token per window
  - This preserves the temporal structure of the time-series:
      window 0  → token t0
      window 1  → token t1
      ...
      window 599→ token t599
  - Sequence per trace: [BOS] t0 t1 ... t599 [EOS]  (length 602)
  - Special tokens: PAD=K, BOS=K+1, EOS=K+2, CLS=K+3
  - Default K=256  → vocab_size = 260

Why K-means (VQ codebook)?
  - Analogous to BPE in NLP: learns the most frequent joint patterns
    across all 6 features simultaneously
  - Single token per window keeps adjacent windows at distance 1 in
    the sequence, so the GPT can directly learn burst/silence rhythms,
    segment-download cycles, and bitrate adaptation curves

No leakage: the scaler + K-means codebook are fit on the TRAIN split only
(same stratified split later used by pre-train.py, computed here first so
both stages agree). Val/test traces are only ever *transformed* with the
already-fitted scaler/kmeans, never seen during fitting.
"""

import os, json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.cluster    import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from model import Config

# ── Constants ─────────────────────────────────────────────────────────────────
SIXTY_S_NS    = 60_000_000_000
WINDOW_NS     = 100_000_000        # 100 ms
NUM_WINDOWS   = 600
NUM_FEATURES  = 6
K             = 256                # codebook size (tune-able)

FEATURE_NAMES = ["bytes_in", "bytes_out", "pkts_in",
                 "pkts_out", "avg_size_in", "gap_before_burst_ms"]

SEED = 42
_SPLIT_CFG = Config()
TRAIN_K, VAL_K, TEST_K = _SPLIT_CFG.train_k, _SPLIT_CFG.val_k, _SPLIT_CFG.test_k


# ── Stratified split (must match pre-train.py exactly — same seed/logic) ─────
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


# ── Step 1: Parse log ─────────────────────────────────────────────────────────
def parse_log(filepath: str) -> pd.DataFrame:
    rows = []
    with open(filepath) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 3:
                continue
            try:
                ts = int(p[0]); d = p[1].strip(); sz = int(p[2])
            except ValueError:
                continue
            if ts > SIXTY_S_NS:
                break
            if d in ("s", "r"):
                rows.append((ts, d, sz))
    return pd.DataFrame(rows, columns=["ts", "dir", "size"])


# ── Step 2: Extract raw feature matrix (NUM_WINDOWS, NUM_FEATURES) ────────────
def extract_features(df: pd.DataFrame) -> np.ndarray:
    feat = np.zeros((NUM_WINDOWS, NUM_FEATURES), dtype=np.float32)
    last_pkt_ts = -1  # ns timestamp of last packet seen in any prior window
    for w in range(NUM_WINDOWS):
        t0, t1 = w * WINDOW_NS, (w + 1) * WINDOW_NS
        win  = df[(df.ts >= t0) & (df.ts < t1)]
        recv = win[win.dir == "r"]
        sent = win[win.dir == "s"]
        if len(win) > 0:
            # gap_before_burst: time (ms) between first packet here and last packet before
            gap = (win.ts.min() - last_pkt_ts) / 1e6 if last_pkt_ts >= 0 else 0.0
            last_pkt_ts = win.ts.max()
        else:
            gap = 0.0
        feat[w] = [
            recv["size"].sum(),
            sent["size"].sum(),
            len(recv),
            len(sent),
            recv["size"].mean() if len(recv) else 0.0,
            gap,
        ]
    return feat


# ── Step 3: Build codebook (VQ vocabulary) ───────────────────────────────────
def build_codebook(all_features: np.ndarray, k: int = K) -> dict:
    """
    Fit StandardScaler + MiniBatchKMeans on non-empty windows only.
    `all_features` must come from TRAIN traces only (see build_dataset) —
    fitting on val/test traces would leak their distribution into the codebook.
    Empty windows (all features == 0) are assigned a dedicated SILENT token (K-1)
    so K-means centroids are not wasted on near-zero noise.
    Special tokens: PAD=K, BOS=K+1, EOS=K+2, CLS=K+3  → vocab_size = K+4
    """
    non_empty_mask = all_features.any(axis=1)
    X_active = all_features[non_empty_mask]

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_active)

    # Reserve centroid K-1 for SILENT windows; fit K-1 centroids for active windows
    n_active_clusters = k - 1
    print(f"  Fitting K-means (K={n_active_clusters} active + 1 silent) "
          f"on {len(X_norm):,} non-empty windows …")
    kmeans = MiniBatchKMeans(n_clusters=n_active_clusters, random_state=42,
                             batch_size=4096, n_init=5, max_iter=300)
    kmeans.fit(X_norm)
    print(f"  K-means inertia : {kmeans.inertia_:.2f}")

    SILENT = k - 1
    special = {"PAD": k, "BOS": k + 1, "EOS": k + 2, "CLS": k + 3}
    return {
        "scaler":     scaler,
        "kmeans":     kmeans,
        "k":          k,
        "silent":     SILENT,
        "special":    special,
        "vocab_size": k + len(special),
    }


# ── Step 4: Tokenize one trace ────────────────────────────────────────────────
def tokenize(features: np.ndarray, vocab: dict) -> np.ndarray:
    """
    Map each of the 600 windows to its nearest centroid ID.
    Empty windows (all features == 0) get the SILENT token.
    Wrap with [BOS] and [EOS].  Returns int array of shape (602,).
    """
    SILENT    = vocab["silent"]
    BOS, EOS  = vocab["special"]["BOS"], vocab["special"]["EOS"]
    token_ids = np.full(NUM_WINDOWS, SILENT, dtype=np.int32)

    non_empty = features.any(axis=1)
    if non_empty.any():
        X_norm = vocab["scaler"].transform(features[non_empty])
        token_ids[non_empty] = vocab["kmeans"].predict(X_norm)

    return np.concatenate([[BOS], token_ids, [EOS]])


# ── Step 5: Label from parent folder name ─────────────────────────────────────
def label_from_filepath(filepath: str) -> str:
    return Path(filepath).parent.name


# ── Step 6: Sanity check — same-video similarity ──────────────────────────────
def sanity_check(X: np.ndarray, y: np.ndarray, label_map: dict):
    """
    For each class, compute the mean pairwise token overlap (Jaccard on
    unigrams) between traces of the same video vs different videos.
    High intra-class similarity and low inter-class similarity = good codebook.
    """
    rev_map = {v: k for k, v in label_map.items()}
    classes = sorted(label_map.values())

    def token_set(seq):
        return set(seq[1:-1].tolist())   # strip BOS/EOS

    print("\nSanity check — token overlap (Jaccard similarity):")
    print(f"  {'':12}  intra-class  inter-class  ratio")
    for c in classes:
        same  = X[y == c]
        other = X[y != c]
        if len(same) < 2 or len(other) == 0:
            continue
        intra = np.mean([
            len(token_set(same[i]) & token_set(same[j])) /
            len(token_set(same[i]) | token_set(same[j]))
            for i in range(len(same)) for j in range(i+1, len(same))
        ])
        inter = np.mean([
            len(token_set(same[i]) & token_set(other[j])) /
            len(token_set(same[i]) | token_set(other[j]))
            for i in range(len(same)) for j in range(len(other))
        ])
        ratio = intra / inter if inter > 0 else float("inf")
        print(f"  video {rev_map[c]:8}  {intra:.3f}        {inter:.3f}        {ratio:.2f}x")


# ── Step 7: Main pipeline ─────────────────────────────────────────────────────
def build_dataset(log_dir: str, output_dir: str = "./data") -> dict:
    log_files = sorted(Path(log_dir).rglob("*.log"),
                       key=lambda p: (int(p.parent.name), p.name))
    if not log_files:
        raise FileNotFoundError(f"No .log files under {log_dir}")

    n_folders = len(set(p.parent for p in log_files))
    print(f"Found {len(log_files)} log files across {n_folders} folders")

    raw_features, video_ids = [], []
    for fp in log_files:
        vid  = label_from_filepath(str(fp))
        df   = parse_log(str(fp))
        feat = extract_features(df)
        raw_features.append(feat)
        video_ids.append(vid)
        print(f"  {fp.parent.name}/{fp.name}  video={vid}  pkts={len(df)}")

    # Labels — sort numerically by folder index
    unique_vids = sorted(set(video_ids), key=int)
    label_map   = {v: i for i, v in enumerate(unique_vids)}
    y = np.array([label_map[v] for v in video_ids], dtype=np.int64)
    print(f"Classes: {len(label_map)}")

    # Stratified split FIRST — computed here so the codebook can be fit on
    # train traces only. pre-train.py loads these same indices (does not
    # recompute them) so training/val/test downstream match exactly.
    train_idx, val_idx, test_idx = stratified_split(y, TRAIN_K, VAL_K, TEST_K)
    print(f"Split       : {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test "
          f"({TRAIN_K}/{VAL_K}/{TEST_K} per class)")

    # Fit codebook on TRAIN traces only — val/test never influence the scaler/kmeans
    train_feat = np.vstack([raw_features[i] for i in train_idx])
    vocab      = build_codebook(train_feat)
    print(f"\nCodebook ready: vocab_size={vocab['vocab_size']}  (fit on {len(train_idx)} train traces)")
    print(f"  Active centroids: 0 – {vocab['k']-2}   Silent: {vocab['silent']}")
    print(f"  Special tokens  : {vocab['special']}")

    # Tokenize every trace (train/val/test) with the train-fitted vocab → shape (N, 602)
    X = np.vstack([tokenize(f, vocab) for f in raw_features])
    print(f"\nToken matrix: {X.shape}   (traces × seq_len)")
    print(f"Token range : [{X.min()}, {X.max()}]")

    # Sanity check (only meaningful with multiple classes)
    if len(label_map) > 1:
        sanity_check(X, y, label_map)

    # Save
    import pickle
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/X_tokens.npy",  X)
    np.save(f"{output_dir}/y_labels.npy",  y)
    np.save(f"{output_dir}/split_train_idx.npy", np.array(train_idx))
    np.save(f"{output_dir}/split_val_idx.npy",   np.array(val_idx))
    np.save(f"{output_dir}/split_test_idx.npy",  np.array(test_idx))
    with open(f"{output_dir}/vocab.pkl",   "wb") as f:
        pickle.dump({"scaler": vocab["scaler"], "kmeans": vocab["kmeans"],
                     "k": vocab["k"], "special": vocab["special"],
                     "vocab_size": vocab["vocab_size"]}, f)
    with open(f"{output_dir}/label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"\nSaved to {output_dir}/")
    return {"X": X, "y": y, "vocab": vocab, "label_map": label_map,
            "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    DATASET_DIR = "/home2/somka34/somiya/LLMproject"
    OUTPUT_DIR  = "/home2/somka34/somiya/LLM/data"

    if len(sys.argv) == 2 and sys.argv[1] == "demo":
        log_path = sorted(Path(DATASET_DIR).rglob("*.log"))[0]
        print("=" * 60)
        print("PART 1 — SINGLE FILE DEMO")
        print("=" * 60)
        df   = parse_log(str(log_path))
        feat = extract_features(df)
        print(f"File                 : {log_path}")
        print(f"Packets in first 60s : {len(df)}")
        print(f"Feature matrix       : {feat.shape}  (windows × features)")
        non_empty = feat.any(axis=1).sum()
        print(f"Non-empty windows    : {non_empty} / {NUM_WINDOWS}  ({non_empty/NUM_WINDOWS*100:.1f}%)\n")

        # Demo uses small K — real K=256 requires all 1000 files (~600k windows)
        vocab  = build_codebook(feat, k=min(K, feat.any(axis=1).sum() // 2))
        tokens = tokenize(feat, vocab)

        print(f"\nSequence length  : {len(tokens)}  (BOS + 600 windows + EOS)")
        print(f"Vocab size       : {vocab['vocab_size']}")
        print(f"Silent token     : {vocab['silent']}")
        silent_count = (tokens[1:-1] == vocab['silent']).sum()
        print(f"Silent windows   : {silent_count} / 600")

        sp_rev = {v: k for k, v in vocab["special"].items()}
        sp_rev[vocab["silent"]] = "[SILENT]"
        print(f"\nFirst 10 tokens:")
        for i, t in enumerate(tokens[:10]):
            label = sp_rev.get(t, f"centroid_{t}")
            print(f"  pos {i:3d}  token {t:3d}  → {label}")

        print(f"\nTop 10 most frequent window tokens:")
        ids, counts = np.unique(tokens[1:-1], return_counts=True)
        top = np.argsort(-counts)[:10]
        for i in top:
            pct = counts[i] / 600 * 100
            if ids[i] == vocab["silent"]:
                print(f"  token {ids[i]:3d}  count={counts[i]:3d} ({pct:4.1f}%)  [SILENT]")
            else:
                c   = vocab["kmeans"].cluster_centers_[ids[i]]
                orig = vocab["scaler"].inverse_transform(c.reshape(1,-1))[0]
                print(f"  token {ids[i]:3d}  count={counts[i]:3d} ({pct:4.1f}%)  "
                      f"bytes_in={orig[0]:8.0f}  pkts_in={orig[2]:5.1f}  "
                      f"gap={orig[5]:.0f}ms")
    else:
        print("=" * 60)
        print("PART 1 — FULL DATASET PREPROCESSING")
        print("=" * 60)
        build_dataset(DATASET_DIR, output_dir=OUTPUT_DIR)
