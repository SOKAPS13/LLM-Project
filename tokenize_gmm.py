"""
Tokenize traffic traces with fixed windows + GMM.

This tokenizer keeps the same 600 x 100ms windows, split logic, and
train-only codebook fitting as the fixed-window baseline, but replaces the
K-means codebook with a Gaussian Mixture Model. A 500ms macro-context can be
enabled as an optional augmentation without changing the output sequence shape.
"""

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

SIXTY_S_NS = 60_000_000_000
WINDOW_NS = 100_000_000
NUM_WINDOWS = 600
MACRO_WINDOW = 5
N_LOCAL_FEATURES = 6
N_MACRO_FEATURES = 4

LOCAL_FEATURE_NAMES = [
    "bytes_recv",
    "bytes_sent",
    "pkts_recv",
    "pkts_sent",
    "avg_pkt_size",
    "gap_before_ms",
]

MACRO_FEATURE_NAMES = [
    "macro_bytes_recv",
    "macro_bytes_sent",
    "macro_pkts_recv",
    "macro_pkts_sent",
]

RICH_VOCAB = {
    (0, 0): "silence",
    (1, 0): "whisper", (1, 1): "murmur", (1, 2): "rustle",
    (2, 0): "trickle", (2, 1): "drip", (2, 2): "drops",
    (3, 0): "stream", (3, 1): "flow", (3, 2): "current",
    (4, 0): "flood", (4, 1): "surge", (4, 2): "wave",
    (5, 0): "burst", (5, 1): "peak", (5, 2): "spike",
}

# 256 short 3-letter English words — each is a single BPE token that GPT-2 knows.
# Same strategy as the DT tokenizer: one unique word per GMM component, no BPE expansion.
_WORD_LIST = [
    "ace","act","add","age","ago","aid","aim","air","ale","all",
    "and","ant","any","ape","arc","ark","arm","art","ask","axe",
    "bad","bag","ban","bar","bat","bay","bed","bet","bid","big",
    "bit","bow","box","boy","bug","bus","buy","can","cap","car",
    "cat","cod","cop","cow","cry","cup","cut","dam","day","den",
    "did","dig","dim","dip","dog","dot","dry","due","ear","eat",
    "egg","elk","end","era","eve","eye","fan","far","fat","fee",
    "few","fig","fit","fly","fog","fox","fun","fur","gap","gas",
    "get","god","got","gum","gun","gut","had","ham","has","hat",
    "hay","hen","her","hew","him","hog","hop","hot","hub","hue",
    "hug","hum","hut","ice","ink","inn","ion","ivy","jam","jar",
    "jaw","jet","joy","jug","key","kid","kin","kit","lab","lag",
    "lap","law","lay","led","leg","let","lid","lip","log","lot",
    "low","mad","man","map","mat","men","mob","mop","mud","mug",
    "net","new","nod","nor","not","nun","nut","oak","oar","oat",
    "odd","off","oil","old","one","orb","ore","our","out","owl",
    "pad","pan","par","pat","paw","pea","peg","pen","pet","pin",
    "pit","pod","pop","pot","pro","pub","pup","rag","ram","rat",
    "raw","ray","red","rib","rid","rig","rim","rip","rod","rot",
    "row","rub","rug","rum","rut","rye","sac","sap","saw","say",
    "sea","set","she","shy","sin","sip","sir","sit","ski","sky",
    "sly","sob","sod","son","sow","soy","spa","sub","sum","sun",
    "tab","tag","tan","tar","tax","ten","tip","toe","ton","top",
    "tow","toy","tub","tug","two","urn","van","vat","via","vow",
    "war","wax","web","wed","wee","wet","who","why","wig","win",
    "wit","woe","won","yes","yet","yew",
]
assert len(_WORD_LIST) >= 256, f"Need ≥256 words, have {len(_WORD_LIST)}"


def bytes_inter_to_word(bytes_in: float, inter_ms: float, bytes_bins, timing_bins) -> str:
    b_lvl = 0
    for i, t in enumerate(bytes_bins):
        if bytes_in > t:
            b_lvl = i + 1
    b_lvl = min(b_lvl, 5)
    if b_lvl == 0:
        return "silence"
    if inter_ms <= timing_bins[0]:
        s_lvl = 2
    elif inter_ms <= timing_bins[1]:
        s_lvl = 1
    else:
        s_lvl = 0
    return RICH_VOCAB.get((b_lvl, s_lvl), "stream")


def _weighted_percentile(vals: np.ndarray, weights: np.ndarray, pcts):
    sorter = np.argsort(vals)
    v_s, w_s = vals[sorter], weights[sorter].astype(float)
    cdf = np.cumsum(w_s)
    cdf /= cdf[-1]
    return np.interp(np.array(pcts) / 100.0, cdf, v_s).tolist()


def calibrate_bins_weighted(component_means: np.ndarray, comp_ids: np.ndarray, comp_counts: np.ndarray):
    """
    Compute BYTES/TIMING bin thresholds from the FITTED GMM component means,
    weighted by how often each component is actually used in the training
    data. This matches the true K-means baseline's calibrate_bins() exactly
    (percentiles over cluster centroids weighted by usage frequency) instead
    of taking raw per-window quantiles before clustering.
    """
    means = component_means[comp_ids]
    bytes_vals = np.maximum(0.0, means[:, 0])
    timing_vals = np.maximum(0.0, means[:, 5])

    nz = bytes_vals > 0
    if nz.sum() >= 4:
        bytes_bins = [0.0] + _weighted_percentile(bytes_vals[nz], comp_counts[nz], [20, 40, 60, 80])
    else:
        bytes_bins = [0.0, 100.0, 1_000.0, 10_000.0, 100_000.0]

    if len(timing_vals) >= 2:
        timing_bins = _weighted_percentile(timing_vals, comp_counts, [33, 67])
    else:
        timing_bins = [1.0, 10.0]
    return bytes_bins, timing_bins


def parse_log(filepath: str):
    packets = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            try:
                ts = int(parts[0])
                d = parts[1].strip()
                sz = int(parts[2])
            except ValueError:
                continue
            if ts > SIXTY_S_NS:
                break
            if d in ("s", "r"):
                packets.append((ts, d, sz))
    return packets


def extract_local_features(packets, window_ns: int = WINDOW_NS, num_windows: int = NUM_WINDOWS):
    feat = np.zeros((num_windows, N_LOCAL_FEATURES), dtype=np.float32)
    silent = np.ones(num_windows, dtype=bool)
    # ns timestamp of the last packet seen in ANY prior window — only updated
    # when a window actually has packets. Matches the baseline's
    # `last_pkt_ts`: it must persist across empty/silent windows so the gap
    # feature reflects true elapsed time since the last burst, not just since
    # the previous window. Resetting this on every empty window (the old
    # behaviour here) zeroed out gap_before_ms almost always, since ~97% of
    # windows are silent.
    last_ts = -1

    for w in range(num_windows):
        t0 = w * window_ns
        t1 = t0 + window_ns
        win = [(ts, d, sz) for ts, d, sz in packets if t0 <= ts < t1]

        if not win:
            continue

        silent[w] = False
        bytes_recv = sum(sz for _, d, sz in win if d == "r")
        bytes_sent = sum(sz for _, d, sz in win if d == "s")
        pkts_recv = sum(1 for _, d, _ in win if d == "r")
        pkts_sent = sum(1 for _, d, _ in win if d == "s")
        # avg_size_in in the baseline = mean of RECEIVED packet sizes only,
        # not a combined sent+recv average.
        avg_pkt_size = bytes_recv / pkts_recv if pkts_recv > 0 else 0.0
        gap_ms = (win[0][0] - last_ts) / 1e6 if last_ts >= 0 else 0.0
        last_ts = win[-1][0]
        feat[w] = [bytes_recv, bytes_sent, pkts_recv, pkts_sent, avg_pkt_size, gap_ms]

    return feat, silent


def add_macro_context(local_feat: np.ndarray, num_windows: int = NUM_WINDOWS, macro_window: int = MACRO_WINDOW):
    feat = np.zeros((num_windows, N_LOCAL_FEATURES + N_MACRO_FEATURES), dtype=np.float32)
    feat[:, :N_LOCAL_FEATURES] = local_feat

    for start in range(0, num_windows, macro_window):
        end = min(num_windows, start + macro_window)
        block = local_feat[start:end]
        macro = np.array([
            block[:, 0].sum(),
            block[:, 1].sum(),
            block[:, 2].sum(),
            block[:, 3].sum(),
        ], dtype=np.float32)
        feat[start:end, 6:] = macro
    return feat


def extract_features(packets, use_macro_context: bool = False, window_ns: int = WINDOW_NS,
                      num_windows: int = NUM_WINDOWS, macro_window: int = MACRO_WINDOW):
    local_feat, silent = extract_local_features(packets, window_ns, num_windows)
    if use_macro_context:
        return add_macro_context(local_feat, num_windows, macro_window), silent
    return local_feat, silent


def get_label_and_offset(filepath: str):
    name = Path(filepath).name
    parts = name.split("-")
    label = int(parts[0]) if len(parts) > 0 else -1
    offset = int(parts[1]) if len(parts) > 1 else -1
    return label, offset


def collect_files(base_path: str, files_per_class: int, fixed_offset: int = None):
    all_files = []
    for folder in sorted(
        Path(base_path).iterdir(),
        key=lambda p: int(p.name) if p.name.isdigit() else -1,
    ):
        if not folder.is_dir():
            continue

        # Plain sorted order (label-offset-rep filename), first N files —
        # matches the baseline: don't round-robin/spread across every offset
        # value, just take the first files_per_class files in natural sorted
        # order (offset stays low/consistent instead of being deliberately
        # diversified).
        files = sorted(f for f in folder.glob("*.log") if ".qoe.log" not in f.name)
        if fixed_offset is not None:
            # Restrict to a single offset value per class — every sample of
            # a given movie is then the SAME 60s segment of that video
            # (only network-timing repeats vary), matching how the pasted
            # baseline's tiny 10-file/class dataset is structured. Reps
            # (third filename field) become the only sampling axis.
            files = [f for f in files if get_label_and_offset(str(f))[1] == fixed_offset]
        for f in files[:files_per_class]:
            label, offset = get_label_and_offset(str(f))
            all_files.append((str(f), label, offset))

    return all_files


def _process_file(fp: str, label: int, offset: int, use_macro_context: bool,
                   window_ns: int = WINDOW_NS, num_windows: int = NUM_WINDOWS,
                   macro_window: int = MACRO_WINDOW):
    try:
        pkts = parse_log(fp)
        feat, silent = extract_features(pkts, use_macro_context=use_macro_context,
                                         window_ns=window_ns, num_windows=num_windows,
                                         macro_window=macro_window)
        return {
            "feat": feat,
            "silent": silent,
            "label": label,
            "offset": offset,
            "filepath": fp,
            "n_pkts": len(pkts),
        }
    except Exception as exc:
        print(f"  ERROR {fp}: {exc}")
        return None


def stratified_split(labels, train_k, val_k, test_k, seed=42):
    rng = np.random.default_rng(seed)
    cls_idx = defaultdict(list)
    for i, lbl in enumerate(labels):
        cls_idx[int(lbl)].append(i)

    train, val, test = [], [], []
    for indices in sorted(cls_idx.values()):
        perm = rng.permutation(len(indices))
        indices = [indices[i] for i in perm]
        t_k = min(train_k, len(indices))
        v_k = min(val_k, len(indices) - t_k)
        e_k = min(test_k, len(indices) - t_k - v_k)
        train.extend(indices[:t_k])
        val.extend(indices[t_k:t_k + v_k])
        test.extend(indices[t_k + v_k:t_k + v_k + e_k])

    return np.array(train), np.array(val), np.array(test)


def _valid_saved_split(train_idx, val_idx, test_idx, n_total):
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    if len(all_idx) == 0:
        return False
    if np.any(all_idx < 0) or np.any(all_idx >= n_total):
        return False
    if len(np.unique(all_idx)) != len(all_idx):
        return False
    return True


def main(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    window_ns = args.window_ms * 1_000_000
    num_windows = SIXTY_S_NS // window_ns
    macro_window = max(1, round(args.macro_ms / args.window_ms))
    if num_windows + 2 > args.max_seq_len:
        raise ValueError(
            f"--window_ms {args.window_ms} gives {num_windows} windows, which needs "
            f"max_seq_len >= {num_windows + 2}; increase --max_seq_len or --window_ms."
        )

    print("=" * 60)
    print("Tokenize GMM — fixed-window tokenizer")
    print("=" * 60, flush=True)
    print(f"  Base path         : {args.base}", flush=True)
    print(f"  Files per class   : {args.files_per_class}", flush=True)
    print(f"  Train files/class : {args.train_files}", flush=True)
    print(f"  Val files/class   : {args.val_files}", flush=True)
    print(f"  GMM codebook size : {args.k}", flush=True)
    print(f"  Window size       : {args.window_ms}ms  ({num_windows} windows over 60s)", flush=True)
    print(f"  Macro context     : {'on, ' + str(macro_window * args.window_ms) + 'ms blocks' if args.use_macro_context else 'off'}", flush=True)
    print(f"  Output dir        : {out_dir}/", flush=True)

    all_files = collect_files(args.base, args.files_per_class, fixed_offset=args.fixed_offset)
    n_classes = len({lbl for _, lbl, _ in all_files})
    print(f"\n[1] Collected {len(all_files)} files from {n_classes} classes", flush=True)

    print("\n[2] Extracting fixed-window features ...", flush=True)
    t0 = time.time()
    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(_process_file)(fp, lbl, offset, args.use_macro_context, window_ns, num_windows, macro_window)
        for fp, lbl, offset in all_files
    )
    results = [r for r in results if r is not None]
    print(f"    Done in {time.time() - t0:.1f}s — {len(results)}/{len(all_files)} files OK", flush=True)

    print("\n[3] Creating/reusing stratified train / val / test split ...", flush=True)
    labels_arr = np.array([r["label"] for r in results], dtype=np.int64)

    split_train_p = out_dir / "split_train_idx.npy"
    split_val_p = out_dir / "split_val_idx.npy"
    split_test_p = out_dir / "split_test_idx.npy"

    use_saved = (
        (not args.force_resplit)
        and split_train_p.exists()
        and split_val_p.exists()
        and split_test_p.exists()
    )

    if use_saved:
        train_idx = np.load(split_train_p)
        val_idx = np.load(split_val_p)
        test_idx = np.load(split_test_p)
        if not _valid_saved_split(train_idx, val_idx, test_idx, len(results)):
            raise RuntimeError("Saved split files are invalid. Re-run with --force_resplit.")
        print("    Reusing existing split indices from data/")
    else:
        train_k = min(args.train_files, args.files_per_class - args.val_files - 1)
        val_k = min(args.val_files, args.files_per_class - train_k - 1)
        test_k = args.files_per_class - train_k - val_k
        if train_k <= 0 or val_k <= 0 or test_k <= 0:
            raise ValueError(
                f"Invalid split sizes per class: train={train_k}, val={val_k}, test={test_k}."
            )
        train_idx, val_idx, test_idx = stratified_split(
            labels_arr, train_k=train_k, val_k=val_k, test_k=test_k
        )
        np.save(split_train_p, train_idx)
        np.save(split_val_p, val_idx)
        np.save(split_test_p, test_idx)
        print("    Created and saved new split indices")

    print(f"    Train : {len(train_idx)}   Val : {len(val_idx)}   Test : {len(test_idx)}")

    print("\n[4] Building GMM training set from TRAIN split only ...", flush=True)
    X_train = []
    for i in train_idx:
        r = results[int(i)]
        mask = ~r["silent"]
        if mask.any():
            X_train.append(r["feat"][mask])
    # float64: GMM covariance estimation is numerically fragile in float32,
    # especially with many components relative to distinct feature values
    # (e.g. clusters of near-duplicate windows can collapse to ~0 variance).
    X_train = np.vstack(X_train).astype(np.float64)
    print(f"    GMM training windows : {len(X_train):,}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    n_active_components = min(args.k - 1, len(X_scaled))
    if n_active_components < 2:
        raise RuntimeError("Not enough active windows to fit a useful GMM codebook.")

    # GMM (not K-means): deliberately kept different from the baseline's
    # MiniBatchKMeans, since the goal here is a GMM + multiresolution
    # pipeline distinct from the plain fixed-window K-means baseline.
    print(f"\n[5] Fitting GaussianMixture(n_components={n_active_components}) ...", flush=True)
    reg_covar = args.reg_covar
    for attempt in range(5):
        gmm = GaussianMixture(
            n_components=n_active_components,
            covariance_type=args.covariance_type,
            reg_covar=reg_covar,
            max_iter=args.max_iter,
            n_init=args.n_init,
            random_state=42,
        )
        try:
            gmm.fit(X_scaled)
            break
        except ValueError as e:
            if "ill-defined empirical covariance" not in str(e) or attempt == 4:
                raise
            reg_covar *= 10
            print(f"    GMM fit failed (collapsed component); retrying with "
                  f"reg_covar={reg_covar:.1e} ...", flush=True)

    SILENT_ID = n_active_components
    PAD_ID = n_active_components + 1
    BOS_ID = n_active_components + 2
    EOS_ID = n_active_components + 3
    CLS_ID = n_active_components + 4
    VOCAB_SIZE = n_active_components + 5
    SEQ_LEN = args.max_seq_len

    # One unique word per GMM component — same strategy as the DT tokenizer
    # (each cluster/leaf gets its own single-BPE-token English word, no
    # collapsing to a shared 15-word vocabulary). Reverted back to this after
    # the 15-word semantic-collapse variant scored clearly worse (best val
    # acc 4.4% vs 15.6% Top-1 previously).
    id2word = {comp_id: _WORD_LIST[comp_id] for comp_id in range(n_active_components)}
    id2word[SILENT_ID] = "silence"
    id2word[PAD_ID] = "pad"
    id2word[BOS_ID] = "bos"
    id2word[EOS_ID] = "eos"
    id2word[CLS_ID] = "cls"

    token_names = [f"GMM{i:03d}" for i in range(n_active_components)] + ["SILENT"]
    token2id = {name: i for i, name in enumerate(token_names)}
    token2id.update({"PAD": PAD_ID, "BOS": BOS_ID, "EOS": EOS_ID, "CLS": CLS_ID})

    print(f"    Active components  : {n_active_components}")
    print(f"    SILENT token ID    : {SILENT_ID}")
    print(f"    Vocab size         : {VOCAB_SIZE}")

    gmm_pkg = {
        "gmm": gmm,
        "scaler": scaler,
        "silent": SILENT_ID,
        "feature_names": LOCAL_FEATURE_NAMES + (MACRO_FEATURE_NAMES if args.use_macro_context else []),
        "use_macro_context": bool(args.use_macro_context),
    }
    with open(out_dir / "gmm_tokenizer.pkl", "wb") as f:
        pickle.dump(gmm_pkg, f)

    print(f"\n[6] Tokenizing {len(results)} files ...")
    dataset, all_seqs, all_labels = [], [], []
    n_total = len(results)

    for i, r in enumerate(results):
        feat, silent = r["feat"], r["silent"]
        lbl = r["label"]
        non_silent = ~silent

        tok_ids = np.full(len(silent), SILENT_ID, dtype=np.int32)
        if non_silent.any():
            X_file = scaler.transform(feat[non_silent])
            tok_ids[non_silent] = gmm.predict(X_file).astype(np.int32)

        seq = np.full(SEQ_LEN, PAD_ID, dtype=np.int32)
        seq[0] = BOS_ID
        n_take = min(len(tok_ids), SEQ_LEN - 2)
        seq[1:1 + n_take] = tok_ids[:n_take]
        seq[1 + n_take] = EOS_ID

        all_seqs.append(seq)
        all_labels.append(lbl)
        dataset.append({
            "tokens": [id2word.get(int(t), str(int(t))) for t in tok_ids[:n_take]],
            "label": lbl,
            "filename": Path(r["filepath"]).name,
            "packets": r["n_pkts"],
            "windows": int(non_silent.sum()),
            "offset": r["offset"],
        })

        if (i + 1) % 100 == 0 or (i + 1) == n_total:
            print(f"  [6] Tokenized {i+1}/{n_total} files ...", flush=True)

    print(f"\n[7] Saving arrays and vocab to {out_dir}/ ...", flush=True)
    X = np.array(all_seqs, dtype=np.int32)
    y = np.array(all_labels, dtype=np.int64)
    np.save(out_dir / "X_tokens.npy", X)
    np.save(out_dir / "y_labels.npy", y)

    vocab_cfg = {
        "gmm": gmm,
        "scaler": scaler,
        "k": n_active_components + 1,
        "silent": SILENT_ID,
        "special": {"PAD": PAD_ID, "BOS": BOS_ID, "EOS": EOS_ID, "CLS": CLS_ID},
        "vocab_size": VOCAB_SIZE,
        "token2id": token2id,
        "id2token": {i: n for n, i in token2id.items()},
        "id2word": id2word,
        "feature_names": LOCAL_FEATURE_NAMES + (MACRO_FEATURE_NAMES if args.use_macro_context else []),
        "use_macro_context": bool(args.use_macro_context),
    }
    with open(out_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab_cfg, f)

    print(f"\n[8] Saving {args.json_out} ...")
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"  Files processed     : {len(dataset)}")
    print(f"  Active components   : {n_active_components}")
    print(f"  Vocab size          : {VOCAB_SIZE}")
    print(f"  Sequence length     : {SEQ_LEN}")
    print(f"  Window count        : {num_windows}  ({args.window_ms}ms each)")
    print(f"  Train / Val / Test  : {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")
    print(f"\nNext step:  python GPT-2.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed-window GMM tokenizer")
    parser.add_argument("--base", default="LongEnough/none-none", help="Root path containing per-video sub-folders 0/ … 99/")
    parser.add_argument("--files_per_class", type=int, default=50, help="Number of .log files to use per class")
    parser.add_argument("--train_files", type=int, default=40, help="Training files per class")
    parser.add_argument("--val_files", type=int, default=5, help="Validation files per class")
    parser.add_argument("--k", type=int, default=256, help="Total codebook size including the dedicated silent token")
    parser.add_argument("--fixed_offset", type=int, default=None, help="If set, only use files with this offset (2nd filename field) per class, isolating offset diversity as a variable")
    parser.add_argument("--window_ms", type=int, default=100, help="Local window duration in milliseconds (600 windows @ 100ms = 60s trace)")
    parser.add_argument("--macro_ms", type=int, default=500, help="Macro-context block duration in milliseconds (rounded to nearest multiple of --window_ms)")
    parser.add_argument("--covariance_type", default="diag", choices=["full", "tied", "diag", "spherical"], help="GaussianMixture covariance type")
    parser.add_argument("--reg_covar", type=float, default=1e-4, help="Regularization added to covariance diagonals")
    parser.add_argument("--max_iter", type=int, default=200, help="Maximum EM iterations")
    parser.add_argument("--n_init", type=int, default=2, help="Number of GMM initializations")
    parser.add_argument("--max_seq_len", type=int, default=800, help="Padded sequence length for GPT-2")
    parser.add_argument("--out", default="data", help="Output directory for numpy arrays and vocab")
    parser.add_argument("--json_out", default="tokenize_gmm.json", help="Output JSON file path")
    parser.add_argument("--force_resplit", action="store_true", help="Ignore saved split indices and create a new split")
    parser.add_argument("--use_macro_context", action="store_true", help="Append 500ms macro-context features before fitting GMM")
    main(parser.parse_args())