#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = (
    "bradley-terry",
    "huber-log",
    "elo",
    "pagerank",
    "colley",
    "rank-centrality",
)

REQUIRED_COLUMNS = {"competition_id", "item_a", "item_b", "winner"}

MAD_SCALE = 0.6744897501960817
IRLS_MAX_ITER = 50
IRLS_TOL = 1e-8
EPS = 1e-12

ELO_K = 32.0

PAGERANK_DAMPING = 0.85
PAGERANK_TOL = 1e-10
PAGERANK_MAX_ITER = 200

BT_MAX_ITER = 5000
BT_TOL = 1e-10


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute item weights from pairwise comparisons.")
    p.add_argument("--csv", type=Path, required=True, help="CSV file or directory of CSV files.")
    p.add_argument(
        "--out",
        type=Path,
        help="Directory where per-method CSV files will be written. Defaults to <input>_weights or <dir>/weights.",
    )
    p.add_argument(
        "--bt-temp",
        type=float,
        default=1.0,
        help="Bradley–Terry softmax temperature (>1 flattens, <1 sharpens). Default: 1.0",
    )
    return p.parse_args(argv)


def default_out_dir(src: Path) -> Path:
    if src.is_file():
        return src.parent / f"{src.stem}_weights"
    if src.is_dir():
        return src / "weights"
    return Path.cwd() / "weights"


def load_csvs(path: Path) -> pd.DataFrame:
    if path.is_file():
        frames = [pd.read_csv(path)]
    elif path.is_dir():
        csvs = sorted(p for p in path.glob("*.csv") if p.is_file())
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in directory {path}.")
        frames = [pd.read_csv(p) for p in csvs]
    else:
        raise FileNotFoundError(f"Input path {path} does not exist.")

    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}.")
    return df


def prepare_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    pairs = frame.dropna(subset=["item_a", "item_b", "winner"]).copy()
    m = (pairs["winner"] == pairs["item_a"]) | (pairs["winner"] == pairs["item_b"])
    if not m.any():
        return pd.DataFrame(columns=["item_a", "item_b", "choice", "timestamp"])

    pairs = pairs[m]
    choice = np.where(pairs["winner"] == pairs["item_a"], 1, 2)

    out = pd.DataFrame(
        {"item_a": pairs["item_a"].astype(str), "item_b": pairs["item_b"].astype(str), "choice": choice.astype(np.int64)}
    )

    if "comparison_created_at" in pairs.columns:
        out["timestamp"] = pd.to_datetime(pairs["comparison_created_at"], errors="coerce", utc=True)
    return out


def clean_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"item_a", "item_b", "choice"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing prepared columns: {', '.join(sorted(missing))}.")
    clean = frame.dropna(subset=list(required)).copy()
    clean["choice"] = pd.to_numeric(clean["choice"], errors="coerce").astype("Int64")
    clean = clean[clean["choice"].isin([1, 2])]
    if clean.empty:
        raise ValueError("No valid comparisons available after cleaning.")
    return clean.reset_index(drop=True)


def unique_items(clean: pd.DataFrame) -> pd.Index:
    items = pd.Index(pd.unique(pd.concat([clean["item_a"], clean["item_b"]], ignore_index=True)))
    if items.empty:
        raise ValueError("No items present in the input data.")
    return items


def design_matrix(clean: pd.DataFrame, items: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    index = {item: i for i, item in enumerate(items)}
    ia = clean["item_a"].map(index).to_numpy(dtype=np.int64)
    ib = clean["item_b"].map(index).to_numpy(dtype=np.int64)

    n, m = len(clean), len(items)
    X = np.zeros((n, m), dtype=float)
    rows = np.arange(n, dtype=np.int64)
    X[rows, ia] = -1.0
    X[rows, ib] = 1.0

    y = np.where(clean["choice"].to_numpy(dtype=int) == 2, 1.0, -1.0)
    return X, y


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = float(temperature) if np.isfinite(temperature) and temperature > 0 else 1.0
    z = logits / t
    z -= np.max(z)
    p = np.exp(z)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Failed to compute finite weights.")
    return p / s


def finalize(items: pd.Index, logits: np.ndarray, *, temperature: float = 1.0) -> pd.DataFrame:
    w = softmax(logits, temperature=temperature)
    return pd.DataFrame({"item": items, "weight": w}).sort_values("weight", ascending=False, ignore_index=True)


def mad(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def huber_logits(clean: pd.DataFrame, items: pd.Index) -> np.ndarray:
    X, y = design_matrix(clean, items)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    w -= w.mean()

    for _ in range(IRLS_MAX_ITER):
        r = y - X @ w
        m = mad(r)
        sigma = m / MAD_SCALE if (np.isfinite(m) and m > 0) else np.mean(np.abs(r - np.mean(r)))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0
        delta = 1.345 * float(sigma)

        a = np.abs(r)
        wts = np.ones_like(r)
        wts[a > delta] = delta / np.maximum(a[a > delta], EPS)

        s = np.sqrt(wts)
        Xw = X * s[:, None]
        yw = y * s
        w_new, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        w_new -= w_new.mean()

        if np.linalg.norm(w_new - w) <= IRLS_TOL * (1.0 + np.linalg.norm(w)):
            w = w_new
            break
        w = w_new

    return w


def bradley_terry_logits(clean: pd.DataFrame, items: pd.Index) -> np.ndarray:
    n = len(items)
    idx = {item: i for i, item in enumerate(items)}

    wins = np.zeros(n, dtype=float)
    pair_counts: dict[tuple[int, int], int] = {}
    for choice, a, b in clean[["choice", "item_a", "item_b"]].itertuples(index=False, name=None):
        i, j = idx[a], idx[b]
        if int(choice) == 1:
            wins[i] += 1.0
        else:
            wins[j] += 1.0
        key = (i, j) if i < j else (j, i)
        pair_counts[key] = pair_counts.get(key, 0) + 1

    neighbors: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for (i, j), cnt in pair_counts.items():
        neighbors[i].append((j, cnt))
        neighbors[j].append((i, cnt))

    s = np.full(n, 1.0 / n, dtype=float)
    for _ in range(BT_MAX_ITER):
        s_new = np.zeros_like(s)
        for i in range(n):
            denom = 0.0
            si = s[i]
            for j, cnt in neighbors[i]:
                denom += cnt / max(si + s[j], EPS)
            s_new[i] = (wins[i] / denom) if denom > 0 else 0.0

        total = s_new.sum()
        s_new[:] = 1.0 / n if (total <= 0 or not np.isfinite(total)) else (s_new / total)

        if np.linalg.norm(s_new - s, ord=1) <= BT_TOL:
            s = s_new
            break
        s = s_new

    logits = np.log(np.clip(s, EPS, None))
    logits -= logits.mean()
    return logits


def elo_logits(clean: pd.DataFrame, items: pd.Index) -> np.ndarray:
    idx = {item: i for i, item in enumerate(items)}
    ratings = np.zeros(len(items), dtype=float)

    ordered = clean
    if "timestamp" in clean.columns:
        ordered = (
            clean.assign(_ts=pd.to_datetime(clean["timestamp"], errors="coerce"))
            .sort_values("_ts", kind="stable")
            .drop(columns=["_ts"])
        )

    for row in ordered.itertuples(index=False):
        a, b, choice = getattr(row, "item_a"), getattr(row, "item_b"), int(getattr(row, "choice"))
        if a not in idx or b not in idx:
            continue
        win, lose = (idx[a], idx[b]) if choice == 1 else (idx[b], idx[a])
        diff = ratings[win] - ratings[lose]
        expected = 1.0 / (1.0 + np.exp(-diff))
        delta = ELO_K * (1.0 - expected)
        ratings[win] += delta
        ratings[lose] -= delta

    ratings -= ratings.mean()
    sd = np.std(ratings)
    if np.isfinite(sd) and sd > 1.0:
        ratings /= sd
    return ratings


def pagerank_logits(clean: pd.DataFrame, items: pd.Index) -> np.ndarray:
    idx = {item: i for i, item in enumerate(items)}
    n = len(items)
    A = np.zeros((n, n), dtype=float)

    for choice, a, b in clean[["choice", "item_a", "item_b"]].itertuples(index=False, name=None):
        if a not in idx or b not in idx:
            continue
        win = idx[a] if int(choice) == 1 else idx[b]
        lose = idx[b] if int(choice) == 1 else idx[a]
        A[lose, win] += 1.0

    row_sums = A.sum(axis=1, keepdims=True)
    T = np.divide(A, np.where(row_sums <= 0, 1.0, row_sums), out=np.full_like(A, 1.0 / n), where=row_sums > 0)

    rank = np.full(n, 1.0 / n, dtype=float)
    teleport = (1.0 - PAGERANK_DAMPING) / n

    for _ in range(PAGERANK_MAX_ITER):
        new = teleport + PAGERANK_DAMPING * (T.T @ rank)
        if np.linalg.norm(new - rank, ord=1) <= PAGERANK_TOL:
            rank = new
            break
        rank = new

    rank = np.clip(rank, EPS, None)
    logits = np.log(rank)
    logits -= logits.mean()
    return logits


def colley_logits(clean: pd.DataFrame, items: pd.Index) -> np.ndarray:
    n = len(items)
    idx = {item: i for i, item in enumerate(items)}
    wins = np.zeros(n, dtype=float)
    losses = np.zeros(n, dtype=float)
    games = np.zeros(n, dtype=float)
    pair_counts: dict[tuple[int, int], int] = {}

    for choice, a, b in clean[["choice", "item_a", "item_b"]].itertuples(index=False, name=None):
        i, j = idx[a], idx[b]
        games[i] += 1.0
        games[j] += 1.0
        if int(choice) == 1:
            wins[i] += 1.0
            losses[j] += 1.0
        else:
            wins[j] += 1.0
            losses[i] += 1.0
        key = (i, j) if i < j else (j, i)
        pair_counts[key] = pair_counts.get(key, 0) + 1

    C = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)
    for i in range(n):
        C[i, i] = 2.0 + games[i]
        b[i] = 1.0 + 0.5 * (wins[i] - losses[i])
    for (i, j), c in pair_counts.items():
        C[i, j] -= c
        C[j, i] -= c

    r = np.linalg.solve(C, b)
    r = np.clip(r, EPS, None)
    logits = np.log(r)
    logits -= logits.mean()
    return logits


def rank_centrality_logits(clean: pd.DataFrame, items: pd.Index) -> np.ndarray:
    n = len(items)
    idx = {item: i for i, item in enumerate(items)}
    wins = np.zeros((n, n), dtype=float)

    for choice, a, b in clean[["choice", "item_a", "item_b"]].itertuples(index=False, name=None):
        i, j = idx[a], idx[b]
        if int(choice) == 1:
            wins[i, j] += 1.0
        else:
            wins[j, i] += 1.0

    totals = wins + wins.T
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            denom = totals[i, j]
            M[i, j] = wins[j, i] / denom if denom > 0 else 0.0

    row_sums = M.sum(axis=1)
    d = float(np.max(row_sums))
    if not np.isfinite(d) or d <= 0:
        return np.zeros(n, dtype=float)

    P = M / d
    np.fill_diagonal(P, 1.0 - P.sum(axis=1))

    pi = np.full(n, 1.0 / n, dtype=float)
    for _ in range(PAGERANK_MAX_ITER):
        new = P.T @ pi
        s = new.sum()
        if s > 0:
            new /= s
        if np.linalg.norm(new - pi, ord=1) <= PAGERANK_TOL:
            pi = new
            break
        pi = new

    pi = np.clip(pi, EPS, None)
    logits = np.log(pi)
    logits -= logits.mean()
    return logits


def compute_all(prepared: pd.DataFrame, *, bt_temp: float) -> dict[str, pd.DataFrame]:
    clean = clean_pairs(prepared)
    items = unique_items(clean)

    out = {}
    out["bradley-terry"] = finalize(items, bradley_terry_logits(clean, items), temperature=bt_temp)
    out["huber-log"] = finalize(items, huber_logits(clean, items))
    out["elo"] = finalize(items, elo_logits(clean, items))
    out["pagerank"] = finalize(items, pagerank_logits(clean, items))
    out["colley"] = finalize(items, colley_logits(clean, items))
    out["rank-centrality"] = finalize(items, rank_centrality_logits(clean, items))
    for m in out:
        out[m].insert(0, "method", m)
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    raw = load_csvs(args.csv)
    out_dir = args.out or default_out_dir(args.csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_method_frames: dict[str, list[pd.DataFrame]] = {m: [] for m in METHODS}

    for comp_id, group in raw.groupby("competition_id", sort=False):
        prepared = prepare_pairs(group)
        if prepared.empty:
            continue

        try:
            results = compute_all(prepared, bt_temp=args.bt_temp)
        except ValueError:
            continue

        for method, df in results.items():
            df = df.copy()
            df.insert(0, "competition_id", comp_id)
            df["rank"] = df["weight"].rank(method="dense", ascending=False).astype(int)
            per_method_frames[method].append(df[["competition_id", "method", "rank", "item", "weight"]])

    wrote_any = False
    for method in METHODS:
        frames = per_method_frames[method]
        if not frames:
            continue
        combined = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["competition_id", "rank", "item"], ignore_index=True)
        )
        dest = out_dir / f"{method}.csv"
        combined.to_csv(dest, index=False)
        print(f"Wrote {dest}")
        wrote_any = True

    if not wrote_any:
        print("No valid comparisons found.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
