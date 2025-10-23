import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd

WeightMethod = Literal[
    "least-squares",
    "bradley-terry-regularized",
    "huber-log",
    "elo",
    "pagerank",
]
SolverMetadata = dict[str, float | int]
SolverOutput = np.ndarray | tuple[np.ndarray, SolverMetadata]
SolverBuilder = Callable[..., SolverOutput]

WEIGHT_METHODS: tuple[WeightMethod, ...] = (
    "least-squares",
    "bradley-terry-regularized",
    "huber-log",
    "elo",
    "pagerank",
)

REQUIRED_COLUMNS = {
    "competition_id",
    "juror_id",
    "item_a",
    "item_b",
    "winner",
    "comparison_created_at",
    "comparison_cost",
}

# These only affect prepare_pairs and non-LS/Huber methods.
_MIN_STRENGTH = 1.2
_MAX_STRENGTH = 6.0

# Robust stats constants
_MAD_SCALE = 0.6744897501960817  # Phi^{-1}(0.75)
_MAX_IRLS_ITER = 75
_IRLS_TOL = 1e-8

# Bradley–Terry / PageRank / Elo
_RIDGE_EPS = 1e-6
_BT_RIDGE_DEFAULT = 0.5
_ELO_K = 16.0
_PAGERANK_DAMPING = 0.85
_PAGERANK_TOL = 1e-10
_PAGERANK_MAX_ITER = 200


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute item ranks and weights from Arbitron comparisons."
    )
    p.add_argument(
        "--csv",
        type=Path,
        required=True,
        help=(
            "Path to a comparison CSV or a directory of CSVs with Arbitron's export schema."
        ),
    )
    p.add_argument(
        "--method",
        choices=WEIGHT_METHODS,
        action="append",
        help="Weighting method to apply. Defaults to all methods unless explicitly set.",
    )
    p.add_argument(
        "--all-methods", action="store_true", help="Run every available method."
    )
    p.add_argument(
        "--bt-ridge",
        type=float,
        default=_BT_RIDGE_DEFAULT,
        help="Bradley–Terry ridge regularization strength (L2 prior).",
    )
    p.add_argument(
        "--out",
        type=Path,
        help="Directory where per-method CSV files will be written. Defaults to stdout.",
    )
    return p.parse_args(argv)


def _load_single_csv(path: Path) -> pd.DataFrame:
    # Keep a small sanity check for schema; otherwise assume well-formed data.
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}.")
    return frame


def load_comparisons(path: Path) -> pd.DataFrame:
    if path.is_file():
        return _load_single_csv(path)
    if path.is_dir():
        csvs = sorted(p for p in path.glob("*.csv") if p.is_file())
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in directory {path}.")
        frames = [_load_single_csv(p) for p in csvs]
        non_empty = [f for f in frames if not f.empty]
        return (
            pd.concat(non_empty, ignore_index=True)
            if non_empty
            else pd.DataFrame(columns=sorted(REQUIRED_COLUMNS))
        )
    raise FileNotFoundError(f"Input path {path} does not exist.")


def prepare_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    # Assume good data; filter rows where winner is one of the items.
    pairs = frame.dropna(subset=["item_a", "item_b", "winner"]).copy()
    m = (pairs["winner"] == pairs["item_a"]) | (pairs["winner"] == pairs["item_b"])
    if not m.any():
        return pd.DataFrame(columns=["item_a", "item_b", "choice", "multiplier"])

    pairs = pairs[m]
    choice = np.where(pairs["winner"] == pairs["item_a"], 1, 2)

    # Keep multiplier for methods that still use it (e.g., PageRank); LS/Huber ignore it.
    cost = pd.to_numeric(pairs["comparison_cost"], errors="coerce").fillna(0.0)
    strength = np.clip(1.0 + cost.clip(lower=0.0), _MIN_STRENGTH, _MAX_STRENGTH)
    multiplier = np.where(choice == 2, strength, 1.0 / strength).astype(float)

    prepared = pd.DataFrame({
        "item_a": pairs["item_a"].astype(str),
        "item_b": pairs["item_b"].astype(str),
        "choice": choice.astype(np.int64),
        "multiplier": multiplier,
    })

    ts = pd.to_datetime(pairs["comparison_created_at"], errors="coerce", utc=True)
    if isinstance(ts, pd.Series):
        ts_series = ts
    else:
        ts_series = pd.Series(ts, index=pairs.index)
    if ts_series.notna().any():
        prepared["timestamp"] = ts_series
    return prepared


def clean_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"item_a", "item_b", "choice", "multiplier"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing prepared columns: {', '.join(sorted(missing))}.")
    clean = frame.dropna(subset=list(required)).copy()
    clean["choice"] = pd.to_numeric(clean["choice"], errors="coerce").astype("Int64")
    clean["multiplier"] = pd.to_numeric(clean["multiplier"], errors="coerce")
    clean = clean[clean["choice"].isin([1, 2]) & (clean["multiplier"] > 0)]
    if clean.empty:
        raise ValueError("No valid comparisons available after cleaning.")
    return clean.reset_index(drop=True)


def unique_items(clean: pd.DataFrame) -> pd.Index:
    items = pd.Index(
        pd.unique(pd.concat([clean["item_a"], clean["item_b"]], ignore_index=True))
    )
    if items.empty:
        raise ValueError("No items present in the input data.")
    return items


def design_matrix(
    clean: pd.DataFrame, items: pd.Index
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the classic pairwise-difference system:
      X @ w ≈ y, where each row encodes (w_b - w_a) and
      y ∈ {-1, +1}: +1 if item_b wins, -1 if item_a wins.
    """
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


def normalize_logits(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:
        return logits
    z = logits - np.max(logits)
    p = np.exp(z)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Failed to compute finite weights from logits.")
    return p / s


def finalize_weights(items: pd.Index, logits: np.ndarray) -> pd.DataFrame:
    w = normalize_logits(logits)
    if np.any(~np.isfinite(w)):
        raise ValueError("Computed weights contain non-finite values.")
    return pd.DataFrame({"item": items, "weight": w}).sort_values(
        "weight", ascending=False, ignore_index=True
    )


def median_absolute_deviation(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def least_squares_logits(clean: pd.DataFrame, items: pd.Index) -> np.ndarray:
    """
    Vanilla least squares on pairwise differences:
      minimize || X w - y ||_2, with y ∈ {-1, +1}.
    """
    X, y = design_matrix(clean, items)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    w -= w.mean()  # fix identifiability (softmax is shift-invariant)
    return w


def solve_bradley_terry(
    clean: pd.DataFrame,
    items: pd.Index,
    *,
    ridge: float,
    learning_rate: float = 0.065,
    max_iter: int = 4000,
    tol: float = 1e-5,
) -> tuple[np.ndarray, int, float]:
    idx = {item: i for i, item in enumerate(items)}
    ia = clean["item_a"].map(idx).to_numpy(dtype=np.int64)
    ib = clean["item_b"].map(idx).to_numpy(dtype=np.int64)
    y = (clean["choice"].to_numpy(dtype=int) == 1).astype(float)

    w = np.zeros(len(items), dtype=float)
    g = np.zeros_like(w)

    for it in range(max_iter):
        diff = w[ia] - w[ib]
        np.clip(diff, -30.0, 30.0, out=diff)  # numerical safety
        p = 1.0 / (1.0 + np.exp(-diff))
        err = y - p

        g.fill(0.0)
        np.add.at(g, ia, err)
        np.add.at(g, ib, -err)
        g -= ridge * w

        step = learning_rate / (1.0 + 0.01 * it)
        w += step * g
        w -= w.mean()

        gnorm = float(np.linalg.norm(g))
        if gnorm <= tol:
            return w, it + 1, gnorm

    return w, max_iter, float(np.linalg.norm(g))


def bradley_terry_regularized_logits(
    clean: pd.DataFrame, items: pd.Index, *, ridge: float = _BT_RIDGE_DEFAULT
) -> tuple[np.ndarray, SolverMetadata]:
    ridge = max(ridge, _RIDGE_EPS)
    logits, iters, grad_norm = solve_bradley_terry(clean, items, ridge=ridge)
    return logits, {"iterations": iters, "gradient_norm": grad_norm}


def huber_llsm_logits(clean: pd.DataFrame, items: pd.Index) -> np.ndarray:
    """
    Robust least squares via IRLS with Huber weights on {−1,+1} targets.
    Parameter-free aside from the standard 1.345 factor; no ridge.
    """
    X, y = design_matrix(clean, items)
    # Start from the OLS solution
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    w -= w.mean()

    for _ in range(_MAX_IRLS_ITER):
        r = y - X @ w

        # Robust scale (MAD) and Huber threshold (standard choice)
        mad = median_absolute_deviation(r)
        sigma = (
            mad / _MAD_SCALE
            if (np.isfinite(mad) and mad > 0)
            else np.mean(np.abs(r - np.mean(r)))
        )
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0
        delta = 1.345 * float(sigma)

        # Huber weights
        a = np.abs(r)
        wts = np.ones_like(r)
        mask = a > delta
        wts[mask] = delta / np.maximum(a[mask], 1e-12)

        # Weighted least squares without ridge
        s = np.sqrt(wts)
        Xw = X * s[:, None]
        yw = y * s
        w_new, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        w_new -= w_new.mean()

        if np.linalg.norm(w_new - w) <= _IRLS_TOL * (1.0 + np.linalg.norm(w)):
            w = w_new
            break
        w = w_new

    return w


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
        a, b, choice = (
            getattr(row, "item_a"),
            getattr(row, "item_b"),
            int(getattr(row, "choice")),
        )
        if a not in idx or b not in idx:
            continue
        win, lose = (idx[a], idx[b]) if choice == 1 else (idx[b], idx[a])
        diff = ratings[win] - ratings[lose]
        expected = 1.0 / (1.0 + np.exp(-diff))
        delta = _ELO_K * (1.0 - expected)
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

    # Edge from loser -> winner with weight=multiplier
    for choice, a, b, mult in clean[
        ["choice", "item_a", "item_b", "multiplier"]
    ].itertuples(index=False, name=None):
        if a not in idx or b not in idx:
            continue
        win = idx[a] if int(choice) == 1 else idx[b]
        lose = idx[b] if int(choice) == 1 else idx[a]
        w = float(mult)
        A[lose, win] += w if np.isfinite(w) and w > 0 else 1.0

    row_sums = A.sum(axis=1, keepdims=True)
    T = np.divide(
        A,
        np.where(row_sums <= 0, 1.0, row_sums),
        out=np.full_like(A, 1.0 / n),
        where=row_sums > 0,
    )

    rank = np.full(n, 1.0 / n, dtype=float)
    teleport = (1.0 - _PAGERANK_DAMPING) / n

    for _ in range(_PAGERANK_MAX_ITER):
        new = teleport + _PAGERANK_DAMPING * (T.T @ rank)
        if np.linalg.norm(new - rank, ord=1) <= _PAGERANK_TOL:
            rank = new
            break
        rank = new

    rank = np.clip(rank, 1e-12, None)
    logits = np.log(rank)
    return logits - logits.mean()


def solve_method_logits(
    clean: pd.DataFrame,
    items: pd.Index,
    method: WeightMethod,
    *,
    bt_ridge: float = _BT_RIDGE_DEFAULT,
) -> tuple[np.ndarray, SolverMetadata | None]:
    try:
        solver = METHOD_BUILDERS[method]
    except KeyError as exc:  # pragma: no cover - defensive branch
        expected = ", ".join(WEIGHT_METHODS)
        raise ValueError(
            f"Unknown method {method!r}. Expected one of: {expected}."
        ) from exc

    if method == "bradley-terry-regularized":
        ridge = max(bt_ridge, _RIDGE_EPS)
        out = solver(clean, items, ridge=ridge)
    else:
        out = solver(clean, items)

    return (out[0], out[1]) if isinstance(out, tuple) else (out, None)


def compute_weights(
    frame: pd.DataFrame,
    *,
    methods: Sequence[WeightMethod] | None = None,
    bt_ridge: float = _BT_RIDGE_DEFAULT,
) -> pd.DataFrame:
    """
    Compute normalized weights for each item using the selected methods.
    Expects a *prepared* and then cleaned (via clean_pairs) DataFrame.
    """
    clean = clean_pairs(frame)
    items = unique_items(clean)
    selected = list(methods) if methods else list(WEIGHT_METHODS)

    outputs: list[pd.DataFrame] = []
    for method in selected:
        logits, _ = solve_method_logits(clean, items, method, bt_ridge=bt_ridge)
        weights = finalize_weights(items, logits)
        weights.insert(0, "method", method)
        outputs.append(weights)
    if not outputs:
        raise ValueError("No weighting methods selected.")
    return pd.concat(outputs, ignore_index=True)


def log_comparison_summary(competition_id: object, *, used: int, total: int) -> None:
    print(f"Comparisons {competition_id}: {used}/{total}", file=sys.stderr)


def _format_metadata(metadata: SolverMetadata) -> str:
    out = []
    for k, v in metadata.items():
        out.append(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}")
    return ", ".join(out)


def log_method_status(
    competition_id: object, method: str, metadata: SolverMetadata | None
) -> None:
    suffix = f"; {_format_metadata(metadata)}" if metadata else ""
    print(f"Method {competition_id}/{method}: completed{suffix}", file=sys.stderr)


METHOD_BUILDERS: dict[WeightMethod, SolverBuilder] = {
    "least-squares": least_squares_logits,
    "bradley-terry-regularized": bradley_terry_regularized_logits,
    "huber-log": huber_llsm_logits,
    "elo": elo_logits,
    "pagerank": pagerank_logits,
}


def select_methods(selected: list[str] | None, all_methods: bool) -> list[WeightMethod]:
    if all_methods or not selected:
        return list(WEIGHT_METHODS)
    # Preserve order, drop duplicates, filter invalid
    return [
        cast(WeightMethod, m) for m in dict.fromkeys(selected) if m in WEIGHT_METHODS
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    frame = load_comparisons(args.csv)
    methods = select_methods(args.method, args.all_methods)

    if not methods:
        requested = ", ".join(args.method or [])
        available = ", ".join(WEIGHT_METHODS)
        print(
            f"No valid weighting methods selected. Requested: {requested or 'none'}. "
            f"Available: {available}.",
            file=sys.stderr,
        )
        return 1

    results: list[pd.DataFrame] = []

    for competition_id, group in frame.groupby("competition_id", sort=False):
        prepared = prepare_pairs(group)
        if prepared.empty:
            continue

        initial = len(prepared)
        try:
            clean = clean_pairs(prepared)
        except ValueError:
            continue

        used = len(clean)
        if used == 0:
            continue

        items = unique_items(clean)
        log_comparison_summary(competition_id, used=used, total=initial)

        for method in methods:
            logits, meta = solve_method_logits(
                clean, items, method, bt_ridge=args.bt_ridge
            )
            weights = finalize_weights(items, logits)
            weights.insert(0, "method", method)
            weights.insert(0, "competition_id", competition_id)
            weights["rank"] = (
                weights["weight"].rank(method="dense", ascending=False).astype("Int64")
            )
            results.append(
                weights[["competition_id", "method", "rank", "item", "weight"]]
            )
            log_method_status(competition_id, method, meta)

    if not results:
        print("No valid comparisons found.", file=sys.stderr)
        return 1

    combined = pd.concat(results, ignore_index=True).sort_values(
        ["competition_id", "method", "rank", "item"], ignore_index=True
    )
    combined["rank"] = combined["rank"].astype(int)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for method_name, group in combined.groupby("method", sort=False):
            dest = args.out / f"{method_name}.csv"
            group.to_csv(dest, index=False)
            print(f"Wrote weights to {dest}")
    else:
        combined.to_csv(sys.stdout, index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
