import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

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
DEFAULT_METHOD: WeightMethod = "least-squares"
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
_MIN_STRENGTH = 1.2
_MAX_STRENGTH = 6.0
_MAD_SCALE = 0.6744897501960817
_RIDGE_EPS = 1e-6
_MAX_IRLS_ITER = 75
_IRLS_TOL = 1e-8
_BT_REGULARIZATION = 0.5
_ELO_K = 16.0
_PAGERANK_DAMPING = 0.85
_PAGERANK_TOL = 1e-10
_PAGERANK_MAX_ITER = 200


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute item ranks and weights from Arbitron comparisons."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help=(
            "Path to a comparison CSV or a directory containing CSV files "
            "with Arbitron's export schema."
        ),
    )
    parser.add_argument(
        "--method",
        choices=WEIGHT_METHODS,
        action="append",
        help="Weighting method to apply. Defaults to least-squares unless --all-methods is used.",
    )
    parser.add_argument(
        "--all-methods",
        action="store_true",
        help="Run every available weighting method.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV destination for the aggregated results. Defaults to stdout.",
    )
    return parser.parse_args(argv)


def _load_single_csv(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=sorted(REQUIRED_COLUMNS))
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        msg = f"Missing required columns: {missing_columns}."
        raise ValueError(msg)
    return frame


def load_comparisons(path: Path) -> pd.DataFrame:
    if path.is_file():
        return _load_single_csv(path)
    if path.is_dir():
        csv_files = sorted(
            file_path for file_path in path.glob("*.csv") if file_path.is_file()
        )
        if not csv_files:
            msg = f"No CSV files found in directory {path}."
            raise FileNotFoundError(msg)
        frames = [_load_single_csv(file_path) for file_path in csv_files]
        non_empty_frames = [frame for frame in frames if not frame.empty]
        if not non_empty_frames:
            return pd.DataFrame(columns=sorted(REQUIRED_COLUMNS))
        return pd.concat(non_empty_frames, ignore_index=True)
    msg = f"Input path {path} does not exist."
    raise FileNotFoundError(msg)


def prepare_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    pairs = frame.dropna(subset=["item_a", "item_b", "winner"]).copy()
    mask = (pairs["winner"] == pairs["item_a"]) | (pairs["winner"] == pairs["item_b"])
    if not mask.any():
        return pd.DataFrame(columns=["item_a", "item_b", "choice", "multiplier"])
    pairs = pairs[mask]
    choice = np.where(pairs["winner"] == pairs["item_a"], 1, 2)
    cost = pd.to_numeric(pairs["comparison_cost"], errors="coerce").fillna(0.0)
    strength = 1.0 + cost.clip(lower=0.0)
    strength = np.clip(strength, _MIN_STRENGTH, _MAX_STRENGTH)
    base_multiplier = np.where(choice == 2, strength, 1.0 / strength)
    prepared = pd.DataFrame({
        "item_a": pairs["item_a"].astype(str),
        "item_b": pairs["item_b"].astype(str),
        "choice": choice.astype(np.int64),
        "multiplier": base_multiplier.astype(float),
    })
    timestamps = pd.to_datetime(
        pairs["comparison_created_at"], errors="coerce", utc=True
    )
    if isinstance(timestamps, pd.Series):
        mask = timestamps.notna()
        if mask.any():
            prepared["timestamp"] = timestamps
    return prepared


def clean_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"item_a", "item_b", "choice", "multiplier"}
    missing = required.difference(frame.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        msg = f"Missing prepared columns: {missing_columns}."
        raise ValueError(msg)
    clean = frame.dropna(subset=list(required)).copy()
    clean["choice"] = pd.to_numeric(clean["choice"], errors="coerce").astype("Int64")
    clean["multiplier"] = pd.to_numeric(clean["multiplier"], errors="coerce")
    clean = clean[clean["choice"].isin([1, 2]) & (clean["multiplier"] > 0)]
    if clean.empty:
        raise ValueError("No valid comparisons available after cleaning.")
    return clean.reset_index(drop=True)


def design_matrix(
    clean: pd.DataFrame, items: pd.Index
) -> tuple[np.ndarray, np.ndarray]:
    item_to_index = {item: idx for idx, item in enumerate(items)}
    comparisons = len(clean)
    design = np.zeros((comparisons, len(items)), dtype=float)
    ratios = np.log(clean["multiplier"].to_numpy(dtype=float))
    choices = clean["choice"].to_numpy(dtype=int)
    ratios[choices == 1] *= -1

    for row_idx, (item_a, item_b) in enumerate(
        clean[["item_a", "item_b"]].itertuples(index=False, name=None)
    ):
        design[row_idx, item_to_index[item_b]] = 1.0
        design[row_idx, item_to_index[item_a]] = -1.0

    return design, ratios


def solve_logits(design: np.ndarray, ratios: np.ndarray) -> np.ndarray:
    solution, *_ = np.linalg.lstsq(design, ratios, rcond=None)
    return solution


def normalize_logits(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:
        return logits
    shifted = logits - np.max(logits)
    positive = np.exp(shifted)
    total = positive.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Failed to compute finite weights from logits.")
    return positive / total


def solve_method_logits(
    clean: pd.DataFrame, items: pd.Index, method: WeightMethod
) -> tuple[np.ndarray, SolverMetadata | None]:
    try:
        solver = METHOD_BUILDERS[method]
    except KeyError as exc:  # pragma: no cover - defensive branch
        expected = ", ".join(WEIGHT_METHODS)
        msg = f"Unknown method {method!r}. Expected one of: {expected}."
        raise ValueError(msg) from exc
    result = solver(clean, items)
    if isinstance(result, tuple):
        logits, metadata = result
    else:
        logits, metadata = result, None
    return logits, metadata


def compute_weights(
    frame: pd.DataFrame, *, method: WeightMethod = DEFAULT_METHOD
) -> pd.DataFrame:
    clean = clean_pairs(frame)
    items = unique_items(clean)
    logits, _ = solve_method_logits(clean, items, method)
    return finalize_weights(items, logits)


def least_squares_logits(clean: pd.DataFrame, items: pd.Index) -> np.ndarray:
    design, ratios = design_matrix(clean, items)
    logits = solve_logits(design, ratios)
    return logits


def solve_bradley_terry(
    clean: pd.DataFrame,
    items: pd.Index,
    *,
    ridge: float,
    learning_rate: float = 0.065,
    max_iter: int = 4000,
    tol: float = 1e-5,
) -> tuple[np.ndarray, int, float]:
    item_to_index = {item: idx for idx, item in enumerate(items)}
    idx_a = clean["item_a"].map(item_to_index)
    idx_b = clean["item_b"].map(item_to_index)
    if idx_a.isna().any() or idx_b.isna().any():
        missing = clean.loc[idx_a.isna() | idx_b.isna(), ["item_a", "item_b"]]
        missing_pairs = ", ".join(
            sorted({f"{row.item_a}-{row.item_b}" for row in missing.itertuples(index=False)})
        )
        msg = f"Pairs reference unknown items: {missing_pairs}."
        raise ValueError(msg)
    idx_a = idx_a.to_numpy(dtype=np.int64)
    idx_b = idx_b.to_numpy(dtype=np.int64)
    outcomes = (clean["choice"].to_numpy(dtype=int) == 1).astype(float)
    logits = np.zeros(len(items), dtype=float)
    gradient = np.zeros_like(logits)
    grad_norm = float("inf")

    for iteration in range(max_iter):
        diff = logits[idx_a] - logits[idx_b]
        np.clip(diff, -30.0, 30.0, out=diff)
        probs = 1.0 / (1.0 + np.exp(-diff))
        error = outcomes - probs
        gradient.fill(0.0)
        np.add.at(gradient, idx_a, error)
        np.add.at(gradient, idx_b, -error)
        gradient -= ridge * logits
        grad_norm = float(np.linalg.norm(gradient))
        step = learning_rate / (1.0 + 0.01 * iteration)
        logits += step * gradient
        logits -= logits.mean()
        if grad_norm <= tol:
            return logits, iteration + 1, grad_norm

    return logits, max_iter, grad_norm


def bradley_terry_regularized_logits(
    clean: pd.DataFrame, items: pd.Index
) -> tuple[np.ndarray, SolverMetadata]:
    ridge = max(_BT_REGULARIZATION, _RIDGE_EPS)
    logits, iterations, grad_norm = solve_bradley_terry(clean, items, ridge=ridge)
    return logits, {"iterations": iterations, "gradient_norm": grad_norm}


def huber_llsm_logits(clean: pd.DataFrame, items: pd.Index) -> np.ndarray:
    design, ratios = design_matrix(clean, items)
    delta = huber_delta(ratios)
    logits = np.zeros(design.shape[1], dtype=float)
    identity = np.eye(design.shape[1], dtype=float)

    for _ in range(_MAX_IRLS_ITER):
        residual = ratios - design @ logits
        weights = huber_weights(residual, delta)
        lhs = design.T @ (weights[:, None] * design) + identity * _RIDGE_EPS
        rhs = design.T @ (weights * ratios)
        updated = np.linalg.solve(lhs, rhs)
        updated -= updated.mean()
        if np.linalg.norm(updated - logits) <= _IRLS_TOL * (
            1.0 + np.linalg.norm(logits)
        ):
            logits = updated
            break
        logits = updated
    else:
        logits -= logits.mean()

    return logits


def elo_logits(clean: pd.DataFrame, items: pd.Index) -> np.ndarray:
    item_to_index = {item: idx for idx, item in enumerate(items)}
    ratings = np.zeros(len(items), dtype=float)

    ordered = clean.copy()
    if "timestamp" in ordered.columns:
        ordered = ordered.assign(
            _timestamp=pd.to_datetime(ordered["timestamp"], errors="coerce")
        )
        ordered = ordered.sort_values("_timestamp", kind="stable").drop(
            columns=["_timestamp"]
        )

    for row in ordered.itertuples(index=False):
        item_a = getattr(row, "item_a")
        item_b = getattr(row, "item_b")
        choice = int(getattr(row, "choice"))
        if item_a not in item_to_index or item_b not in item_to_index:
            continue
        if choice == 1:
            winner_idx, loser_idx = item_to_index[item_a], item_to_index[item_b]
        else:
            winner_idx, loser_idx = item_to_index[item_b], item_to_index[item_a]
        winner_rating = ratings[winner_idx]
        loser_rating = ratings[loser_idx]
        diff = winner_rating - loser_rating
        expected = 1.0 / (1.0 + np.exp(-diff))
        delta = _ELO_K * (1.0 - expected)
        ratings[winner_idx] = winner_rating + delta
        ratings[loser_idx] = loser_rating - delta

    ratings -= ratings.mean()
    scale = np.std(ratings)
    if np.isfinite(scale) and scale > 1.0:
        ratings /= scale
    return ratings


def pagerank_logits(clean: pd.DataFrame, items: pd.Index) -> np.ndarray:
    item_to_index = {item: idx for idx, item in enumerate(items)}
    n = len(items)
    adjacency = np.zeros((n, n), dtype=float)

    for choice, item_a, item_b, multiplier in clean[
        ["choice", "item_a", "item_b", "multiplier"]
    ].itertuples(index=False, name=None):
        if item_a not in item_to_index or item_b not in item_to_index:
            continue
        if int(choice) == 1:
            winner_idx = item_to_index[item_a]
            loser_idx = item_to_index[item_b]
        else:
            winner_idx = item_to_index[item_b]
            loser_idx = item_to_index[item_a]
        weight = float(multiplier)
        if not np.isfinite(weight) or weight <= 0:
            weight = 1.0
        adjacency[loser_idx, winner_idx] += weight

    row_sums = adjacency.sum(axis=1, keepdims=True)
    transition = np.divide(
        adjacency,
        np.where(row_sums <= 0, 1.0, row_sums),
        out=np.full_like(adjacency, 1.0 / n),
        where=row_sums > 0,
    )
    rank = np.full(n, 1.0 / n, dtype=float)

    teleport = (1.0 - _PAGERANK_DAMPING) / n
    for _ in range(_PAGERANK_MAX_ITER):
        updated = teleport + _PAGERANK_DAMPING * transition.T @ rank
        if np.linalg.norm(updated - rank, ord=1) <= _PAGERANK_TOL:
            rank = updated
            break
        rank = updated

    rank = np.clip(rank, 1e-12, None)
    logits = np.log(rank)
    logits -= logits.mean()
    return logits


def huber_delta(values: np.ndarray) -> float:
    mad = median_absolute_deviation(values)
    if not np.isfinite(mad) or mad <= 0:
        mad = np.mean(np.abs(values - np.mean(values))) if values.size else 1.0
    if not np.isfinite(mad) or mad <= 0:
        mad = 1.0
    sigma = mad / _MAD_SCALE
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return float(1.345 * sigma)


def median_absolute_deviation(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def huber_weights(residual: np.ndarray, delta: float) -> np.ndarray:
    weights = np.ones_like(residual, dtype=float)
    abs_residual = np.abs(residual)
    mask = abs_residual > delta
    weights[mask] = delta / np.maximum(abs_residual[mask], 1e-12)
    return weights


def unique_items(clean: pd.DataFrame) -> pd.Index:
    items_series = pd.concat(
        [clean["item_a"], clean["item_b"]], ignore_index=True
    ).dropna()
    items = pd.Index(pd.unique(items_series))
    if items.empty:
        raise ValueError("No items present in the input data.")
    return items


def finalize_weights(items: pd.Index, logits: np.ndarray) -> pd.DataFrame:
    weights = normalize_logits(logits)
    if np.any(~np.isfinite(weights)):
        raise ValueError("Computed weights contain non-finite values.")
    return pd.DataFrame({"item": items, "weight": weights}).sort_values(
        "weight", ascending=False, ignore_index=True
    )


def log_comparison_summary(
    competition_id: object, *, used: int, total: int
) -> None:
    message = f"Comparisons {competition_id}: {used}/{total}"
    print(message, file=sys.stderr)


def _format_metadata(metadata: SolverMetadata) -> str:
    parts: list[str] = []
    for key, value in metadata.items():
        if isinstance(value, float):
            formatted = f"{value:.3g}"
        elif isinstance(value, int):
            formatted = f"{value}"
        else:  # pragma: no cover - defensive branch
            formatted = str(value)
        parts.append(f"{key}={formatted}")
    return ", ".join(parts)


def log_method_status(
    competition_id: object, method: str, metadata: SolverMetadata | None
) -> None:
    details = _format_metadata(metadata) if metadata else ""
    suffix = f"; {details}" if details else ""
    print(f"Method {competition_id}/{method}: completed{suffix}", file=sys.stderr)


METHOD_BUILDERS: dict[
    WeightMethod, Callable[[pd.DataFrame, pd.Index], SolverOutput]
] = {
    "least-squares": least_squares_logits,
    "bradley-terry-regularized": bradley_terry_regularized_logits,
    "huber-log": huber_llsm_logits,
    "elo": elo_logits,
    "pagerank": pagerank_logits,
}


def select_methods(selected: list[str] | None, all_methods: bool) -> list[str]:
    if all_methods or not selected:
        return list(WEIGHT_METHODS) if all_methods else [DEFAULT_METHOD]
    ordered = dict.fromkeys(selected)
    return [method for method in ordered if method in WEIGHT_METHODS]


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
        initial_count = len(prepared)
        try:
            clean = clean_pairs(prepared)
        except ValueError:
            continue
        used_count = len(clean)
        if used_count == 0:
            continue
        items = unique_items(clean)
        log_comparison_summary(
            competition_id,
            used=used_count,
            total=initial_count,
        )
        for method in methods:
            logits, metadata = solve_method_logits(clean, items, method)
            weights = finalize_weights(items, logits)
            weights.insert(0, "method", method)
            weights.insert(0, "competition_id", competition_id)
            weights["rank"] = (
                weights["weight"].rank(method="dense", ascending=False).astype("Int64")
            )
            weights = weights[["competition_id", "method", "rank", "item", "weight"]]
            results.append(weights)
            log_method_status(competition_id, method, metadata)

    if not results:
        print("No valid comparisons found.", file=sys.stderr)
        return 1

    combined = pd.concat(results, ignore_index=True).sort_values(
        ["competition_id", "method", "rank", "item"], ignore_index=True
    )
    combined["rank"] = combined["rank"].astype(int)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.output, index=False)
        print(f"Wrote weights to {args.output}")
    else:
        combined.to_csv(sys.stdout, index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
