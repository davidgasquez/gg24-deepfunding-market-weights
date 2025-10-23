import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RunStats:
    files: tuple[Path, ...]
    total_comparisons: int
    repo_comparisons: Counter[str]
    repo_wins: Counter[str]
    juror_comparisons: Counter[str]
    total_cost: float


@dataclass(frozen=True)
class WeightEvaluation:
    method: str
    path: Path
    brier: float
    log_loss: float


_MIN_STRENGTH = 1.2
_MAX_STRENGTH = 6.0


def load_run_pairs(runs_dir: Path) -> pd.DataFrame:
    """Load and combine all run CSVs; ensure required columns exist."""
    csv_paths = sorted(p for p in runs_dir.glob("*.csv") if p.is_file())
    frames: list[pd.DataFrame] = []
    for p in csv_paths:
        try:
            frames.append(pd.read_csv(p))
        except pd.errors.EmptyDataError:
            continue

    if not frames:
        return pd.DataFrame(
            columns=["item_a", "item_b", "winner", "comparison_cost", "juror_id"]
        )

    df = pd.concat(frames, ignore_index=True)
    required = {"item_a", "item_b", "winner"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Runs data missing required columns: {', '.join(sorted(missing))}."
        )

    # Make optional columns present for downstream simplicity
    for col in ("comparison_cost", "juror_id"):
        if col not in df.columns:
            df[col] = pd.NA

    return df


def collect_stats(runs_dir: Path) -> RunStats:
    """Derive summary counters and totals from the combined runs."""
    files = tuple(sorted(runs_dir.glob("*.csv")))
    if not files:
        return RunStats(files, 0, Counter(), Counter(), Counter(), 0.0)

    df = load_run_pairs(runs_dir)

    # Repository comparison counts = counts in A + counts in B
    repo_counts_series = (
        df["item_a"]
        .astype(str)
        .value_counts()
        .add(df["item_b"].astype(str).value_counts(), fill_value=0)
        .astype(int)
    )
    repo_comparisons = Counter(repo_counts_series.to_dict())

    # Wins per repo
    repo_wins = Counter(df["winner"].astype(str).value_counts().astype(int).to_dict())

    # Juror comparison counts (optional column)
    juror_comparisons = Counter(
        df["juror_id"].dropna().astype(str).value_counts().astype(int).to_dict()
    )

    # Total cost
    total_cost = float(
        pd.to_numeric(df["comparison_cost"], errors="coerce").fillna(0.0).sum()
    )

    return RunStats(
        files=files,
        total_comparisons=len(df),
        repo_comparisons=repo_comparisons,
        repo_wins=repo_wins,
        juror_comparisons=juror_comparisons,
        total_cost=total_cost,
    )


def format_repository_line(
    repo: str, comparisons: int, wins: int, win_rate: float | None = None
) -> str:
    base = f"{repo}: {comparisons} comparisons, {wins} wins"
    return base if win_rate is None else f"{base}, {win_rate:.1%} win rate"


def least_common(counter: Counter[str], limit: int) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda it: (it[1], it[0]))[:limit]


def win_rate_table(
    comparisons: Counter[str], wins: Counter[str], *, min_comparisons: int
) -> list[tuple[str, int, int, float]]:
    return [
        (repo, total, wins.get(repo, 0), wins.get(repo, 0) / total)
        for repo, total in comparisons.items()
        if total >= min_comparisons
    ]


def print_stats(
    stats: RunStats,
    *,
    runs_dir: Path,
    top_n: int = 10,
    min_comparisons_for_win_rate: int = 25,
) -> None:
    if not stats.files:
        print(f"No CSV files found in {runs_dir}")
        return

    print(f"Processed {len(stats.files)} run file(s).")
    print(f"Total comparisons: {stats.total_comparisons}")
    print(f"Unique repositories: {len(stats.repo_comparisons)}")
    print(f"Unique jurors: {len(stats.juror_comparisons)}")
    if stats.total_cost:
        print(f"Total comparison cost: {stats.total_cost:.6f}")

    print("\nTop repositories by comparisons:")
    for repo, count in stats.repo_comparisons.most_common(top_n):
        print(f"  {format_repository_line(repo, count, stats.repo_wins.get(repo, 0))}")

    print("\nLeast compared repositories:")
    for repo, count in least_common(stats.repo_comparisons, top_n):
        print(f"  {format_repository_line(repo, count, stats.repo_wins.get(repo, 0))}")

    ranked = win_rate_table(
        stats.repo_comparisons,
        stats.repo_wins,
        min_comparisons=min_comparisons_for_win_rate,
    )
    if ranked:
        print(
            f"\nTop repositories by win rate (min {min_comparisons_for_win_rate} comparisons):"
        )
        for repo, total, wins, rate in sorted(
            ranked, key=lambda e: (-e[3], -e[1], e[0])
        )[:top_n]:
            print(f"  {format_repository_line(repo, total, wins, rate)}")

        print(
            f"\nLowest repositories by win rate (min {min_comparisons_for_win_rate} comparisons):"
        )
        for repo, total, wins, rate in sorted(ranked, key=lambda e: (e[3], e[1], e[0]))[
            :top_n
        ]:
            print(f"  {format_repository_line(repo, total, wins, rate)}")
    else:
        print(
            f"\nNo repositories with at least {min_comparisons_for_win_rate} comparisons to show win-rate rankings."
        )

    print("\nComparisons by juror:")
    for juror, count in stats.juror_comparisons.most_common():
        print(f"  {juror}: {count}")


def prepare_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Produce (item_a, item_b, choice, multiplier) with strength derived from cost."""
    pairs = frame.dropna(subset=["item_a", "item_b", "winner"]).copy()

    # Keep only rows where winner is one of the two items (defensive, but cheap)
    mask = (pairs["winner"] == pairs["item_a"]) | (pairs["winner"] == pairs["item_b"])
    pairs = pairs[mask]

    choice = np.where(pairs["winner"].eq(pairs["item_a"]), 1, 2)
    cost = pd.to_numeric(pairs["comparison_cost"], errors="coerce").fillna(0.0)
    strength = (1.0 + cost).clip(lower=_MIN_STRENGTH, upper=_MAX_STRENGTH)

    # If B (choice==2) wins, multiplier > 1; if A wins, multiplier < 1
    multiplier = np.where(choice == 2, strength, 1.0 / strength).astype(float)

    return pd.DataFrame({
        "item_a": pairs["item_a"].astype(str),
        "item_b": pairs["item_b"].astype(str),
        "choice": choice.astype(np.int64),
        "multiplier": multiplier,
    })


def clean_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Minimal cleaning to ensure valid numeric types and positivity."""
    clean = frame.dropna(subset=["item_a", "item_b", "choice", "multiplier"]).copy()
    clean["choice"] = pd.to_numeric(clean["choice"], errors="coerce").astype("Int64")
    clean["multiplier"] = pd.to_numeric(clean["multiplier"], errors="coerce")
    clean = clean[clean["choice"].isin([1, 2]) & (clean["multiplier"] > 0)]
    return clean.reset_index(drop=True)


def ground_truth_probabilities(clean: pd.DataFrame) -> np.ndarray:
    """
    Convert (choice, multiplier) into a probability that item_b is correct.

    NOTE: Bug fix vs original:
      Do NOT flip sign for choice==1. The multiplier is already <1 for A-wins
      and >1 for B-wins. Using log(multiplier) directly yields <0.5 when A wins
      and >0.5 when B wins, which aligns with the 'preds = P(B)' scoring below.
    """
    log_ratio = np.log(clean["multiplier"].to_numpy(dtype=float))
    log_ratio = np.clip(log_ratio, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-log_ratio))


def load_candidate_weights(path: Path) -> tuple[str, pd.Series]:
    """Load a weight CSV with columns (item, weight[, method]) and return normalized positive mass."""
    frame = pd.read_csv(path)
    missing = {"item", "weight"}.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Weights file missing required columns: {', '.join(sorted(missing))}."
        )

    weights = (
        frame[["item", "weight"]]
        .dropna()
        .assign(weight=lambda df: pd.to_numeric(df["weight"], errors="coerce"))
        .dropna(subset=["weight"])
    )
    aggregated = weights.groupby("item", sort=False)["weight"].sum()
    positive = aggregated[aggregated > 0.0]
    if positive.empty:
        raise ValueError("Weights must include positive mass.")
    normalized = positive / positive.sum()

    method = None
    if "method" in frame.columns:
        methods = frame["method"].dropna().astype(str).unique().tolist()
        if len(methods) == 1:
            method = methods[0]
    if method is None:
        method = path.stem

    return method, normalized


def score_weights(
    clean: pd.DataFrame, truth_probs: np.ndarray, weights: pd.Series
) -> dict[str, float]:
    w_a = weights.reindex(clean["item_a"]).fillna(0.0).to_numpy(dtype=float)
    w_b = weights.reindex(clean["item_b"]).fillna(0.0).to_numpy(dtype=float)
    total = w_a + w_b

    # P(B) prediction: if no mass on either item, use 0.5
    raw_preds = np.divide(
        w_b, total, out=np.full_like(total, 0.5, dtype=float), where=total > 0
    )
    preds = np.clip(raw_preds, 1e-12, 1.0 - 1e-12)
    tclip = np.clip(truth_probs, 1e-12, 1.0 - 1e-12)

    return {
        "brier": float(np.mean((raw_preds - truth_probs) ** 2)),
        "log_loss": float(
            -np.mean(tclip * np.log(preds) + (1.0 - tclip) * np.log(1.0 - preds))
        ),
    }


def evaluate_weight_distances(
    runs_dir: Path, weights_dir: Path | None
) -> list[WeightEvaluation]:
    if weights_dir is None or not weights_dir.exists():
        return []

    raw_pairs = load_run_pairs(runs_dir)
    prepared = prepare_pairs(raw_pairs)
    clean = clean_pairs(prepared)
    if clean.empty:
        return []

    truth_probs = ground_truth_probabilities(clean)

    evaluations: list[WeightEvaluation] = []
    for weights_path in sorted(weights_dir.glob("*.csv")):
        try:
            method, weights = load_candidate_weights(weights_path)
            metrics = score_weights(clean, truth_probs, weights)
        except (OSError, ValueError) as exc:
            print(f"[ERROR] Failed to score {weights_path}: {exc}")
            continue

        evaluations.append(
            WeightEvaluation(
                method=method,
                path=weights_path,
                brier=metrics["brier"],
                log_loss=metrics["log_loss"],
            )
        )
    return evaluations


def print_weight_evaluations(evaluations: list[WeightEvaluation]) -> None:
    if not evaluations:
        print("\nNo weight files evaluated.")
        return

    sorted_evals = sorted(evaluations, key=lambda e: e.method)
    method_width = max(len("Method"), max(len(ev.method) for ev in sorted_evals))

    print("\nWeight distance metrics (per weights file):")
    print(
        f"  {'Method'.ljust(method_width)}  {'Brier':>12}  {'LogLoss':>12}"
    )
    print(
        f"  {'-' * method_width}  {'-' * 12:>12}  {'-' * 12:>12}"
    )
    for ev in sorted_evals:
        print(
            f"  {ev.method.ljust(method_width)}  {ev.brier:12.6f}  {ev.log_loss:12.6f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print summary statistics for competition run CSV files."
    )
    parser.add_argument(
        "runs_path",
        type=Path,
        help="Directory containing the competition CSV run files.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="How many repositories to include in the summary tables (default: 10).",
    )
    parser.add_argument(
        "--min-comparisons",
        type=int,
        default=25,
        help="Minimum comparisons required to show a repository in win-rate tables (default: 25).",
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        help="Directory containing weights CSV files to score against the runs.",
    )
    args = parser.parse_args()

    runs_dir = args.runs_path.expanduser()
    if not runs_dir.exists():
        parser.error(f"Runs directory not found: {runs_dir}")
    if not runs_dir.is_dir():
        parser.error(f"Runs path is not a directory: {runs_dir}")
    args.runs_path = runs_dir

    # Default weights dir: sibling "weights" beside runs directory
    if args.weights_path is None:
        inferred = runs_dir.parent / "weights"
        args.weights_path = (
            inferred if inferred.exists() and inferred.is_dir() else None
        )
    else:
        weights_dir = args.weights_path.expanduser()
        if not weights_dir.exists():
            parser.error(f"Weights directory not found: {weights_dir}")
        if not weights_dir.is_dir():
            parser.error(f"Weights path is not a directory: {weights_dir}")
        args.weights_path = weights_dir

    return args


def main() -> None:
    args = parse_args()

    stats = collect_stats(args.runs_path)
    print_stats(
        stats,
        runs_dir=args.runs_path,
        top_n=args.top_n,
        min_comparisons_for_win_rate=args.min_comparisons,
    )

    evaluations = evaluate_weight_distances(args.runs_path, args.weights_path)
    print_weight_evaluations(evaluations)


if __name__ == "__main__":
    main()
