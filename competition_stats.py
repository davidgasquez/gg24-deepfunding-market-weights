from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "data" / "runs"


@dataclass(frozen=True)
class RunStats:
    files: tuple[Path, ...]
    total_comparisons: int
    repo_comparisons: Counter[str]
    repo_wins: Counter[str]
    juror_comparisons: Counter[str]
    total_cost: float


def collect_stats(runs_dir: Path) -> RunStats:
    csv_files = tuple(sorted(runs_dir.glob("*.csv")))
    repo_comparisons: Counter[str] = Counter()
    repo_wins: Counter[str] = Counter()
    juror_comparisons: Counter[str] = Counter()
    total_cost = 0.0
    total_comparisons = 0

    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                item_a = row.get("item_a")
                item_b = row.get("item_b")
                winner = row.get("winner")
                juror_id = row.get("juror_id")
                cost = row.get("comparison_cost")

                if item_a:
                    repo_comparisons[item_a] += 1
                if item_b:
                    repo_comparisons[item_b] += 1
                if winner:
                    repo_wins[winner] += 1
                if juror_id:
                    juror_comparisons[juror_id] += 1
                if cost:
                    try:
                        total_cost += float(cost)
                    except ValueError:
                        pass

                total_comparisons += 1

    return RunStats(
        files=csv_files,
        total_comparisons=total_comparisons,
        repo_comparisons=repo_comparisons,
        repo_wins=repo_wins,
        juror_comparisons=juror_comparisons,
        total_cost=total_cost,
    )


def format_repository_line(
    repo: str,
    comparisons: int,
    wins: int,
    win_rate: float | None = None,
) -> str:
    base = f"{repo}: {comparisons} comparisons, {wins} wins"
    if win_rate is None:
        return base
    return f"{base}, {win_rate:.1%} win rate"


def least_common(counter: Counter[str], limit: int) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (item[1], item[0]))[:limit]


def win_rate_table(
    comparisons: Counter[str],
    wins: Counter[str],
    *,
    min_comparisons: int,
) -> list[tuple[str, int, int, float]]:
    entries: list[tuple[str, int, int, float]] = []
    for repo, total in comparisons.items():
        if total < min_comparisons or total == 0:
            continue
        win_count = wins.get(repo, 0)
        rate = win_count / total
        entries.append((repo, total, win_count, rate))
    return entries


def print_stats(
    stats: RunStats,
    *,
    top_n: int = 10,
    min_comparisons_for_win_rate: int = 25,
) -> None:
    if not stats.files:
        print(f"No CSV files found in {RUNS_DIR}")
        return

    print(f"Processed {len(stats.files)} run file(s).")
    print(f"Total comparisons: {stats.total_comparisons}")
    print(f"Unique repositories: {len(stats.repo_comparisons)}")
    print(f"Unique jurors: {len(stats.juror_comparisons)}")
    if stats.total_cost:
        print(f"Total comparison cost: {stats.total_cost:.6f}")

    print("\nTop repositories by comparisons:")
    for repo, count in stats.repo_comparisons.most_common(top_n):
        wins = stats.repo_wins.get(repo, 0)
        print(f"  {format_repository_line(repo, count, wins)}")

    print("\nLeast compared repositories:")
    for repo, count in least_common(stats.repo_comparisons, top_n):
        wins = stats.repo_wins.get(repo, 0)
        print(f"  {format_repository_line(repo, count, wins)}")

    ranked_by_win_rate = win_rate_table(
        stats.repo_comparisons,
        stats.repo_wins,
        min_comparisons=min_comparisons_for_win_rate,
    )
    if ranked_by_win_rate:
        print(
            f"\nTop repositories by win rate "
            f"(min {min_comparisons_for_win_rate} comparisons):"
        )
        for repo, total, wins, rate in sorted(
            ranked_by_win_rate, key=lambda entry: (-entry[3], -entry[1], entry[0])
        )[:top_n]:
            print(f"  {format_repository_line(repo, total, wins, rate)}")

        print(
            f"\nLowest repositories by win rate "
            f"(min {min_comparisons_for_win_rate} comparisons):"
        )
        for repo, total, wins, rate in sorted(
            ranked_by_win_rate, key=lambda entry: (entry[3], entry[1], entry[0])
        )[:top_n]:
            print(f"  {format_repository_line(repo, total, wins, rate)}")
    else:
        print(
            f"\nNo repositories with at least "
            f"{min_comparisons_for_win_rate} comparisons to show win-rate rankings."
        )

    print("\nComparisons by juror:")
    for juror, count in stats.juror_comparisons.most_common():
        print(f"  {juror}: {count}")


def main() -> None:
    stats = collect_stats(RUNS_DIR)
    print_stats(stats)


if __name__ == "__main__":
    main()
