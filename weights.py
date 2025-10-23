#!/usr/bin/env python3
"""
Compute ranks and allocation weights from pairwise comparisons using the Colley method.

Input rows (CSV):
  competition_id,juror_id,item_a,item_b,winner,comparison_created_at,comparison_cost

- Each row is a comparison between item_a and item_b with a single winner (or tie).
- We compute per-competition ratings via Colley:
      C r = b
  where C_ii = 2 + games_i,  C_ij = - (# matches between i and j),
        b_i = 1 + 0.5*(wins_i - losses_i).
- Ratings r are then normalized to weights that sum to 1 within each competition.
- Output columns: competition_id,item,score,weight,rank,wins,losses,ties,games

Usage:
  python weights.py --csv data/runs --out data/weights.csv

Requires:
  numpy (for solving the linear system)
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

REQUIRED_COLUMNS = {
    "competition_id",
    "juror_id",
    "item_a",
    "item_b",
    "winner",
    "comparison_created_at",
    "comparison_cost",
}


def iter_csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    """Yield dict rows from a file or all *.csv files under a directory."""
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.rglob("*.csv"))
    else:
        raise FileNotFoundError(f"No such file or directory: {path}")

    for f in files:
        with f.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            # Basic schema check (non-fatal: just ensure we have the needed fields)
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                print(
                    f"[warn] {f} missing columns {sorted(missing)}; attempting to proceed if core fields are present.",
                    file=sys.stderr,
                )
            for row in reader:
                yield row


def build_counts(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict]:
    """
    Aggregate wins/losses/ties and pair counts per competition.

    Returns a dict: competition_id -> {
        'items': set[str],
        'wins': defaultdict(int),
        'losses': defaultdict(int),
        'ties': defaultdict(int),
        'games': defaultdict(int),
        'pair_totals': defaultdict(int)  # key: (min_item, max_item) -> matches
    }
    """
    by_comp: Dict[str, Dict] = defaultdict(
        lambda: {
            "items": set(),
            "wins": defaultdict(int),
            "losses": defaultdict(int),
            "ties": defaultdict(int),
            "games": defaultdict(int),
            "pair_totals": defaultdict(int),
        }
    )

    for r in rows:
        comp = (r.get("competition_id") or "").strip()
        a = (r.get("item_a") or "").strip()
        b = (r.get("item_b") or "").strip()
        winner = (r.get("winner") or "").strip()

        if not comp or not a or not b:
            # Skip malformed rows quietly, but warn.
            print(
                f"[warn] Skipping row with missing competition_id/item_a/item_b: {r}",
                file=sys.stderr,
            )
            continue

        agg = by_comp[comp]
        agg["items"].update([a, b])
        agg["games"][a] += 1
        agg["games"][b] += 1
        agg["pair_totals"][tuple(sorted((a, b)))] += 1

        if winner == a:
            agg["wins"][a] += 1
            agg["losses"][b] += 1
        elif winner == b:
            agg["wins"][b] += 1
            agg["losses"][a] += 1
        else:
            # Treat anything else (empty/malformed) as a tie.
            agg["ties"][a] += 1
            agg["ties"][b] += 1

    return by_comp


def colley_ratings(
    items: List[str],
    wins: Dict[str, int],
    losses: Dict[str, int],
    ties: Dict[str, int],
    games: Dict[str, int],
    pair_totals: Dict[Tuple[str, str], int],
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Solve the Colley system for the given competition; return ratings and index map.
    """
    n = len(items)
    idx = {item: i for i, item in enumerate(items)}
    C = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)

    for i, it in enumerate(items):
        total = games.get(it, 0)
        w = wins.get(it, 0)
        l = losses.get(it, 0)
        C[i, i] = 2.0 + total
        b[i] = 1.0 + 0.5 * (w - l)

    for (x, y), c in pair_totals.items():
        i = idx[x]
        j = idx[y]
        C[i, j] -= c
        C[j, i] -= c

    # Solve C r = b
    r = np.linalg.solve(C, b)

    # Numerical guard: clip to non-negative
    r = np.clip(r, 0.0, None)
    return r, idx


def dense_ranks(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Dense ranks (1,2,3,...) where ties get the same rank and the next distinct value
    gets the next integer. Highest value -> rank 1.
    """
    order = np.argsort(-values)
    ranks = np.empty(len(values), dtype=int)
    rank = 1
    ranks[order[0]] = rank
    for k in range(1, len(values)):
        if abs(values[order[k]] - values[order[k - 1]]) > eps:
            rank += 1
        ranks[order[k]] = rank
    return ranks


def compute_weights(input_path: Path) -> List[Dict[str, str]]:
    """
    Full pipeline: read rows, aggregate per competition, solve ratings, normalize to weights.
    Returns a list of dict rows ready to be written to CSV.
    """
    rows = iter_csv_rows(input_path)
    by_comp = build_counts(rows)

    output_rows: List[Dict[str, str]] = []

    for comp, agg in sorted(by_comp.items()):
        items = sorted(agg["items"])
        if not items:
            continue

        r, idx = colley_ratings(
            items,
            agg["wins"],
            agg["losses"],
            agg["ties"],
            agg["games"],
            agg["pair_totals"],
        )
        total = float(r.sum())
        if total <= 0:
            weights = np.full_like(r, 1.0 / len(r))
        else:
            weights = r / total

        ranks = dense_ranks(weights)

        # Emit one row per item
        for it in items:
            i = idx[it]
            w = agg["wins"].get(it, 0)
            l = agg["losses"].get(it, 0)
            t = agg["ties"].get(it, 0)
            g = agg["games"].get(it, 0)
            output_rows.append({
                "competition_id": comp,
                "item": it,
                "score": f"{r[i]:.12f}",  # raw Colley rating
                "weight": f"{weights[i]:.12f}",  # normalized to sum=1 per competition
                "rank": str(int(ranks[i])),
                "wins": str(int(w)),
                "losses": str(int(l)),
                "ties": str(int(t)),
                "games": str(int(g)),
            })

    # Sort output: competition_id asc, weight desc, item asc
    output_rows.sort(
        key=lambda d: (d["competition_id"], -float(d["weight"]), d["item"])
    )
    return output_rows


def write_csv(rows: List[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "competition_id",
        "item",
        "score",
        "weight",
        "rank",
        "wins",
        "losses",
        "ties",
        "games",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute ranks and allocation weights from pairwise CSVs (Colley method)."
    )
    p.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="Path to a CSV file or a directory containing CSV files.",
    )
    p.add_argument(
        "--out", required=True, type=Path, help="Path to write the output weights CSV."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        rows = compute_weights(args.csv)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(2)
    write_csv(rows, args.out)
    # Small summary to stderr
    comps = len({r["competition_id"] for r in rows})
    items = len(rows)
    print(
        f"[ok] wrote {items} item rows across {comps} competition(s) -> {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
