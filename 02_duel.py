import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from arbitron import Competition, Item, Juror
from arbitron.pairing import PairSampler, RandomPairsSampler
from pydantic import BaseModel
from tqdm import tqdm


class RepositoryContext(BaseModel):
    organization: str
    name: str
    description: str | None = None
    stars: int | None = None
    github_url: str | None = None

    def to_item(self) -> Item:
        item_id = f"{self.organization}/{self.name}"
        return Item(id=item_id, payload=self.model_dump(mode="json"))


def load_repository_items(context_path: Path) -> list[Item]:
    with context_path.open("r", encoding="utf-8") as json_file:
        context_data = json.load(json_file)

    repositories = [RepositoryContext(**entry) for entry in context_data]
    return [repository.to_item() for repository in repositories]


def create_jurors() -> list[Juror]:
    return [
        Juror(
            id="dev",
            instructions=(
                "You are a seasoned developer in the Ethereum ecosystem specialized in tooling and infrastructure."
                "Choose the dependency that has been more valuable to the success of Ethereum."
                "Consider your knowledge and experience with the given dependencies."
                "Provided an unbiased answer based on their actual impact and value provided."
            ),
            model="openai:gpt-5-mini",
        ),
        Juror(
            id="senior-dev",
            instructions=(
                "You are someone deep in the Ethereum ecosystem. You specialize in tooling and infrastructure."
                "Analyze which dependency has been more valuable to the success of Ethereum."
                "Consider only your knowledge and experience with the dependency/repository."
                "Provided an unbiased answer based on their actual impact/value."
            ),
            model="openai:gpt-5-mini",
        ),
        Juror(
            id="ethereum-dev",
            instructions=(
                "You've been developing tools and infrastructure in the Ethereum ecosystem for many years."
                "Share which dependency has been more valuable to the success of Ethereum and yourself."
                "Consider your knowledge and experience with the dependency/repository."
                "Provided an unbiased answer based on their actual impact/value."
            ),
            model="openai:gpt-5-mini",
        ),
        Juror(
            id="founder",
            instructions=(
                "You are a founder or builder in the Ethereum ecosystem who relies on open-source dependencies (tooling and infrastructure) to deliver products."
                "Determine which dependency has been more valuable to the success of Ethereum."
                "Consider your knowledge and experience with the dependency/repository."
            ),
            model="openai:gpt-5-mini",
        ),
        Juror(
            id="builder",
            instructions=(
                "You are a builder in the Ethereum ecosystem who uses diverse tooling and infrastructure to deliver applications."
                "Decide which dependency has been more valuable to the success of Ethereum."
                "Consider your knowledge and experience with the dependency/repository."
            ),
            model="openai:gpt-5-mini",
        ),
        Juror(
            id="meta",
            instructions=(
                "You are a group of experienced members of the Ethereum ecosystem familiar with tooling and infrastructure."
                "Evaluate which dependency has been more valuable to the success of Ethereum."
                "Consider your knowledge and experience with the dependency/repository."
            ),
            model="openai:gpt-5-mini",
        ),
    ]


def stream_to_csv(competition: Competition, csv_path: Path) -> None:
    fieldnames = [
        "competition_id",
        "juror_id",
        "item_a",
        "item_b",
        "winner",
        "comparison_created_at",
        "comparison_cost",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        with tqdm(
            total=competition.total_comparisons,
            unit="comparison",
        ) as progress:
            for comparison in competition.run():
                writer.writerow({
                    "competition_id": competition.id,
                    "juror_id": comparison.juror_id,
                    "item_a": comparison.item_a,
                    "item_b": comparison.item_b,
                    "winner": comparison.winner,
                    "comparison_created_at": comparison.created_at.isoformat(),
                    "comparison_cost": (
                        str(comparison.cost) if comparison.cost is not None else ""
                    ),
                })
                csvfile.flush()
                progress.update(1)

    print(f"Total cost: {competition.cost}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a head-to-head competition between repositories."
    )
    parser.add_argument(
        "context_path",
        type=Path,
        help="Path to the repository context JSON file.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where run CSV files will be stored.",
    )
    parser.add_argument(
        "--focus-id",
        type=str,
        default=None,
        help="Repository id (organization/name) to include in every comparison.",
    )
    parser.add_argument(
        "--pair-count",
        type=int,
        default=2000,
        help="Number of pairwise comparisons to generate.",
    )
    args = parser.parse_args()
    args.context_path = args.context_path.expanduser()
    if not args.context_path.exists():
        parser.error(f"Context file not found: {args.context_path}")
    args.output_dir = args.output_dir.expanduser()
    if args.output_dir.exists() and not args.output_dir.is_dir():
        parser.error("Output path must be a directory.")
    if args.pair_count < 1:
        parser.error("--pair-count must be at least 1.")
    return args


class FocusedPairsSampler(PairSampler):
    """Generate comparisons where a single repository faces random opponents."""

    def __init__(
        self,
        focus_id: str,
        count: int,
        seed: int | None = None,
    ) -> None:
        if count < 1:
            raise ValueError("count must be at least 1")
        self._focus_id = focus_id
        self._count = count
        self._seed = seed

    def sample(self, items: Sequence[Item]) -> list[tuple[Item, Item]]:
        id_to_item = {item.id: item for item in items}
        try:
            focus_item = id_to_item[self._focus_id]
        except KeyError as exc:
            raise ValueError(f"focus item '{self._focus_id}' not found in items") from exc

        opponents = [item for item in items if item.id != self._focus_id]
        if not opponents:
            raise ValueError("focus item must have at least one opponent")

        rng = random.Random(self._seed)
        pairs: list[tuple[Item, Item]] = []
        for _ in range(self._count):
            opponent = rng.choice(opponents)
            if rng.random() < 0.5:
                pairs.append((focus_item, opponent))
            else:
                pairs.append((opponent, focus_item))
        return pairs


def main() -> None:
    args = parse_args()
    items = load_repository_items(args.context_path)
    jurors = create_jurors()
    if args.focus_id is not None:
        item_ids = {item.id for item in items}
        if args.focus_id not in item_ids:
            sample_ids = ", ".join(sorted(item_ids)[:5])
            message = f"--focus-id '{args.focus_id}' not found in context."
            if sample_ids:
                message += f" Example ids: {sample_ids}"
            raise SystemExit(message)
        pair_sampler: PairSampler = FocusedPairsSampler(
            focus_id=args.focus_id,
            count=args.pair_count,
        )
    else:
        pair_sampler = RandomPairsSampler(count=args.pair_count)
    competition = Competition(
        id="gg24",
        description=(
            "Which dependency has been more valuable to the success of Ethereum?"
        ),
        jurors=jurors,
        items=items,
        concurrency=100,
        pair_sampler=pair_sampler,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    output_path = output_dir / f"{timestamp}.csv"
    stream_to_csv(competition, output_path)


if __name__ == "__main__":
    main()
