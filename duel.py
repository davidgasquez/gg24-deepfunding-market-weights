import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from arbitron import Competition, Item, Juror
from arbitron.pairing import RandomPairsSampler
from pydantic import BaseModel
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent


class RepositoryContext(BaseModel):
    organization: str
    name: str
    description: str | None = None
    stars: int | None = None
    github_url: str | None = None

    def to_item(self) -> Item:
        item_id = f"{self.organization}/{self.name}"
        return Item(id=item_id, payload=self.model_dump(mode="json"))


def load_repository_items() -> list[Item]:
    context_path = BASE_DIR / "data" / "phase_2" / "repository_context.json"
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


def main() -> None:
    items = load_repository_items()
    jurors = create_jurors()
    competition = Competition(
        id="gg24",
        description=(
            "Which dependency has been more valuable to the success of Ethereum?"
        ),
        jurors=jurors,
        items=items,
        concurrency=100,
        pair_sampler=RandomPairsSampler(count=5000),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    output_dir = BASE_DIR / "data" / "phase_2" / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{timestamp}.csv"
    stream_to_csv(competition, output_path)


if __name__ == "__main__":
    main()
