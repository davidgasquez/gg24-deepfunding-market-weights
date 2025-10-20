import argparse
import asyncio
import json
import uuid
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from random import shuffle
from typing import Any, Literal, cast

from pydantic import BaseModel
from pydantic_ai import Agent
from tqdm import tqdm

REPOSITORY_DATA_PATH = Path("data/repository_context.json")

INSTRUCTIONS = """
You are an Ethereum ecosystem expert comparing two repositories.
Prioritize protocol impact, tooling support, adoption, security posture, and
long-term sustainability.

<output_rule>
    Return only one option. Either `item_a` or `item_b`.
</output_rule>
"""

COMPARISON_TEMPLATE = """
Which dependency has been more valuable to the success of Ethereum?
<choices>
    <choice id="item_a">
    <full_name>{item_a_full_name}</full_name>
    <url>{item_a_url}</url>
    <description>{item_a_description}</description>
    <stars>{item_a_stars}</stars>
    <contributors>{item_a_contributors}</contributors>
    </choice>
    <choice id="item_b">
    <full_name>{item_b_full_name}</full_name>
    <url>{item_b_url}</url>
    <description>{item_b_description}</description>
    <stars>{item_b_stars}</stars>
    <contributors>{item_b_contributors}</contributors>
    </choice>
</choices>

<response_format>
    Reply only with `item_a` or `item_b`.
</response_format>
""".strip()


class ComparisonDecision(BaseModel):
    winner: Literal["item_a", "item_b"]


comparator = Agent(
    "openai:gpt-5-nano",
    instructions=INSTRUCTIONS,
    output_type=ComparisonDecision,
)


Repository = dict[str, Any]


def load_repositories(path: Path) -> list[Repository]:
    with path.open("r", encoding="utf-8") as repo_file:
        raw_entries = json.load(repo_file)

    if not isinstance(raw_entries, list):
        msg = f"Repository data must be a list, got {type(raw_entries).__name__}"
        raise ValueError(msg)

    repositories: list[Repository] = []
    seen_full_names: set[str] = set()
    for entry in raw_entries:
        if not isinstance(entry, dict):
            raise ValueError("Each repository entry must be a JSON object")

        full_name = entry.get("full_name")
        if not full_name:
            raise ValueError("Each repository entry must include a `full_name` field")

        key = str(full_name)
        if key in seen_full_names:
            continue

        seen_full_names.add(key)
        repositories.append(entry)

    return repositories


def field_text(item: Repository, key: str, fallback: str) -> str:
    value = item.get(key)
    return fallback if value in (None, "") else str(value)


def build_prompt(item_a: Repository, item_b: Repository) -> str:
    return COMPARISON_TEMPLATE.format(
        item_a_full_name=field_text(item_a, "full_name", "Unknown"),
        item_a_url=field_text(item_a, "html_url", "Unknown"),
        item_a_description=field_text(item_a, "description", "No description provided"),
        item_a_stars=field_text(item_a, "stars", "0"),
        item_a_contributors=field_text(item_a, "contributors", "0"),
        item_b_full_name=field_text(item_b, "full_name", "Unknown"),
        item_b_url=field_text(item_b, "html_url", "Unknown"),
        item_b_description=field_text(item_b, "description", "No description provided"),
        item_b_stars=field_text(item_b, "stars", "0"),
        item_b_contributors=field_text(item_b, "contributors", "0"),
    )


DEFAULT_LIMIT = 50
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_CONCURRENCY = 10
DEFAULT_FLUSH_INTERVAL = DEFAULT_CONCURRENCY


def build_output_path(output_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{timestamp}.ndjson"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise repository comparison.")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of comparisons to execute.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write NDJSON comparison results.",
    )
    parser.add_argument(
        "--agent-id",
        type=str,
        default=None,
        help="Identifier stored with each result (random if omitted).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Maximum number of concurrent agent requests.",
    )
    return parser.parse_args()


async def evaluate_pair(
    semaphore: asyncio.Semaphore,
    agent_id: str,
    item_a: Repository,
    item_b: Repository,
) -> dict[str, Any]:
    async with semaphore:
        prompt = build_prompt(item_a, item_b)
        result = await comparator.run(prompt)
        decision = cast(ComparisonDecision, result.output)
        return {
            "agent_id": agent_id,
            "item_a": field_text(item_a, "full_name", "Unknown"),
            "item_b": field_text(item_b, "full_name", "Unknown"),
            "winner": decision.winner,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


async def process_comparisons(
    pairs: list[tuple[Repository, Repository]],
    agent_id: str,
    output_path: Path,
    concurrency: int,
    flush_interval: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(evaluate_pair(semaphore, agent_id, item_a, item_b))
        for item_a, item_b in pairs
    ]

    results: list[dict[str, Any]] = []
    with (
        output_path.open("w", encoding="utf-8") as output_file,
        tqdm(total=len(tasks), desc="Comparisons") as progress,
    ):
        for task in asyncio.as_completed(tasks):
            record = await task
            output_file.write(json.dumps(record, ensure_ascii=False))
            output_file.write("\n")
            results.append(record)
            if flush_interval > 0 and len(results) % flush_interval == 0:
                output_file.flush()
            progress.update(1)

    return results


def main() -> None:
    args = parse_arguments()
    if args.limit <= 0:
        raise ValueError("`--limit` must be greater than zero.")
    if args.concurrency <= 0:
        raise ValueError("`--concurrency` must be greater than zero.")

    output_path = build_output_path(args.output_dir)

    repositories = load_repositories(REPOSITORY_DATA_PATH)
    if len(repositories) < 2:
        msg = f"Need at least two repositories to compare, found {len(repositories)}"
        raise ValueError(msg)

    pairs = list(combinations(repositories, 2))
    if not pairs:
        raise ValueError("No repository pairs available for comparison.")

    shuffle(pairs)

    limit = min(args.limit, len(pairs))
    selected_pairs = pairs[:limit]
    agent_id = args.agent_id or uuid.uuid4().hex

    asyncio.run(
        process_comparisons(
            selected_pairs,
            agent_id,
            output_path,
            args.concurrency,
            DEFAULT_FLUSH_INTERVAL,
        )
    )

    print(f"Completed {len(selected_pairs)} comparisons with agent {agent_id}.")
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
