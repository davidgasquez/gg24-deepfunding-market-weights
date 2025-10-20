from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Sequence

import arbitron
from arbitron.models import Agent, Comparison, Item

DATA_DIR = Path("data")
CANDIDATES_PATH = DATA_DIR / "candidate_repositories.csv"
CONTEXT_PATH = DATA_DIR / "repository_context.json"
COMPETITIONS_DIR = DATA_DIR / "competitions"


def load_candidate_urls(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        urls = [row["repository"].strip() for row in reader if row.get("repository")]
    if not urls:
        raise ValueError(f"No repositories found in {path}")
    return urls


def load_context_lookup(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Repository context must be a JSON list.")
    lookup: dict[str, dict[str, Any]] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError("Each context entry must be an object.")
        full_name = entry.get("full_name")
        html_url = entry.get("html_url")
        if not full_name or not html_url:
            raise ValueError("Context entries must include 'full_name' and 'html_url'.")
        lookup[full_name] = entry
        lookup[html_url] = entry
    return lookup


def build_item_description(context: dict[str, Any]) -> str:
    description = context.get("description")
    if isinstance(description, str):
        summary = " ".join(description.split())
    else:
        summary = ""
    if not summary:
        summary = "No GitHub description provided."
    repo_name = context.get("full_name")
    if not isinstance(repo_name, str) or not repo_name:
        repo_name = context.get("html_url", "unknown repository")
    json_blob = json.dumps(context, sort_keys=True)
    safe_blob = escape(json_blob)
    return (
        f"# Repository: {repo_name}\n\n"
        f"> {summary}\n\n"
        "<context>\n"
        f"  <json>{safe_blob}</json>\n"
        "</context>"
    )


def build_items(
    urls: Sequence[str],
    contexts: dict[str, dict[str, Any]],
) -> list[Item]:
    items: list[Item] = []
    missing: set[str] = set()
    for url in urls:
        slug = url.removeprefix("https://github.com/")
        context = contexts.get(url) or contexts.get(slug)
        if context is None:
            missing.add(slug)
            context = {
                "full_name": slug,
                "html_url": url,
                "description": None,
                "context_available": False,
            }
        repo_id = context.get("full_name")
        if not isinstance(repo_id, str) or not repo_id:
            repo_id = slug
        items.append(
            Item(
                id=repo_id,
                description=build_item_description(context),
            )
        )
    if missing:
        names = ", ".join(sorted(missing))
        print(f"Warning: no stored context for {names}; using placeholders.")
    return items


def run_competition(items: Sequence[Item]) -> list[Comparison]:
    comparisons = arbitron.run(
        description="Which dependency has been more valuable to the success of Ethereum?",
        agents=[
            Agent(
                id="Ethereum Expert",
                prompt=(
                    "You are an Ethereum ecosystem expert deciding which repository matters more "
                    "to Ethereum's success. Focus on protocol impact, tooling, adoption, security, "
                    "and long-term sustainability. Use the provided context plus your reliable prior knowledge."
                ),
                model="openai:gpt-5-nano",
            )
        ],
        items=list(items),
        include_reasoning=False,
        verbose=True,
    )
    print(f"Received {len(comparisons)} comparisons for {len(items)} items.")
    return comparisons


def write_comparisons_csv(comparisons: Sequence[Comparison]) -> Path:
    COMPETITIONS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    unique_id = f"{timestamp}-{uuid.uuid4().hex[:8]}"
    target_path = COMPETITIONS_DIR / f"{unique_id}.csv"
    tmp_path = target_path.with_suffix(".csv.tmp")
    fieldnames = ["agent_id", "item_a", "item_b", "winner", "created_at"]
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for comparison in comparisons:
            writer.writerow({
                "agent_id": comparison.agent_id,
                "item_a": comparison.item_a,
                "item_b": comparison.item_b,
                "winner": comparison.winner,
                "created_at": comparison.created_at.isoformat(),
            })
    tmp_path.replace(target_path)
    return target_path


def main() -> None:
    candidate_urls = load_candidate_urls(CANDIDATES_PATH)
    context_lookup = load_context_lookup(CONTEXT_PATH)
    items = build_items(candidate_urls, context_lookup)
    comparisons = run_competition(items)
    output_path = write_comparisons_csv(comparisons)
    print(f"Competition results stored at {output_path}")


if __name__ == "__main__":
    main()
