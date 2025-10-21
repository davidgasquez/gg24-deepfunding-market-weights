"""Generate repository metadata from a CSV list of GitHub repositories."""

import argparse
import asyncio
import csv
import json
import os
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

API_BASE = "https://api.github.com"
CONCURRENCY_LIMIT = 8
TIMEOUT = httpx.Timeout(connect=5.0, read=15.0, write=10.0, pool=5.0)


def parse_repo_slug(raw: str) -> str:
    """Normalise repository entries to the ``owner/name`` form."""

    value = raw.strip()
    if not value:
        raise ValueError("Repository entry is empty.")

    if value.startswith(("http://", "https://")):
        path = urlparse(value).path
    else:
        path = value

    parts = [segment for segment in path.split("/") if segment]
    if len(parts) < 2:
        raise ValueError(f"Unable to parse repository identifier from '{raw}'.")

    owner, name = parts[:2]
    return f"{owner}/{name}"


def clean_text(value: str) -> str | None:
    """Normalise unicode whitespace/punctuation and collapse runs."""

    normalised = unicodedata.normalize("NFKC", value)
    pieces: list[str] = []
    for char in normalised:
        category = unicodedata.category(char)
        if category.startswith("C"):
            pieces.append(" ")
        elif category == "Zs":
            pieces.append(" ")
        elif category == "Pd" and char != "-":
            pieces.append("-")
        else:
            pieces.append(char)

    collapsed = " ".join("".join(pieces).split())
    return collapsed or None


def read_repository_slugs(csv_path: Path) -> list[str]:
    """Read distinct repository slugs from ``csv_path`` preserving order."""

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "repository" not in reader.fieldnames:
            raise ValueError("CSV file must contain a 'repository' column.")

        slugs = [
            parse_repo_slug(row["repository"])
            for row in reader
            if row.get("repository")
        ]

    unique_slugs = list(dict.fromkeys(slugs))
    if not unique_slugs:
        raise ValueError("No repositories found in CSV file.")

    return unique_slugs


async def fetch_repository(client: httpx.AsyncClient, slug: str) -> dict[str, Any]:
    """Retrieve minimal metadata for a single repository."""

    response = await client.get(f"/repos/{slug}")
    response.raise_for_status()
    repo = response.json()
    description = repo.get("description")
    if isinstance(description, str):
        description = clean_text(description)
    else:
        description = None
    return {
        "name": repo["name"],
        "description": description,
        "stars": repo.get("stargazers_count", 0),
        "organization": repo["owner"]["login"],
        "github_url": repo["html_url"],
    }


async def fetch_repositories(token: str, slugs: Sequence[str]) -> list[dict[str, Any]]:
    """Fetch metadata for each slug, preserving input order."""

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "gg24-deepfunding-market-weights/0.1.0",
        "Authorization": f"Bearer {token}",
    }

    async with httpx.AsyncClient(
        base_url=API_BASE,
        headers=headers,
        timeout=TIMEOUT,
        follow_redirects=True,
        limits=httpx.Limits(
            max_connections=CONCURRENCY_LIMIT * 2,
            max_keepalive_connections=CONCURRENCY_LIMIT * 2,
        ),
    ) as client:
        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
        results: list[dict[str, Any] | None] = [None] * len(slugs)

        async def fetch_into(index: int, slug: str) -> None:
            async with semaphore:
                results[index] = await fetch_repository(client, slug)

        async with asyncio.TaskGroup() as group:
            for index, slug in enumerate(slugs):
                group.create_task(fetch_into(index, slug))

    return [record for record in results if record is not None]


def write_output(data: Sequence[dict[str, Any]], destination: Path | None) -> None:
    """Write JSON output to ``destination`` or stdout when ``None``."""

    payload = json.dumps(list(data), indent=2, ensure_ascii=False) + "\n"
    if destination is None:
        print(payload, end="")
        return
    destination.write_text(payload, encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Fetch GitHub repository metadata from a CSV file.",
    )
    parser.add_argument(
        "csv",
        type=Path,
        help="CSV file containing a 'repository' column.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Destination file for JSON output (stdout when omitted).",
    )
    return parser.parse_args(argv)


async def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = parse_args(argv)
    token = os.environ["GITHUB_TOKEN"]
    slugs = read_repository_slugs(args.csv)
    records = await fetch_repositories(token, slugs)
    write_output(records, args.out)
    if args.out:
        print(f"Wrote {len(records)} repositories to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
