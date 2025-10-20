from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx


RETRYABLE_STATUS = {502, 503, 504}
MAX_RETRIES = 3
REQUEST_TIMEOUT = httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=5.0)
CONCURRENCY_LIMIT = 6


class RepositoryListError(RuntimeError):
    pass


def parse_repo_slug(raw: str) -> str:
    value = raw.strip()
    if not value:
        raise RepositoryListError("Repository entry is empty.")

    if value.startswith(("http://", "https://")):
        parsed = urlparse(value)
        path = parsed.path.strip("/")
    else:
        path = value

    parts = [segment for segment in path.split("/") if segment]
    if len(parts) < 2:
        raise RepositoryListError(f"Unable to parse repository identifier from '{raw}'.")

    owner, name = parts[:2]
    return f"{owner}/{name}"


def load_repository_slugs(csv_path: Path) -> list[str]:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "repository" not in reader.fieldnames:
            raise RepositoryListError("CSV file must contain a 'repository' column.")

        slugs: list[str] = []
        seen: set[str] = set()
        for row in reader:
            raw = row.get("repository", "")
            if not raw:
                continue
            slug = parse_repo_slug(raw)
            if slug not in seen:
                seen.add(slug)
                slugs.append(slug)

    if not slugs:
        raise RepositoryListError("No repositories found in CSV file.")

    return slugs


def build_client(token: str) -> httpx.AsyncClient:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "gg24-deepfunding-market-weights/0.1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.AsyncClient(
        base_url="https://api.github.com",
        headers=headers,
        timeout=REQUEST_TIMEOUT,
        follow_redirects=True,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=20),
    )


async def get_with_retries(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    allow_status: Iterable[int] = (),
) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await client.get(url, params=params)
        except httpx.HTTPError as exc:
            last_exc = exc
        else:
            if response.status_code in allow_status:
                return response
            if response.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES:
                await asyncio.sleep(0.5 * attempt)
                continue
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                last_exc = exc
            else:
                return response
        await asyncio.sleep(0.5 * attempt)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Request failed without exception detail.")


def extract_count(response: httpx.Response) -> int:
    last_link = response.links.get("last")
    if last_link and (url := last_link.get("url")):
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        page_values = query.get("page")
        if page_values:
            try:
                return int(page_values[0])
            except (TypeError, ValueError):
                pass

    try:
        payload = response.json()
    except ValueError:
        return 0
    if isinstance(payload, list):
        return len(payload)
    return 0


async def fetch_repository_summary(client: httpx.AsyncClient, slug: str) -> dict[str, Any]:
    repo_resp = await get_with_retries(client, f"/repos/{slug}")
    repo = repo_resp.json()

    contributors_resp = await get_with_retries(
        client,
        f"/repos/{slug}/contributors",
        params={"per_page": 1, "anon": 1},
    )
    contributors = extract_count(contributors_resp)

    commits_resp = await get_with_retries(
        client,
        f"/repos/{slug}/commits",
        params={"per_page": 1},
        allow_status=(409,),
    )
    if commits_resp.status_code == 409:
        commits = 0
    else:
        commits = extract_count(commits_resp)

    return {
        "full_name": repo["full_name"],
        "name": repo["name"],
        "description": repo.get("description"),
        "stars": repo.get("stargazers_count", 0),
        "forks": repo.get("forks_count", 0),
        "contributors": contributors,
        "commits": commits,
        "html_url": repo.get("html_url"),
    }


async def fetch_with_limit(
    client: httpx.AsyncClient, slug: str, limiter: asyncio.Semaphore
) -> dict[str, Any]:
    print(f"Fetching {slug}...", flush=True)
    async with limiter:
        return await fetch_repository_summary(client, slug)


def write_output(data: list[dict[str, Any]], destination: Path) -> None:
    destination.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate repository context JSON from GitHub metadata.")
    parser.add_argument("--csv", default="repositories.csv", type=Path, help="Path to repositories.csv file.")
    parser.add_argument(
        "--out",
        default="repository_context.json",
        type=Path,
        help="Destination path for generated JSON file.",
    )
    return parser.parse_args(argv)


async def collect_repository_data(token: str, slugs: list[str]) -> list[dict[str, Any]]:
    limiter = asyncio.Semaphore(CONCURRENCY_LIMIT)
    async with build_client(token) as client:
        tasks = [fetch_with_limit(client, slug, limiter) for slug in slugs]
        return await asyncio.gather(*tasks)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN environment variable is required.", file=sys.stderr)
        return 1

    try:
        slugs = load_repository_slugs(args.csv)
    except RepositoryListError as exc:
        print(f"Error reading repositories: {exc}", file=sys.stderr)
        return 1

    try:
        output = asyncio.run(collect_repository_data(token, slugs))
    except Exception as exc:  # noqa: BLE001 - surface the failure cleanly
        print(f"Failed to fetch repository data: {exc}", file=sys.stderr)
        return 1

    write_output(output, args.out)
    print(f"Wrote {len(output)} repository entries to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
