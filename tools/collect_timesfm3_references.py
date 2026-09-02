# /// script
# requires-python = ">=3.10"
# dependencies = ["crawl4ai==0.9.3"]
# ///
"""Collect the dynamic, fixed-source portion of the TimesFM-3 OKF bundle."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.async_crawler_strategy import AsyncHTTPCrawlerStrategy, HTTPCrawlerConfig

MAX_CONTENT_BYTES = 2 * 1024 * 1024
ALLOWED_HOSTS = frozenset({"research.google", "huggingface.co", "docs.cloud.google.com"})
SOURCES = (
    (
        "google-timesfm-3-blog",
        "Google Research TimesFM-3 launch post",
        "https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/",
    ),
    (
        "huggingface-timesfm-3-model-card",
        "Official TimesFM 3.0 PyTorch model card",
        "https://huggingface.co/google/timesfm-3.0-pytorch",
    ),
    (
        "bigquery-ai-forecast",
        "BigQuery ML AI.FORECAST reference",
        "https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-ai-forecast",
    ),
    (
        "fev-bench-leaderboard",
        "FEV-Bench leaderboard",
        "https://huggingface.co/spaces/autogluon/fev-bench",
    ),
    (
        "gift-eval-leaderboard",
        "GIFT-Eval leaderboard",
        "https://huggingface.co/spaces/Salesforce/GIFT-Eval",
    ),
    (
        "time-leaderboard",
        "TIME leaderboard",
        "https://huggingface.co/spaces/Real-TSF/TIME-leaderboard",
    ),
)


def validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"URL is outside the fixed public allowlist: {url}")


def render_reference(title: str, url: str, markdown: str) -> str:
    collected_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return f"""---
type: Reference
title: {title}
description: Cleaned source extract collected for the TimesFM-3 research bundle.
resource: {url}
tags: [timesfm, timesfm-3, source]
status: draft
generated:
  by: crawl4ai/0.9.3
  at: {collected_at}
sources:
  - id: canonical-source
    resource: {url}
    title: {title}
---

# Source extract

Canonical source: <{url}>.

## Extracted content

{markdown.rstrip()}
"""


async def collect(output_dir: Path, dry_run: bool) -> int:
    for _, _, url in SOURCES:
        validate_url(url)
    if dry_run:
        for slug, _, url in SOURCES:
            print(f"{url} -> {output_dir / f'{slug}.md'}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        check_robots_txt=True,
        excluded_tags=["nav", "footer", "aside", "script", "style"],
        word_count_threshold=3,
    )
    failures: list[str] = []
    http_strategy = AsyncHTTPCrawlerStrategy(
        browser_config=HTTPCrawlerConfig(verify_ssl=True)
    )
    async with AsyncWebCrawler(crawler_strategy=http_strategy) as crawler:
        for slug, title, url in SOURCES:
            result = await crawler.arun(url=url, config=config)
            final_host = urlparse(result.url).hostname if result.url else None
            if not result.success:
                failures.append(f"{url}: {result.error_message}")
                continue
            if final_host not in ALLOWED_HOSTS:
                failures.append(f"{url}: redirected outside the allowlist to {result.url}")
                continue
            markdown = result.markdown.raw_markdown if result.markdown else ""
            if not markdown.strip():
                failures.append(f"{url}: no Markdown extracted")
                continue
            if len(markdown.encode("utf-8")) > MAX_CONTENT_BYTES:
                failures.append(f"{url}: extracted content exceeds {MAX_CONTENT_BYTES} bytes")
                continue
            (output_dir / f"{slug}.md").write_text(
                render_reference(title, url, markdown), encoding="utf-8", newline="\n"
            )
            print(f"collected: {url}")
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("knowledge/references"))
    args = parser.parse_args()
    output_dir = args.output.resolve()
    expected_root = (Path.cwd() / "knowledge" / "references").resolve()
    if output_dir != expected_root:
        raise ValueError(f"Output must be the fixed bundle directory: {expected_root}")
    return asyncio.run(collect(output_dir, args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
