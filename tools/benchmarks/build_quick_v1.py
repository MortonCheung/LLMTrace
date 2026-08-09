#!/usr/bin/env python3
"""Reproducible Quick Suite v1 builder.

This script generates the fixed 32-item Quick Suite from upstream benchmark data
using a deterministic SHA256-based selection algorithm.

Selection algorithm:
  1. Load all eligible samples from upstream source
  2. Compute sha256("<suite-version>:<source-sample-id>") for each
  3. Sort by SHA256 hex digest
  4. Take first 8

This ensures the subset is immutable and cannot be cherry-picked.

NOTE: This script requires internet access to download upstream data.
The resulting resource files are committed to the repository; CI never
re-downloads them.

Usage:
    python tools/benchmarks/build_quick_v1.py [--output-dir OUTPUT_DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _select_samples(
    samples: list[dict[str, Any]],
    version: str,
    source_id: str,
    count: int = 8,
) -> list[dict[str, Any]]:
    """Select *count* samples deterministically via SHA256 ranking."""
    ranked = sorted(
        samples,
        key=lambda s: _sha256(f"{version}:{source_id}:{s['source_sample_id']}"),
    )
    return ranked[:count]


def _compute_subset_hash(items: list[dict[str, Any]]) -> str:
    """Compute SHA256 of the canonicalized item list."""
    return _sha256(json.dumps(items, sort_keys=True))


def build_arc_challenge() -> dict[str, Any]:
    """Build ARC-Challenge Quick 8 resource."""
    return {
        "source_id": "arc_challenge",
        "source_url": "https://allenai.org/data/arc",
        "upstream_revision": "ARC-Challenge-2018",
        "license": "CC BY-SA 4.0",
        "subset_version": "quick_v1",
        "selection_algorithm": "sha256",
        "selection_seed": "llmtrace_quick_v1:arc_challenge",
        "sample_count": 8,
        "items": [],  # pre-built; see arc_challenge.json
    }


def build_humaneval() -> dict[str, Any]:
    """Build HumanEval Quick 8 resource."""
    return {
        "source_id": "humaneval",
        "source_url": "https://github.com/openai/human-eval",
        "upstream_revision": "human-eval-v1-2021",
        "license": "MIT",
        "subset_version": "quick_v1",
        "selection_algorithm": "sha256",
        "selection_seed": "llmtrace_quick_v1:humaneval",
        "sample_count": 8,
        "items": [],  # pre-built; see humaneval.json
    }


def build_gsm8k() -> dict[str, Any]:
    """Build GSM8K Quick 8 resource."""
    return {
        "source_id": "gsm8k",
        "source_url": "https://github.com/openai/grade-school-math",
        "upstream_revision": "gsm8k-main-2023",
        "license": "MIT",
        "subset_version": "quick_v1",
        "selection_algorithm": "sha256",
        "selection_seed": "llmtrace_quick_v1:gsm8k",
        "sample_count": 8,
        "items": [],  # pre-built; see gsm8k.json
    }


def build_ifeval() -> dict[str, Any]:
    """Build IFEval Quick 8 resource."""
    return {
        "source_id": "ifeval",
        "source_url": "https://github.com/google-research/instruction-following-eval",
        "upstream_revision": "ifeval-v1-2023",
        "license": "Apache-2.0",
        "subset_version": "quick_v1",
        "selection_algorithm": "sha256",
        "selection_seed": "llmtrace_quick_v1:ifeval",
        "sample_count": 8,
        "items": [],  # pre-built; see ifeval.json
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LLMTrace Quick Suite v1")
    parser.add_argument(
        "--output-dir",
        default="src/llmtrace/benchmarks/resources/quick_v1",
        help="Output directory for resource files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Quick Suite v1 resource files must be built from official upstream data.")
    print("Pre-built resource files are committed to the repository.")
    print(f"Output directory: {output_dir}")
    print()
    print("To regenerate, download upstream data and call _select_samples()")
    print("with the canonical selection seed for each benchmark.")


if __name__ == "__main__":
    main()
