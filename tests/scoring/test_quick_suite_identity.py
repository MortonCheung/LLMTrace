"""Suite Content Identity tests (§36): deterministic, canonical, fail-closed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import llmtrace.adapters.quick_suite as quick_suite
from llmtrace.adapters.base import BenchmarkAdapterError
from llmtrace.adapters.quick_suite import (
    get_quick_suite_content_sha256,
    get_quick_suite_source_revisions,
)

_MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "llmtrace"
    / "benchmarks"
    / "resources"
    / "quick_v1"
    / "manifest.json"
)


def _load_real_manifest() -> dict[str, object]:
    with open(_MANIFEST_PATH) as f:
        return json.load(f)


class TestContentSha256:
    def test_returns_64_lowercase_hex(self) -> None:
        digest = get_quick_suite_content_sha256()
        assert len(digest) == 64
        assert digest == digest.lower()
        int(digest, 16)  # valid hex

    def test_deterministic_across_calls(self) -> None:
        assert get_quick_suite_content_sha256() == get_quick_suite_content_sha256()

    def test_not_raw_manifest_file_bytes(self) -> None:
        raw = _MANIFEST_PATH.read_bytes()
        assert get_quick_suite_content_sha256() != hashlib.sha256(raw).hexdigest()

    def test_whitespace_and_key_order_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        baseline = get_quick_suite_content_sha256()
        manifest = _load_real_manifest()
        # Reorder top-level keys and every task's keys — a cosmetic change.
        reordered: dict[str, object] = {
            "selection_seed_format": manifest["selection_seed_format"],
            "selection_algorithm": manifest["selection_algorithm"],
            "total_items": manifest["total_items"],
            "description": manifest["description"],
            "suite_version": manifest["suite_version"],
            "suite_id": manifest["suite_id"],
            "created_at": manifest["created_at"],
            "tasks": {tid: dict(reversed(list(meta.items()))) for tid, meta in manifest["tasks"].items()},
        }
        # Re-serialize with different whitespace — semantic identity must hold.
        dumped = json.dumps(reordered, indent=4)
        reloaded = json.loads(dumped)
        monkeypatch.setattr(quick_suite, "_load_quick_suite_manifest", lambda: reloaded)
        assert get_quick_suite_content_sha256() == baseline

    def test_sample_count_change_changes_hash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        baseline = get_quick_suite_content_sha256()
        manifest = _load_real_manifest()
        manifest["tasks"]["arc_challenge_quick_v1"]["sample_count"] = 9
        monkeypatch.setattr(quick_suite, "_load_quick_suite_manifest", lambda: manifest)
        assert get_quick_suite_content_sha256() != baseline

    def test_selection_algorithm_change_changes_hash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        baseline = get_quick_suite_content_sha256()
        manifest = _load_real_manifest()
        manifest["selection_algorithm"] = "sha512"
        monkeypatch.setattr(quick_suite, "_load_quick_suite_manifest", lambda: manifest)
        assert get_quick_suite_content_sha256() != baseline

    def test_corrupt_resources_fail_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A missing resource directory makes verify_quick_suite_resources raise
        # before any identity is minted.
        monkeypatch.setattr(quick_suite, "_RESOURCE_DIR", Path("/nonexistent/quick_v1"))
        with pytest.raises(BenchmarkAdapterError):
            get_quick_suite_content_sha256()

    def test_manual_canonical_payload_reconstruction(self) -> None:
        # Recompute the canonical payload exactly as §7 specifies; the
        # function must agree byte-for-byte.
        manifest = _load_real_manifest()
        tasks = manifest["tasks"]
        task_payloads = []
        for task_id in sorted(tasks):
            meta = tasks[task_id]
            task_payloads.append(
                {
                    "task_id": meta["task_id"],
                    "dimension": meta["dimension"],
                    "source_id": meta["source_id"],
                    "upstream_revision": meta["upstream_revision"],
                    "subset_sha256": meta["subset_sha256"],
                    "sample_count": meta["sample_count"],
                    "adapter_id": meta["adapter_id"],
                }
            )
        payload = {
            "suite_id": manifest["suite_id"],
            "suite_version": manifest["suite_version"],
            "total_items": manifest["total_items"],
            "selection_algorithm": manifest["selection_algorithm"],
            "selection_seed_format": manifest["selection_seed_format"],
            "tasks": task_payloads,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        assert get_quick_suite_content_sha256() == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class TestSourceRevisions:
    def test_returns_manifest_single_source(self) -> None:
        revisions = get_quick_suite_source_revisions()
        assert revisions == {
            "arc_challenge_quick_v1": "ARC-Challenge-2018",
            "humaneval_quick_v1": "human-eval-v1-2021",
            "gsm8k_quick_v1": "gsm8k-main-2023",
            "ifeval_quick_v1": "ifeval-v1-2023",
        }

    def test_keys_sorted(self) -> None:
        revisions = get_quick_suite_source_revisions()
        assert list(revisions.keys()) == sorted(revisions.keys())

    def test_corrupt_resources_fail_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(quick_suite, "_RESOURCE_DIR", Path("/nonexistent/quick_v1"))
        with pytest.raises(BenchmarkAdapterError):
            get_quick_suite_source_revisions()
