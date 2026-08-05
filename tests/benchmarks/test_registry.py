"""Tests for registry functionality."""

from __future__ import annotations

import pytest

from llmtrace.benchmarks.models import BenchmarkSource, BenchmarkSuite, SuiteVersion
from llmtrace.benchmarks.registry import (
    BenchmarkAdapterRegistry,
    BenchmarkSourceRegistry,
    BenchmarkSuiteRegistry,
    DuplicateRegistrationError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source(source_id: str = "mmlu") -> BenchmarkSource:
    return BenchmarkSource(source_id=source_id, name=source_id.upper())


def _make_suite(suite_id: str = "mmlu") -> BenchmarkSuite:
    return BenchmarkSuite(
        suite_id=suite_id,
        name=suite_id.upper(),
        version=SuiteVersion(version="1.0.0"),
        source_id=suite_id,
        source_revision="abc123",
    )


# ---------------------------------------------------------------------------
# Fake adapter for registry testing
# ---------------------------------------------------------------------------


class _FakeAdapterForRegistry:
    """Minimal fake adapter for registry tests only."""

    def __init__(self, adapter_id: str = "fake", adapter_version: str = "1.0.0") -> None:
        self._adapter_id = adapter_id
        self._adapter_version = adapter_version

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    @property
    def adapter_version(self) -> str:
        return self._adapter_version


# ============================================================================
# BenchmarkSourceRegistry
# ============================================================================


class TestBenchmarkSourceRegistry:
    def test_register_and_get(self) -> None:
        reg = BenchmarkSourceRegistry()
        src = _make_source("mmlu")
        reg.register(src)
        assert reg.get("mmlu") == src

    def test_list_ids(self) -> None:
        reg = BenchmarkSourceRegistry()
        reg.register(_make_source("mmlu"))
        reg.register(_make_source("livebench"))
        ids = reg.list_ids()
        assert sorted(ids) == ["livebench", "mmlu"]

    def test_list_all(self) -> None:
        reg = BenchmarkSourceRegistry()
        reg.register(_make_source("mmlu"))
        reg.register(_make_source("livebench"))
        all_sources = reg.list_all()
        assert len(all_sources) == 2
        # Verify they are copies (deepcopy), not the original references
        all_sources[0].name = "Modified"
        assert reg.get(all_sources[0].source_id).name == all_sources[0].source_id.upper()  # type: ignore[union-attr]

    def test_duplicate_registration_raises(self) -> None:
        reg = BenchmarkSourceRegistry()
        reg.register(_make_source("mmlu"))
        with pytest.raises(DuplicateRegistrationError, match="already registered"):
            reg.register(_make_source("mmlu"))

    def test_get_nonexistent(self) -> None:
        reg = BenchmarkSourceRegistry()
        assert reg.get("nonexistent") is None

    def test_contains(self) -> None:
        reg = BenchmarkSourceRegistry()
        reg.register(_make_source("mmlu"))
        assert "mmlu" in reg
        assert "other" not in reg

    def test_len(self) -> None:
        reg = BenchmarkSourceRegistry()
        assert len(reg) == 0
        reg.register(_make_source("a"))
        assert len(reg) == 1


# ============================================================================
# BenchmarkSuiteRegistry
# ============================================================================


class TestBenchmarkSuiteRegistry:
    def test_register_and_get(self) -> None:
        reg = BenchmarkSuiteRegistry()
        suite = _make_suite("mmlu")
        reg.register(suite)
        assert reg.get("mmlu") == suite

    def test_duplicate_registration_raises(self) -> None:
        reg = BenchmarkSuiteRegistry()
        reg.register(_make_suite("mmlu"))
        with pytest.raises(DuplicateRegistrationError, match="already registered"):
            reg.register(_make_suite("mmlu"))

    def test_list_all_returns_copies(self) -> None:
        reg = BenchmarkSuiteRegistry()
        reg.register(_make_suite("mmlu"))
        all_suites = reg.list_all()
        all_suites[0].name = "Modified"
        assert reg.get("mmlu").name == "MMLU"  # type: ignore[union-attr]

    def test_get_nonexistent(self) -> None:
        reg = BenchmarkSuiteRegistry()
        assert reg.get("nonexistent") is None

    def test_len(self) -> None:
        reg = BenchmarkSuiteRegistry()
        assert len(reg) == 0
        reg.register(_make_suite("a"))
        reg.register(_make_suite("b"))
        assert len(reg) == 2


# ============================================================================
# BenchmarkAdapterRegistry
# ============================================================================


class TestBenchmarkAdapterRegistry:
    def test_register_and_get(self) -> None:
        reg = BenchmarkAdapterRegistry()
        adapter = _FakeAdapterForRegistry("lm-eval")
        reg.register(adapter)  # type: ignore[arg-type]
        assert reg.get("lm-eval") is adapter

    def test_duplicate_registration_raises(self) -> None:
        reg = BenchmarkAdapterRegistry()
        reg.register(_FakeAdapterForRegistry("lm-eval"))  # type: ignore[arg-type]
        with pytest.raises(DuplicateRegistrationError, match="already registered"):
            reg.register(_FakeAdapterForRegistry("lm-eval"))  # type: ignore[arg-type]

    def test_list_ids(self) -> None:
        reg = BenchmarkAdapterRegistry()
        reg.register(_FakeAdapterForRegistry("a"))  # type: ignore[arg-type]
        reg.register(_FakeAdapterForRegistry("b"))  # type: ignore[arg-type]
        assert sorted(reg.list_ids()) == ["a", "b"]

    def test_contains(self) -> None:
        reg = BenchmarkAdapterRegistry()
        reg.register(_FakeAdapterForRegistry("lm-eval"))  # type: ignore[arg-type]
        assert "lm-eval" in reg
        assert "other" not in reg

    def test_len(self) -> None:
        reg = BenchmarkAdapterRegistry()
        assert len(reg) == 0
        reg.register(_FakeAdapterForRegistry("a"))  # type: ignore[arg-type]
        assert len(reg) == 1
