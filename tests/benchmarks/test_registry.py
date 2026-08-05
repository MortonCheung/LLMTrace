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
        self._called: int = 0

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

    def test_list_ids_returns_tuple(self) -> None:
        reg = BenchmarkSourceRegistry()
        reg.register(_make_source("mmlu"))
        reg.register(_make_source("livebench"))
        ids = reg.list_ids()
        assert isinstance(ids, tuple)
        assert sorted(ids) == ["livebench", "mmlu"]

    def test_list_all_returns_tuple_of_deep_copies(self) -> None:
        reg = BenchmarkSourceRegistry()
        reg.register(_make_source("mmlu"))
        all_sources = reg.list_all()
        assert isinstance(all_sources, tuple)
        # Mutate returned object must not affect registry
        all_sources[0].name = "Modified"
        assert reg.get("mmlu").name == "MMLU"  # type: ignore[union-attr]

    def test_get_returns_deep_copy(self) -> None:
        reg = BenchmarkSourceRegistry()
        reg.register(_make_source("mmlu"))
        src = reg.get("mmlu")
        assert src is not None
        src.name = "Modified"
        # Internal state unchanged
        assert reg.get("mmlu").name == "MMLU"  # type: ignore[union-attr]

    def test_register_stores_deep_copy(self) -> None:
        reg = BenchmarkSourceRegistry()
        src = _make_source("mmlu")
        reg.register(src)
        src.name = "Modified"
        assert reg.get("mmlu").name == "MMLU"  # type: ignore[union-attr]

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

    def test_get_returns_deep_copy(self) -> None:
        reg = BenchmarkSuiteRegistry()
        reg.register(_make_suite("mmlu"))
        sut = reg.get("mmlu")
        assert sut is not None
        sut.name = "Modified"
        assert reg.get("mmlu").name == "MMLU"  # type: ignore[union-attr]

    def test_register_stores_deep_copy(self) -> None:
        reg = BenchmarkSuiteRegistry()
        suite = _make_suite("mmlu")
        reg.register(suite)
        suite.name = "Modified"
        assert reg.get("mmlu").name == "MMLU"  # type: ignore[union-attr]

    def test_list_all_returns_tuple_of_deep_copies(self) -> None:
        reg = BenchmarkSuiteRegistry()
        reg.register(_make_suite("mmlu"))
        all_suites = reg.list_all()
        assert isinstance(all_suites, tuple)
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
    def test_register_and_get_returns_same_instance(self) -> None:
        reg = BenchmarkAdapterRegistry()
        adapter = _FakeAdapterForRegistry("lm-eval")
        reg.register(adapter)  # type: ignore[arg-type]
        assert reg.get("lm-eval") is adapter

    def test_duplicate_registration_raises(self) -> None:
        reg = BenchmarkAdapterRegistry()
        reg.register(_FakeAdapterForRegistry("lm-eval"))  # type: ignore[arg-type]
        with pytest.raises(DuplicateRegistrationError, match="already registered"):
            reg.register(_FakeAdapterForRegistry("lm-eval"))  # type: ignore[arg-type]

    def test_list_ids_returns_tuple(self) -> None:
        reg = BenchmarkAdapterRegistry()
        reg.register(_FakeAdapterForRegistry("a"))  # type: ignore[arg-type]
        reg.register(_FakeAdapterForRegistry("b"))  # type: ignore[arg-type]
        ids = reg.list_ids()
        assert isinstance(ids, tuple)
        assert sorted(ids) == ["a", "b"]

    def test_list_all_returns_tuple(self) -> None:
        reg = BenchmarkAdapterRegistry()
        reg.register(_FakeAdapterForRegistry("a"))  # type: ignore[arg-type]
        all_adapters = reg.list_all()
        assert isinstance(all_adapters, tuple)

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
