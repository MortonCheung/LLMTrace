"""Registry for benchmark sources, suites, and adapters.

Provides simple ID-unique registries with:
- Duplicate registration errors
- ID-based lookup
- Read-only views (deep-copies for Source/Suite, tuple for Adapter list)
- Adapter instances returned by identity (not deep-copied)
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmtrace.adapters.base import BenchmarkAdapter
    from llmtrace.benchmarks.models import BenchmarkSource, BenchmarkSuite


class RegistryError(Exception):
    """Base exception for registry errors."""


class DuplicateRegistrationError(RegistryError):
    """Raised when attempting to register an item with a duplicate ID."""


class BenchmarkSourceRegistry:
    """Registry for BenchmarkSource objects.

    Stores deep-copies on register; returns deep-copies on get/list_all.
    External mutation of returned objects never affects internal state.
    """

    def __init__(self) -> None:
        self._sources: dict[str, BenchmarkSource] = {}

    def register(self, source: BenchmarkSource) -> None:
        """Register a benchmark source (stored as deep-copy)."""
        if source.source_id in self._sources:
            raise DuplicateRegistrationError(f"BenchmarkSource with id '{source.source_id}' is already registered")
        self._sources[source.source_id] = deepcopy(source)

    def get(self, source_id: str) -> BenchmarkSource | None:
        """Look up a source by ID (returns deep-copy)."""
        src = self._sources.get(source_id)
        return deepcopy(src) if src is not None else None

    def list_ids(self) -> tuple[str, ...]:
        """Return a tuple of all registered source IDs."""
        return tuple(self._sources.keys())

    def list_all(self) -> tuple[BenchmarkSource, ...]:
        """Return a tuple of deep-copied registered sources."""
        return tuple(deepcopy(s) for s in self._sources.values())

    def __len__(self) -> int:
        return len(self._sources)

    def __contains__(self, source_id: str) -> bool:
        return source_id in self._sources


class BenchmarkSuiteRegistry:
    """Registry for BenchmarkSuite objects.

    Stores deep-copies on register; returns deep-copies on get/list_all.
    External mutation of returned objects never affects internal state.
    """

    def __init__(self) -> None:
        self._suites: dict[str, BenchmarkSuite] = {}

    def register(self, suite: BenchmarkSuite) -> None:
        """Register a benchmark suite (stored as deep-copy)."""
        if suite.suite_id in self._suites:
            raise DuplicateRegistrationError(f"BenchmarkSuite with id '{suite.suite_id}' is already registered")
        self._suites[suite.suite_id] = deepcopy(suite)

    def get(self, suite_id: str) -> BenchmarkSuite | None:
        """Look up a suite by ID (returns deep-copy)."""
        sut = self._suites.get(suite_id)
        return deepcopy(sut) if sut is not None else None

    def list_ids(self) -> tuple[str, ...]:
        """Return a tuple of all registered suite IDs."""
        return tuple(self._suites.keys())

    def list_all(self) -> tuple[BenchmarkSuite, ...]:
        """Return a tuple of deep-copied registered suites."""
        return tuple(deepcopy(s) for s in self._suites.values())

    def __len__(self) -> int:
        return len(self._suites)

    def __contains__(self, suite_id: str) -> bool:
        return suite_id in self._suites


class BenchmarkAdapterRegistry:
    """Registry for BenchmarkAdapter instances.

    Adapters are executable service objects and are NOT deep-copied.
    The internal collection is never exposed directly; list_all returns
    a tuple of the registered instances, and get returns the instance
    by identity.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, BenchmarkAdapter] = {}

    def register(self, adapter: BenchmarkAdapter) -> None:
        """Register a benchmark adapter."""
        if adapter.adapter_id in self._adapters:
            raise DuplicateRegistrationError(f"BenchmarkAdapter with id '{adapter.adapter_id}' is already registered")
        self._adapters[adapter.adapter_id] = adapter

    def get(self, adapter_id: str) -> BenchmarkAdapter | None:
        """Look up an adapter by ID (returns the registered instance)."""
        return self._adapters.get(adapter_id)

    def list_ids(self) -> tuple[str, ...]:
        """Return a tuple of all registered adapter IDs."""
        return tuple(self._adapters.keys())

    def list_all(self) -> tuple[BenchmarkAdapter, ...]:
        """Return a tuple of the registered adapter instances."""
        return tuple(self._adapters.values())

    def __len__(self) -> int:
        return len(self._adapters)

    def __contains__(self, adapter_id: str) -> bool:
        return adapter_id in self._adapters
