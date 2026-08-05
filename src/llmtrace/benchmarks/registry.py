"""Registry for benchmark sources, suites, and adapters.

Provides simple ID-unique registries with:
- duplicate registration errors
- ID-based lookup
- listing of all registered items
- read-only views (copies)
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

    Ensures unique source_id per registered source.
    """

    def __init__(self) -> None:
        self._sources: dict[str, BenchmarkSource] = {}

    def register(self, source: BenchmarkSource) -> None:
        """Register a benchmark source.

        Args:
            source: The BenchmarkSource to register.

        Raises:
            DuplicateRegistrationError: If source_id already registered.
        """
        if source.source_id in self._sources:
            raise DuplicateRegistrationError(f"BenchmarkSource with id '{source.source_id}' is already registered")
        self._sources[source.source_id] = source

    def get(self, source_id: str) -> BenchmarkSource | None:
        """Look up a source by ID.

        Returns:
            The BenchmarkSource, or None if not found.
        """
        return self._sources.get(source_id)

    def list_ids(self) -> list[str]:
        """List all registered source IDs.

        Returns:
            A copy of the list of source IDs.
        """
        return list(self._sources.keys())

    def list_all(self) -> list[BenchmarkSource]:
        """List all registered sources.

        Returns:
            A copy of the list of registered sources.
        """
        return [deepcopy(s) for s in self._sources.values()]

    def __len__(self) -> int:
        return len(self._sources)

    def __contains__(self, source_id: str) -> bool:
        return source_id in self._sources


class BenchmarkSuiteRegistry:
    """Registry for BenchmarkSuite objects.

    Ensures unique suite_id per registered suite.
    """

    def __init__(self) -> None:
        self._suites: dict[str, BenchmarkSuite] = {}

    def register(self, suite: BenchmarkSuite) -> None:
        """Register a benchmark suite.

        Args:
            suite: The BenchmarkSuite to register.

        Raises:
            DuplicateRegistrationError: If suite_id already registered.
        """
        if suite.suite_id in self._suites:
            raise DuplicateRegistrationError(f"BenchmarkSuite with id '{suite.suite_id}' is already registered")
        self._suites[suite.suite_id] = suite

    def get(self, suite_id: str) -> BenchmarkSuite | None:
        """Look up a suite by ID.

        Returns:
            The BenchmarkSuite, or None if not found.
        """
        return self._suites.get(suite_id)

    def list_ids(self) -> list[str]:
        """List all registered suite IDs.

        Returns:
            A copy of the list of suite IDs.
        """
        return list(self._suites.keys())

    def list_all(self) -> list[BenchmarkSuite]:
        """List all registered suites.

        Returns:
            A copy of the list of registered suites.
        """
        return [deepcopy(s) for s in self._suites.values()]

    def __len__(self) -> int:
        return len(self._suites)

    def __contains__(self, suite_id: str) -> bool:
        return suite_id in self._suites


class BenchmarkAdapterRegistry:
    """Registry for BenchmarkAdapter instances.

    Ensures unique adapter_id per registered adapter.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, BenchmarkAdapter] = {}

    def register(self, adapter: BenchmarkAdapter) -> None:
        """Register a benchmark adapter.

        Args:
            adapter: The BenchmarkAdapter to register.

        Raises:
            DuplicateRegistrationError: If adapter_id already registered.
        """
        if adapter.adapter_id in self._adapters:
            raise DuplicateRegistrationError(f"BenchmarkAdapter with id '{adapter.adapter_id}' is already registered")
        self._adapters[adapter.adapter_id] = adapter

    def get(self, adapter_id: str) -> BenchmarkAdapter | None:
        """Look up an adapter by ID.

        Returns:
            The BenchmarkAdapter, or None if not found.
        """
        return self._adapters.get(adapter_id)

    def list_ids(self) -> list[str]:
        """List all registered adapter IDs.

        Returns:
            A copy of the list of adapter IDs.
        """
        return list(self._adapters.keys())

    def list_all(self) -> list[BenchmarkAdapter]:
        """List all registered adapters.

        Returns:
            A copy of the list of registered adapters.
        """
        return list(self._adapters.values())

    def __len__(self) -> int:
        return len(self._adapters)

    def __contains__(self, adapter_id: str) -> bool:
        return adapter_id in self._adapters
