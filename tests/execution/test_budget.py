"""Tests for the RequestBudget guard (execution/budget.py)."""

from __future__ import annotations

import pytest

from llmtrace.execution.budget import RequestBudget, RequestBudgetExceededError


class TestRequestBudget:
    def test_consumes_within_budget(self) -> None:
        b = RequestBudget(3)
        b.consume(1)
        assert b.consumed_requests == 1
        assert b.remaining_requests == 2

    def test_exceeding_budget_raises(self) -> None:
        b = RequestBudget(1)
        b.consume(1)
        with pytest.raises(RequestBudgetExceededError):
            b.consume(1)
        # The failed consume must not increment the counter.
        assert b.consumed_requests == 1

    def test_bulk_consume_counts_exactly(self) -> None:
        b = RequestBudget(5)
        b.consume(3)
        assert b.consumed_requests == 3
        assert b.remaining_requests == 2

    def test_bulk_consume_exceeding_raises_atomically(self) -> None:
        b = RequestBudget(2)
        with pytest.raises(RequestBudgetExceededError):
            b.consume(3)
        assert b.consumed_requests == 0

    def test_negative_maximum_rejected(self) -> None:
        with pytest.raises(ValueError):
            RequestBudget(-1)

    def test_non_positive_count_rejected(self) -> None:
        b = RequestBudget(5)
        with pytest.raises(ValueError):
            b.consume(0)
        with pytest.raises(ValueError):
            b.consume(-2)

    def test_zero_budget_immediately_exhausted(self) -> None:
        b = RequestBudget(0)
        assert b.remaining_requests == 0
        with pytest.raises(RequestBudgetExceededError):
            b.consume(1)

    def test_error_code_is_machine_readable(self) -> None:
        assert RequestBudgetExceededError.error_code == "REQUEST_BUDGET_EXCEEDED"
