"""Scoring-specific exceptions."""

from __future__ import annotations


class ScoringError(Exception):
    """Base exception for all scoring-related errors."""


class InvalidPolicyError(ScoringError):
    """Raised when a scoring policy is invalid (e.g. weight sum != 1.0)."""


class TaskRegistrationError(ScoringError):
    """Raised when a task cannot be registered in the scoring registry."""


class AggregationError(ScoringError):
    """Raised when dimension aggregation fails (e.g. no valid graded tasks)."""
