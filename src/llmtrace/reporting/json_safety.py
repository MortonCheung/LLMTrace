"""Public JSON-safety validation for metadata and report fields.

This module provides reusable functions that validate values are
JSON-safe — meaning they can be serialized via json.dumps() and
deserialized back without loss or ambiguity.

Allowed types:
  - None
  - bool
  - int
  - float
  - str
  - list (homogeneous, recursively checked)
  - dict[str, allowed_type] (recursively checked)

Rejected types that raise ValueError:
  - Non-string dict keys (int, float, tuple, etc.)
  - Pydantic BaseModel (hasattr model_dump)
  - BaseException and subclasses
  - bytes, bytearray
  - tuple, set, frozenset
  - datetime (all forms)
  - Any other custom / unknown type

Design note: We deliberately reject non-string dict keys rather than
converting them with str().  Python dicts {1: "a", "1": "b"} silently
overwrite each other after str() conversion, which is a data-loss
footgun.
"""

from __future__ import annotations

from collections.abc import Mapping

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sanitize_json_value(value: object, path: str = "$") -> None:
    """Validate that *value* is JSON-safe, raising ValueError otherwise.

    This is a validating-only function.  It does not return a new value.
    See validate_json_mapping() for dict-level validation.

    Args:
        value: The value to validate.
        path: Context path for error messages (e.g. "metadata['key']").

    Raises:
        ValueError: If *value* is not JSON-safe.
    """
    if value is None:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        return
    if isinstance(value, str):
        return

    # --- Reject forbidden types ---
    _refuse(value, path)

    if isinstance(value, list):
        for i, item in enumerate(value):
            sanitize_json_value(item, f"{path}[{i}]")
        return

    if isinstance(value, dict):
        for dk, dv in value.items():
            if not isinstance(dk, str):
                raise ValueError(
                    f"{path}: non-string dict key {dk!r} (type={type(dk).__name__}) — JSON requires string keys"
                )
            sanitize_json_value(dv, f"{path}.{dk}")
        return

    raise ValueError(f"{path}: unknown type {type(value).__name__} — not JSON-safe")


def validate_json_mapping(raw: Mapping[str, object]) -> None:
    """Validate that every value in *raw* is JSON-safe.

    Convenience wrapper around sanitize_json_value() for dict validation.

    Raises:
        ValueError: If any value is not JSON-safe.
    """
    for key, value in raw.items():
        sanitize_json_value(value, f"metadata['{key}']")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Types that are explicitly forbidden
_FORBIDDEN_JAVA_TYPES = frozenset({bytes, bytearray, tuple, set, frozenset})
# Base classes to check for datetime-like objects
_DATETIME_BASES = frozenset(
    {
        "date",
        "time",
        "datetime",
        "timedelta",
        "tzinfo",
        "timezone",
    }
)


def _refuse(value: object, path: str) -> None:
    """Raise ValueError if *value* is a forbidden type."""
    t = type(value)

    # Forbidden container types
    if t in _FORBIDDEN_JAVA_TYPES:
        raise ValueError(f"{path}: forbidden type {t.__name__} — not JSON-safe")

    # Pydantic BaseModel (duck-type)
    if hasattr(value, "model_dump"):
        raise ValueError(f"{path}: Pydantic model {t.__name__} — not allowed")

    # Exceptions
    if isinstance(value, BaseException):
        raise ValueError(f"{path}: Exception {t.__name__} — not allowed")

    # datetime-like objects (check MRO for known base names)
    mro_names = {c.__name__ for c in t.__mro__}
    if mro_names & _DATETIME_BASES:
        raise ValueError(f"{path}: datetime-like type {t.__name__} — must be ISO-8601 str, not raw object")
