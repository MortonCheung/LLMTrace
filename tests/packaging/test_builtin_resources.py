"""Verify built-in smoke resources are included in wheel distributions.

Tests that:
1. python -m build produces a wheel
2. The wheel contains the YAML and JSON smoke resources
3. From an installed environment, BUILTIN_SMOKE_TASK_ROOT is accessible
4. Both files are readable

These tests validate the [tool.setuptools.package-data] configuration.
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


def _find_wheels(dist_dir: Path) -> list[Path]:
    """Find .whl files in the dist directory."""
    return sorted(dist_dir.glob("*.whl"))


def _build_wheel(project_root: Path) -> Path:
    """Build a wheel in an isolated temp dir and return the dist directory."""
    dist_dir = project_root / "dist"

    # Clean any stale builds first
    if dist_dir.exists():
        import shutil

        shutil.rmtree(dist_dir)

    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Wheel build failed:\n{result.stderr}")

    wheels = _find_wheels(dist_dir)
    if not wheels:
        raise RuntimeError("No .whl files found in dist/ after build")
    return dist_dir


# ---------------------------------------------------------------------------
# Session-scoped fixtures (shared across all tests in this module)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory (parent of the tests directory)."""
    return Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="session")
def dist_dir(project_root: Path) -> Path:
    """Build a wheel and return the dist directory."""
    return _build_wheel(project_root)


@pytest.fixture(scope="session")
def wheel_path(dist_dir: Path) -> Path:
    """Return the path to the single built wheel."""
    wheels = _find_wheels(dist_dir)
    assert len(wheels) == 1, f"Expected exactly one wheel, got {len(wheels)}"
    return wheels[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuiltinResourcesInWheel:
    """Verify that the built-in YAML and JSON resources are in the wheel."""

    def test_wheel_contains_smoke_yaml(self, wheel_path: Path) -> None:
        """The wheel must contain llmtrace/adapters/_resources/llmtrace_smoke.yaml."""
        with zipfile.ZipFile(wheel_path) as zf:
            names = zf.namelist()
            assert "llmtrace/adapters/_resources/llmtrace_smoke.yaml" in names, (
                f"llmtrace_smoke.yaml not found in wheel. Files: {sorted(names)}"
            )

    def test_wheel_contains_smoke_json(self, wheel_path: Path) -> None:
        """The wheel must contain llmtrace/adapters/_resources/llmtrace_smoke.json."""
        with zipfile.ZipFile(wheel_path) as zf:
            names = zf.namelist()
            assert "llmtrace/adapters/_resources/llmtrace_smoke.json" in names, (
                f"llmtrace_smoke.json not found in wheel. Files: {sorted(names)}"
            )

    def test_wheel_yaml_is_readable(self, wheel_path: Path) -> None:
        """The YAML resource in the wheel is readable and non-empty."""
        with zipfile.ZipFile(wheel_path) as zf:
            content = zf.read("llmtrace/adapters/_resources/llmtrace_smoke.yaml")
            assert len(content) > 0
            text = content.decode("utf-8")
            assert "output_type" in text

    def test_wheel_json_is_readable(self, wheel_path: Path) -> None:
        """The JSON resource in the wheel is readable and parseable."""
        with zipfile.ZipFile(wheel_path) as zf:
            content = zf.read("llmtrace/adapters/_resources/llmtrace_smoke.json")
            assert len(content) > 0
            data = json.loads(content.decode("utf-8"))
            assert isinstance(data, list)
            assert len(data) == 4


class TestBuiltinResourcesInstalled:
    """Verify BUILTIN_SMOKE_TASK_ROOT is accessible from the installed package.

    These tests run against the currently installed llmtrace (editable or not).
    """

    def test_builtin_root_exists(self) -> None:
        from llmtrace.adapters.lm_eval import BUILTIN_SMOKE_TASK_ROOT

        root = Path(BUILTIN_SMOKE_TASK_ROOT)
        assert root.exists(), f"BUILTIN_SMOKE_TASK_ROOT does not exist: {root}"
        assert root.is_dir()

    def test_builtin_yaml_exists(self) -> None:
        from llmtrace.adapters.lm_eval import BUILTIN_SMOKE_TASK_ROOT

        yaml_path = Path(BUILTIN_SMOKE_TASK_ROOT) / "llmtrace_smoke.yaml"
        assert yaml_path.exists(), f"YAML not found: {yaml_path}"
        assert yaml_path.is_file()

    def test_builtin_json_exists(self) -> None:
        from llmtrace.adapters.lm_eval import BUILTIN_SMOKE_TASK_ROOT

        json_path = Path(BUILTIN_SMOKE_TASK_ROOT) / "llmtrace_smoke.json"
        assert json_path.exists(), f"JSON not found: {json_path}"
        assert json_path.is_file()

    def test_builtin_yaml_readable(self) -> None:
        from llmtrace.adapters.lm_eval import BUILTIN_SMOKE_TASK_ROOT

        yaml_path = Path(BUILTIN_SMOKE_TASK_ROOT) / "llmtrace_smoke.yaml"
        content = yaml_path.read_text()
        assert "output_type" in content

    def test_builtin_json_readable(self) -> None:
        from llmtrace.adapters.lm_eval import BUILTIN_SMOKE_TASK_ROOT

        json_path = Path(BUILTIN_SMOKE_TASK_ROOT) / "llmtrace_smoke.json"
        data = json.loads(json_path.read_text())
        assert len(data) == 4
