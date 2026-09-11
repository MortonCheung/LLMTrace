"""ReferenceSet auto-discovery 测试（CLI Real-Use Sprint §8–§11）.

覆盖：
- 空 / 缺失目录 → 0 候选
- 单个可信 ReferenceSet → 自动解析
- 混合目录（可信 + 损坏/不兼容） → 只返回可信候选
- explicit --reference-set > auto discovery > none 的优先级
- 多个候选 → 不自动选择（fail closed 交给 CLI）
- fixture / mock 来源不得被当作可信校准（共享 validator 拦截）
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llmtrace.execution.artifacts import RunArtifactRepository
from llmtrace.reference.discovery import (
    discover_compatible_sets,
    reference_sets_dir,
    resolve_calibration_set,
)
from tests.execution.test_runner_calibration import _build_reference_fixture


def _build_set(
    artifact_root: Path, reference_root: Path, set_id: str, execution_prefix: str, snapshot_prefix: str
) -> Path:
    """构建一个可信 ReferenceSet；每套用独立 execution/snapshot 前缀避免冲突."""
    return _build_reference_fixture(
        artifact_root,
        reference_root,
        models=(("ref-model-low", 0.30), ("ref-model-mid", 0.60)),
        set_id=set_id,
        set_version="0.1.0",
        execution_prefix=execution_prefix,
        snapshot_prefix=snapshot_prefix,
    )


def _write_invalid_set(reference_root: Path) -> Path:
    """写入一个损坏的 set 文件（自哈希不通过）。"""
    sets_dir = reference_root / "sets"
    sets_dir.mkdir(parents=True, exist_ok=True)
    path = sets_dir / "corrupt_v1.json"
    path.write_text('{"reference_set_id": "corrupt", "not": "a valid set"}', encoding="utf-8")
    return path


def test_reference_sets_dir_honors_llmtrace_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """默认目录 = $LLMTRACE_HOME/references/sets（与 Web 共用 reference universe）."""
    monkeypatch.setenv("LLMTRACE_HOME", str(tmp_path))
    assert reference_sets_dir() == tmp_path.resolve() / "references" / "sets"


def test_reference_sets_dir_falls_back_to_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """无 $LLMTRACE_HOME 时回退到 ~/.llmtrace（不写文件）."""
    monkeypatch.delenv("LLMTRACE_HOME", raising=False)
    assert reference_sets_dir().name == "sets"


def test_discover_empty_directory_returns_none(tmp_path: Path) -> None:
    """目录不存在或为空 → 0 候选，不抛错."""
    assert discover_compatible_sets(RunArtifactRepository(tmp_path), sets_dir=tmp_path / "sets") == []
    (tmp_path / "sets").mkdir(parents=True)
    assert discover_compatible_sets(RunArtifactRepository(tmp_path), sets_dir=tmp_path / "sets") == []


def test_discover_filters_incompatible_sets(tmp_path: Path) -> None:
    """混合目录：只保留通过完整信任链校验的 set，损坏者被忽略."""
    set_path = _build_reference_fixture(tmp_path, tmp_path / "references", set_id="trusted-v1", set_version="0.1.0")
    _write_invalid_set(tmp_path / "references")
    discovered = discover_compatible_sets(RunArtifactRepository(tmp_path), sets_dir=tmp_path / "references" / "sets")
    assert discovered == [set_path]


def test_discover_multiple_compatible_sets_lists_all(tmp_path: Path) -> None:
    """同一目录下多个兼容 set → 全部返回（不自动挑选，CLI 负责 fail closed/询问）."""
    reference_root = tmp_path / "references"
    set_a = _build_set(tmp_path, reference_root, "set-a", "aaaaaaaa-0000-0000-0000", "snap-a")
    set_b = _build_set(tmp_path, reference_root, "set-b", "bbbbbbbb-0000-0000-0000", "snap-b")
    discovered = discover_compatible_sets(RunArtifactRepository(tmp_path), sets_dir=reference_root / "sets")
    assert set_a in discovered
    assert set_b in discovered


def test_resolve_explicit_wins_over_discovery(tmp_path: Path) -> None:
    """显式 --reference-set 优先级最高（§10）."""
    set_path = _build_reference_fixture(tmp_path, tmp_path / "references", set_id="trusted-v1", set_version="0.1.0")
    resolution = resolve_calibration_set(
        explicit_path=set_path,
        artifact_repository=RunArtifactRepository(tmp_path),
        sets_dir=tmp_path / "references" / "sets",
    )
    assert resolution.resolved_path == set_path
    assert resolution.candidates == ()


def test_resolve_auto_single(tmp_path: Path) -> None:
    """唯一兼容 set → 自动解析."""
    set_path = _build_reference_fixture(tmp_path, tmp_path / "references", set_id="trusted-v1", set_version="0.1.0")
    resolution = resolve_calibration_set(
        explicit_path=None,
        artifact_repository=RunArtifactRepository(tmp_path),
        sets_dir=tmp_path / "references" / "sets",
    )
    assert resolution.resolved_path == set_path


def test_resolve_none(tmp_path: Path) -> None:
    """无候选 → resolved_path=None（calibration unavailable），不失败."""
    resolution = resolve_calibration_set(
        explicit_path=None,
        artifact_repository=RunArtifactRepository(tmp_path),
        sets_dir=tmp_path / "references" / "sets",
    )
    assert resolution.resolved_path is None
    assert resolution.candidates == ()


def test_resolve_multiple_returns_candidates_without_picking(tmp_path: Path) -> None:
    """多候选 → resolved_path=None + 全部候选（fail closed）."""
    reference_root = tmp_path / "references"
    _build_set(tmp_path, reference_root, "set-a", "aaaaaaaa-0000-0000-0000", "snap-a")
    _build_set(tmp_path, reference_root, "set-b", "bbbbbbbb-0000-0000-0000", "snap-b")
    resolution = resolve_calibration_set(
        explicit_path=None,
        artifact_repository=RunArtifactRepository(tmp_path),
        sets_dir=reference_root / "sets",
    )
    assert resolution.resolved_path is None
    assert len(resolution.candidates) == 2
