"""RunService vertical tests — create / estimate / start / progress / cancel /
result / history against a mocked OpenAI endpoint + trusted test backend.

Checks that the thin service layer (v0.5 §十三/§十四) drives the real
``UnifiedAuditRunner``, persists terminal state into SQLite, never leaks the
API key to disk, and keeps cancelled runs free of report artifacts.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import httpx
import pytest
import respx

from llmtrace.appdir import AppLayout
from llmtrace.execution.progress import EVENT_PROGRESS, STAGE_BENCHMARK
from llmtrace.service import (
    CreateRunInput,
    RunCreateError,
    RunKeyUnavailableError,
    RunNotFoundError,
    RunService,
    RunStateConflictError,
)
from llmtrace.service.runs import TERMINAL_EVENT
from tests.execution.conftest import TrustedFakeBackend

API_KEY = "sk-super-secret-123"
BASE_URL = "http://test.example.com/v1"
MODEL = "my-real-model"


def _completion_json(content: str, model: str = MODEL) -> dict[str, object]:
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
    }


def _models_json(model: str = MODEL) -> dict[str, object]:
    return {"object": "list", "data": [{"id": model, "object": "model"}]}


def _mock_openai(respx_mock: respx.MockRouter, *, delay: float = 0.0) -> None:
    respx_mock.get(f"{BASE_URL}/models").respond(status_code=200, json=_models_json())

    async def _slow_completion(request: httpx.Request) -> httpx.Response:
        if delay > 0:
            await asyncio.sleep(delay)
        return httpx.Response(
            status_code=200,
            json=_completion_json("The answer is (A). The answer is 42."),
            headers={"content-type": "application/json"},
        )

    if delay > 0:
        # 带延迟的 mock 给取消测试留出 item 边界上的取消窗口。
        respx_mock.post(f"{BASE_URL}/chat/completions").mock(side_effect=_slow_completion)
    else:
        respx_mock.post(f"{BASE_URL}/chat/completions").respond(
            status_code=200,
            json=_completion_json("The answer is (A). The answer is 42."),
            headers={"content-type": "application/json"},
        )


def _make_service(tmp_path: Path) -> RunService:
    return RunService(AppLayout.resolve(tmp_path), code_backend=TrustedFakeBackend())


def _make_input() -> CreateRunInput:
    return CreateRunInput(
        base_url=BASE_URL,
        model=MODEL,
        api_key=API_KEY,
        repeat=1,
        timeout=10.0,
        check_streaming=False,
    )


async def _wait_terminal(svc: RunService, run_id: str, *, timeout: float = 30.0) -> dict[str, object]:
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        snap = svc.progress(run_id)
        if snap["terminal"]:
            return snap
        if asyncio.get_event_loop().time() > deadline:
            pytest.fail(f"run {run_id} did not finish in time; status={snap['status']}")
        await asyncio.sleep(0.05)


class TestCreateEstimate:
    def test_create_estimate_and_key_never_on_disk(self, tmp_path: Path) -> None:
        svc = _make_service(tmp_path)
        try:
            record = svc.create(_make_input())
            assert record.status == "PENDING"

            est = svc.estimate(record.run_id)
            assert est.reference_calibration is False
            assert est.benchmark_requests == 32
            assert est.maximum_requests >= est.planned_requests
            assert est.requires_secure_code_sandbox is True

            # API key 永不越过持久化边界。
            conn = sqlite3.connect(str(tmp_path / "llmtrace.db"))
            blob = " ".join(str(r) for r in conn.execute("SELECT * FROM runs").fetchall())
            conn.close()
            assert API_KEY not in blob
        finally:
            svc.close()

    def test_credentials_in_base_url_are_redacted_on_disk(self, tmp_path: Path) -> None:
        svc = _make_service(tmp_path)
        try:
            record = svc.create(
                CreateRunInput(
                    base_url="https://user:sekret-pass@host.example/v1",
                    model=MODEL,
                    api_key=API_KEY,
                )
            )
            conn = sqlite3.connect(str(tmp_path / "llmtrace.db"))
            blob = " ".join(str(r) for r in conn.execute("SELECT * FROM runs").fetchall())
            conn.close()
            assert "sekret-pass" not in blob
            assert API_KEY not in blob
            assert "user:sekret-pass" not in record.base_url_redacted
        finally:
            svc.close()

    def test_estimate_missing_run_raises(self, tmp_path: Path) -> None:
        svc = _make_service(tmp_path)
        try:
            with pytest.raises(RunNotFoundError):
                svc.estimate("nope")
        finally:
            svc.close()

    def test_create_rejects_empty_api_key(self, tmp_path: Path) -> None:
        svc = _make_service(tmp_path)
        try:
            with pytest.raises(RunCreateError):
                svc.create(_make_input().__class__(base_url=BASE_URL, model=MODEL, api_key="  "))
        finally:
            svc.close()


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_full_run_completes_and_persists(self, tmp_path: Path) -> None:
        svc = _make_service(tmp_path)
        try:
            record = svc.create(_make_input())
            with respx.mock as mock:
                _mock_openai(mock)
                svc.start(record.run_id)

                snap = await _wait_terminal(svc, record.run_id)
            assert snap["terminal_status"] == "completed"
            assert snap["status"] == "COMPLETED"
            assert snap["events"][-1]["type"] == TERMINAL_EVENT
            # Benchmark item progress reached the full suite.
            progress = [e for e in snap["events"] if e["type"] == EVENT_PROGRESS]
            assert progress and progress[-1]["completed"] == 32

            result = svc.result(record.run_id)
            assert result["report"] is not None
            assert result["run"]["capability_score"] is not None
            assert result["run"]["confidence"] in ("High", "Medium", "Low", "Unavailable")
            assert result["run"]["request_count"] >= 1
            assert Path(result["run"]["report_json_path"]).is_file()
            assert Path(result["run"]["report_html_path"]).is_file()

            history = svc.history()
            assert [r["run_id"] for r in history] == [record.run_id]
            assert history[0]["status"] == "COMPLETED"

            # Restart-safe view: a fresh service instance (no in-memory key)
            # can still read history / result but cannot start a terminal run.
            svc.close()
            svc2 = RunService(AppLayout.resolve(tmp_path))
            try:
                assert svc2.result(record.run_id)["run"]["status"] == "COMPLETED"
                with pytest.raises(RunStateConflictError):
                    svc2.start(record.run_id)  # 终态 run 拒绝 start
            finally:
                svc2.close()

            # Key only lives in process memory: a PENDING run created before a
            # "restart" has no key in the fresh process -> RUN_KEY_UNAVAILABLE.
            svc3 = _make_service(tmp_path)
            pending = svc3.create(_make_input())
            svc3.close()
            svc4 = RunService(AppLayout.resolve(tmp_path))
            try:
                assert svc4.history()[0]["status"] == "PENDING"
                with pytest.raises(RunKeyUnavailableError):
                    svc4.start(pending.run_id)
            finally:
                svc4.close()
        finally:
            pass

    @pytest.mark.asyncio
    async def test_result_view_tolerates_the_v06_report_schema(self, tmp_path: Path) -> None:
        """Task 45：Web 冻结 —— 新增的 optional fingerprint 段不得让结果视图 crash。

        两条保证：(1) 未接线身份证据的 run 报告形状不变（consumers 读的
        ``capability_profile`` 仍在、没有 surprise key）；(2) 一旦报告里出现
        v0.6 的 fingerprint 段，service 的 raw JSON 视图照常返回而不报错。
        """
        svc = _make_service(tmp_path)
        try:
            record = svc.create(_make_input())
            with respx.mock as mock:
                _mock_openai(mock)
                svc.start(record.run_id)
                await _wait_terminal(svc, record.run_id)

            report = svc.result(record.run_id)["report"]
            assert report is not None
            assert report["schema_version"] == "1.4"
            # web/static/app.js 依赖的字段仍在（additive change only）。
            assert "capability_profile" in report
            assert "fingerprint" not in report  # 未接线 → 不写该段

            # 手工注入 v0.6 段：模拟将来接线的 run，Web 仍必须能读。
            report_path = Path(svc.result(record.run_id)["run"]["report_json_path"])
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            payload["fingerprint"] = {"available": True, "experimental": True, "verdict": None}
            report_path.write_text(json.dumps(payload), encoding="utf-8")

            reloaded = svc.result(record.run_id)
            assert reloaded["report"]["fingerprint"]["experimental"] is True
            assert reloaded["report"]["capability_profile"] == report["capability_profile"]
        finally:
            svc.close()

    @pytest.mark.asyncio
    async def test_double_start_conflicts(self, tmp_path: Path) -> None:
        svc = _make_service(tmp_path)
        try:
            record = svc.create(_make_input())
            with respx.mock as mock:
                _mock_openai(mock)
                svc.start(record.run_id)
                with pytest.raises(RunStateConflictError):
                    svc.start(record.run_id)
                await _wait_terminal(svc, record.run_id)
            with pytest.raises(RunStateConflictError):
                svc.start(record.run_id)  # 终态后不可再 start
        finally:
            svc.close()

    @pytest.mark.asyncio
    async def test_cancel_lands_cancelled_without_artifacts(self, tmp_path: Path) -> None:
        svc = _make_service(tmp_path)
        try:
            record = svc.create(_make_input())
            with respx.mock as mock:
                # 15ms/请求的延迟给轮询留出 item 边界上的取消窗口（瞬时 mock
                # 会在 cancel 命中前跑完全部 32 个 item）。
                _mock_openai(mock, delay=0.015)
                svc.start(record.run_id)
                deadline = asyncio.get_event_loop().time() + 15
                cancelled = False
                while asyncio.get_event_loop().time() < deadline:
                    snap = svc.progress(record.run_id)
                    first_item = any(
                        e["type"] == EVENT_PROGRESS and e["stage"] == STAGE_BENCHMARK and e["completed"] == 1
                        for e in snap["events"]
                    )
                    if first_item:
                        svc.cancel(record.run_id)
                        cancelled = True
                        break
                    if snap["terminal"]:
                        break
                    await asyncio.sleep(0.02)
                if not cancelled:
                    pytest.skip("mock run finished before a cancel point was observed")

                snap = await _wait_terminal(svc, record.run_id)
            assert snap["terminal_status"] == "cancelled"
            assert snap["status"] == "CANCELLED"
            view = svc.result(record.run_id)
            assert view["report"] is None  # 取消不产生 report 工件
        finally:
            svc.close()
