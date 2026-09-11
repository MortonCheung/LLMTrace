"""Web API vertical tests — create / estimate / start / SSE / result / history /
cancel through the FastAPI layer against a mocked OpenAI endpoint.

Covers §七十–§七十二: every route must map HTTP semantics correctly, service
errors render as ``{"error": {...}}`` JSON, and the API key never crosses the
wire into responses or disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import respx
from fastapi.testclient import TestClient

from llmtrace.appdir import AppLayout
from llmtrace.web.app import create_app
from tests.execution.conftest import TrustedFakeBackend

API_KEY = "sk-super-secret-123"
BASE_URL = "http://test.example.com/v1"
MODEL = "my-real-model"

# Base URL 保留凭据，用于验证 redact 边界。
_BASE_URL_WITH_CREDS = "https://user:sekret-pass@test.example.com/v1"


def _completion_json(content: str = "The answer is (A). The answer is 42.") -> dict[str, object]:
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
    }


def _mock_openai(respx_mock: respx.MockRouter, *, base_url: str = BASE_URL) -> None:
    respx_mock.get(f"{base_url}/models").respond(
        status_code=200, json={"object": "list", "data": [{"id": MODEL, "object": "model"}]}
    )
    respx_mock.post(f"{base_url}/chat/completions").respond(
        status_code=200,
        json=_completion_json(),
        headers={"content-type": "application/json"},
    )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = create_app(
        AppLayout.resolve(tmp_path),
        code_backend=TrustedFakeBackend(),
    )
    return TestClient(app)


def _run_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "base_url": BASE_URL,
        "model": MODEL,
        "api_key": API_KEY,
        "protocol": "openai",
        "auth_style": "auto",
        "repeat": 1,
        "timeout": 10.0,
        "check_streaming": False,
    }
    payload.update(overrides)
    return payload


def _read_sse_done(client: TestClient, run_id: str, *, timeout: float = 90.0) -> list[dict[str, object]]:
    """消费 /api/runs/{id}/events 直到 ``done`` 事件，返回全部事件。"""
    events: list[dict[str, object]] = []
    with client.stream("GET", f"/api/runs/{run_id}/events") as stream:
        for line in stream.iter_lines():
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[len("data:") :].strip())
            except json.JSONDecodeError:
                continue
            events.append(event)
            if event.get("done"):
                return events
    pytest.fail(f"SSE closed before done event; got {len(events)} events")
    return events


class TestPages:
    def test_index_page_renders_form(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "New Audit" in resp.text
        assert 'id="audit-form"' in resp.text
        assert "DEMO" not in resp.text  # 非 demo 不出现 DEMO banner

    def test_history_page_empty(self, client: TestClient) -> None:
        resp = client.get("/history")
        assert resp.status_code == 200

    def test_static_assets_served(self, client: TestClient) -> None:
        css = client.get("/static/app.css")
        assert css.status_code == 200 and "text/css" in css.headers["content-type"]
        js = client.get("/static/app.js")
        assert js.status_code == 200 and "javascript" in js.headers["content-type"]

    def test_healthz(self, client: TestClient) -> None:
        assert client.get("/healthz").json() == {"status": "ok"}

    def test_run_page_missing_run_is_404(self, client: TestClient) -> None:
        resp = client.get("/run/does-not-exist")
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "RUN_NOT_FOUND"


class TestErrorSemantics:
    def test_create_rejects_empty_key(self, client: TestClient) -> None:
        resp = client.post("/api/runs", json=_run_payload(api_key="   "))
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["code"] == "RUN_CREATE_INVALID"
        assert API_KEY not in json.dumps(body)

    def test_estimate_missing_run_404(self, client: TestClient) -> None:
        resp = client.post("/api/runs/nope/estimate")
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "RUN_NOT_FOUND"

    def test_get_missing_run_404(self, client: TestClient) -> None:
        resp = client.get("/api/runs/nope")
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "RUN_NOT_FOUND"

    def test_cancel_missing_run_404(self, client: TestClient) -> None:
        resp = client.post("/api/runs/nope/cancel")
        assert resp.status_code == 404

    def test_events_missing_run_404_json_not_sse(self, client: TestClient) -> None:
        resp = client.get("/api/runs/nope/events")
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "RUN_NOT_FOUND"

    def test_double_start_conflict(self, client: TestClient, tmp_path: Path) -> None:
        with respx.mock as mock:
            _mock_openai(mock)
            created = client.post("/api/runs", json=_run_payload()).json()["run"]
            run_id = created["run_id"]
            assert client.post(f"/api/runs/{run_id}/start").status_code == 200
            resp = client.post(f"/api/runs/{run_id}/start")
            assert resp.status_code == 409
            assert resp.json()["error"]["code"] == "RUN_STATE_CONFLICT"
            # 等待后台跑完，避免 fixture 关闭时任务仍在写 DB。
            events = _read_sse_done(client, run_id)
            assert any(e.get("type") == "status" for e in events)


class TestRunLifecycle:
    def test_full_flow_via_api(self, client: TestClient, tmp_path: Path) -> None:
        with respx.mock as mock:
            _mock_openai(mock)
            # create：201 且响应绝不含 key，base_url 已脱敏
            resp = client.post("/api/runs", json=_run_payload())
            assert resp.status_code == 201
            run = resp.json()["run"]
            run_id = run["run_id"]
            assert run["status"] == "PENDING"
            assert API_KEY not in json.dumps(run)
            assert "sekret" not in json.dumps(run)

            # estimate：纯脱机，32 benchmark items
            est = client.post(f"/api/runs/{run_id}/estimate")
            assert est.status_code == 200
            assert est.json()["estimate"]["benchmark_requests"] == 32

            # start → SSE 完整事件流 → done
            assert client.post(f"/api/runs/{run_id}/start").json()["status"] == "RUNNING"
            events = _read_sse_done(client, run_id)
            types = {e.get("type") for e in events}
            assert "progress" in types
            assert "stage" in types
            assert any(e.get("type") == "status" for e in events)

            # 终态 view / result / history
            view = client.get(f"/api/runs/{run_id}").json()["run"]
            assert view["status"] in ("COMPLETED", "COMPLETED_WITH_WARNINGS", "PARTIAL")
            assert API_KEY not in json.dumps(view)

            result = client.get(f"/api/runs/{run_id}/result").json()
            assert result["run"]["run_id"] == run_id
            assert result["report"] is not None  # report.json 工件已生成

            history = client.get("/api/runs").json()["runs"]
            assert [r["run_id"] for r in history] == [run_id]
            assert API_KEY not in json.dumps(history)

    def test_reference_sets_empty_by_default(self, client: TestClient) -> None:
        resp = client.get("/api/reference-sets")
        assert resp.status_code == 200
        assert resp.json() == {"reference_sets": []}

    def test_cancel_terminal_run_is_idempotent(self, client: TestClient) -> None:
        """对终态 run 的 cancel 不应抛错，返回当前状态（幂等语义）。"""
        with respx.mock as mock:
            _mock_openai(mock)
            created = client.post("/api/runs", json=_run_payload()).json()["run"]
            run_id = created["run_id"]
            client.post(f"/api/runs/{run_id}/start")
            _read_sse_done(client, run_id)
            resp = client.post(f"/api/runs/{run_id}/cancel")
            assert resp.status_code == 200
            assert resp.json()["status"] in ("COMPLETED", "COMPLETED_WITH_WARNINGS", "PARTIAL")


class TestCredentialsRedaction:
    def test_credentials_in_base_url_never_returned(self, client: TestClient) -> None:
        resp = client.post("/api/runs", json=_run_payload(base_url=_BASE_URL_WITH_CREDS))
        assert resp.status_code == 201
        blob = json.dumps(resp.json())
        assert "sekret-pass" not in blob
        assert "user:sekret-pass" not in blob
        run = resp.json()["run"]
        assert "test.example.com" in run["base_url_redacted"]
        assert "sekret" not in run["base_url_redacted"]
