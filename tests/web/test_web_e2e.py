"""Web E2E — ``demo`` mode full flow through the FastAPI application.

Verifies the v0.5 Definition of Done (§六十三 Scenario A / §九十六): without
any real API Key the page prefills the in-process Mock endpoint, and the
create → estimate → start → SSE → result → history loop terminates with a
usable capability score.

The demo upstream is an ASGI router living inside the app, so under
``TestClient`` the runner's outbound HTTP to ``http://testserver/__mock__/v1``
is stubbed with respx (real-socket coverage happens in the manual §九十五
uvicorn E2E). The demo router itself is exercised directly through the ASGI
layer below.
"""

from __future__ import annotations

import json
from pathlib import Path

import respx
from fastapi.testclient import TestClient

from llmtrace.appdir import AppLayout
from llmtrace.web.app import create_app
from llmtrace.web.demo import DEMO_MODEL, MOCK_PREFIX
from tests.execution.conftest import TrustedFakeBackend

DEMO_KEY = "demo-key"


def _make_demo_client(tmp_path: Path) -> TestClient:
    app = create_app(
        AppLayout.resolve(tmp_path),
        demo=True,
        code_backend=TrustedFakeBackend(),
    )
    return TestClient(app)


def _demo_base_url(client: TestClient) -> str:
    return f"{str(client.base_url).rstrip('/')}{MOCK_PREFIX}"


def _demo_payload(base_url: str) -> dict[str, object]:
    return {
        "base_url": base_url,
        "model": DEMO_MODEL,
        "api_key": DEMO_KEY,
        "protocol": "openai",
        "auth_style": "auto",
        "repeat": 1,
        "timeout": 10.0,
        "check_streaming": False,
    }


def _read_sse_done(client: TestClient, run_id: str, *, timeout: float = 120.0) -> list[dict[str, object]]:
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
    raise AssertionError(f"SSE closed before done; events={len(events)}")


class TestDemoUpstream:
    """进程内 OpenAI 兼容 Mock 端点（§六十一）本身可用。"""

    def test_models_endpoint(self, tmp_path: Path) -> None:
        client = _make_demo_client(tmp_path)
        resp = client.get(f"{MOCK_PREFIX}/models")
        assert resp.status_code == 200
        models = resp.json()["data"]
        assert [m["id"] for m in models] == [DEMO_MODEL]

    def test_chat_completions_endpoint(self, tmp_path: Path) -> None:
        client = _make_demo_client(tmp_path)
        resp = client.post(
            f"{MOCK_PREFIX}/chat/completions",
            json={"model": DEMO_MODEL, "messages": [{"role": "user", "content": "q"}]},
        )
        assert resp.status_code == 200
        content = resp.json()["choices"][0]["message"]["content"]
        assert "(A)" in content and "42" in content


class TestDemoPage:
    def test_index_prefills_demo_endpoint_and_shows_demo_banner(self, tmp_path: Path) -> None:
        client = _make_demo_client(tmp_path)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "DEMO" in resp.text  # §六十二：页面必须显示 DEMO，不得冒充正式校准
        base_url = _demo_base_url(client)
        assert f'value="{base_url}"' in resp.text
        assert f'value="{DEMO_MODEL}"' in resp.text

    def test_reference_sets_empty(self, tmp_path: Path) -> None:
        client = _make_demo_client(tmp_path)
        resp = client.get("/api/reference-sets")
        assert resp.status_code == 200
        assert resp.json() == {"reference_sets": []}


class TestDemoFullFlow:
    def test_create_estimate_start_sse_result_history(self, tmp_path: Path) -> None:
        client = _make_demo_client(tmp_path)
        base_url = _demo_base_url(client)

        with respx.mock as mock:
            # 拦截 runner 对进程内 mock 的出站 HTTP（TestClient 无真实 socket）。
            mock.get(f"{base_url}/models").respond(
                status_code=200,
                json={"object": "list", "data": [{"id": DEMO_MODEL, "object": "model"}]},
            )
            mock.post(f"{base_url}/chat/completions").respond(
                status_code=200,
                json={
                    "id": "chatcmpl-demo",
                    "object": "chat.completion",
                    "created": 1677652288,
                    "model": DEMO_MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "The answer is (A). The answer is 42."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
                },
            )

            # 页面预填的值就是 demo mock 端点 —— 无任何真实 key。
            created = client.post("/api/runs", json=_demo_payload(base_url))
            assert created.status_code == 201
            run_id = created.json()["run"]["run_id"]
            assert DEMO_KEY not in json.dumps(created.json())

            est = client.post(f"/api/runs/{run_id}/estimate")
            assert est.status_code == 200
            assert est.json()["estimate"]["benchmark_requests"] == 32

            started = client.post(f"/api/runs/{run_id}/start")
            assert started.json()["status"] == "RUNNING"

            events = _read_sse_done(client, run_id)
            types = {e.get("type") for e in events}
            assert "progress" in types and "stage" in types
            assert any(e.get("type") == "status" for e in events)

            # 结果页/API：capability 可见（raw / uncalibrated）。
            view = client.get(f"/api/runs/{run_id}").json()["run"]
            assert view["status"] in ("COMPLETED", "COMPLETED_WITH_WARNINGS", "PARTIAL")
            assert view["capability_score"] is not None

            result = client.get(f"/api/runs/{run_id}/result").json()
            assert result["report"] is not None
            assert "capability_profile" in result["report"]

            history = client.get("/api/runs").json()["runs"]
            assert [r["run_id"] for r in history] == [run_id]
            assert DEMO_KEY not in json.dumps(history)

            # 页面可渲染结果视图（HTML smoke）。
            page = client.get(f"/run/{run_id}")
            assert page.status_code == 200
            assert f'data-run-id="{run_id}"' in page.text
