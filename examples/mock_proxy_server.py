#!/usr/bin/env python3
"""LLMTrace Mock Proxy Server.

使用 Python 标准库实现的本地模拟中转站服务器。
支持三种模式: honest, fallback, inconsistent
同时支持 OpenAI-compatible 和 Anthropic-compatible 协议。

用法:
    python examples/mock_proxy_server.py --mode honest --port 8080
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

MODE = "honest"
PORT = 8080
DEFAULT_MODEL = "mock-model-v1"
ALT_MODEL = "mock-model-v2"

# 线程安全的请求日志：记录每次实际收到的审计请求
REQUEST_LOG: list[dict[str, str]] = []
LOG_LOCK = threading.Lock()
# 只对"正常模型"请求递增的计数器，用于 inconsistent 模式轮换
NORMAL_COUNTER = 0


def reset_state() -> None:
    """清空请求日志和计数器（每次测试启动新服务器时调用）."""
    global NORMAL_COUNTER
    with LOG_LOCK:
        REQUEST_LOG.clear()
        NORMAL_COUNTER = 0


def _log_request(method: str, path: str) -> None:
    """记录一次审计请求（线程安全）."""
    with LOG_LOCK:
        REQUEST_LOG.append({"method": method, "path": path})


def get_request_log() -> list[dict[str, str]]:
    """返回请求日志的副本."""
    with LOG_LOCK:
        return list(REQUEST_LOG)


def generate_openai_response(
    model: str, content: str, include_usage: bool = True, include_id: bool = True
) -> dict[str, Any]:
    """生成 OpenAI-compatible 响应."""
    resp: dict[str, Any] = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "created": int(time.time()),
        "model": model,
        "object": "chat.completion",
    }
    if include_id:
        resp["id"] = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    if include_usage:
        resp["usage"] = {
            "prompt_tokens": len(content) // 2,
            "completion_tokens": len(content) // 2,
            "total_tokens": len(content),
        }
    return resp


def generate_openai_stream_events(
    model: str, content: str, include_usage: bool = True, include_id: bool = True
) -> list[str]:
    """生成 OpenAI-compatible 流式 SSE 事件."""
    rid = f"chatcmpl-{uuid.uuid4().hex[:24]}" if include_id else ""
    events = []

    # 第一个 delta
    events.append(
        "data: "
        + json.dumps(
            {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            }
        )
        + "\n\n"
    )

    # 内容 delta
    for _i, char in enumerate(content):
        events.append(
            "data: "
            + json.dumps(
                {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": char}, "finish_reason": None}],
                }
            )
            + "\n\n"
        )

    # 结束事件
    finish_data: dict[str, Any] = {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    if include_usage:
        finish_data["usage"] = {
            "prompt_tokens": len(content) // 2,
            "completion_tokens": len(content) // 2,
            "total_tokens": len(content),
        }
    events.append(f"data: {json.dumps(finish_data)}\n\n")
    events.append("data: [DONE]\n\n")
    return events


def generate_anthropic_response(
    model: str, content: str, include_usage: bool = True, include_id: bool = True
) -> dict[str, Any]:
    """生成 Anthropic-compatible 响应."""
    resp: dict[str, Any] = {
        "id": f"msg_{uuid.uuid4().hex[:24]}" if include_id else "",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
    }
    if include_usage:
        resp["usage"] = {
            "input_tokens": len(content) // 2,
            "output_tokens": len(content) // 2,
        }
    return resp


def generate_anthropic_stream_events(
    model: str, content: str, include_usage: bool = True, include_id: bool = True
) -> list[str]:
    """生成 Anthropic-compatible 流式 SSE 事件."""
    rid = f"msg_{uuid.uuid4().hex[:24]}" if include_id else ""
    events = []

    # message_start
    events.append(
        "data: "
        + json.dumps(
            {
                "type": "message_start",
                "message": {
                    "id": rid,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }
        )
        + "\n\n"
    )

    # content_block_start
    events.append(
        "data: "
        + json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
        + "\n\n"
    )

    # content_block_delta
    for part in content.split():
        events.append(
            "data: "
            + json.dumps(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": part + " "},
                }
            )
            + "\n\n"
        )

    # content_block_stop
    events.append(f"data: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n")

    # message_delta
    delta: dict[str, Any] = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
    }
    if include_usage:
        delta["usage"] = {"output_tokens": len(content) // 2}
    events.append(f"data: {json.dumps(delta)}\n\n")

    # message_stop
    events.append(f"data: {json.dumps({'type': 'message_stop'})}\n\n")
    return events


def get_model_list() -> list[dict[str, Any]]:
    """生成模型列表."""
    return [
        {"id": DEFAULT_MODEL, "object": "model", "created": int(time.time())},
        {"id": ALT_MODEL, "object": "model", "created": int(time.time())},
    ]


class MockProxyHandler(BaseHTTPRequestHandler):
    """Mock 代理服务器处理器."""

    def log_message(self, format: str, *args: Any) -> None:
        """自定义日志格式."""
        sys.stderr.write(f"[MockProxy] {self.address_string()} - {format % args}\n")

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        """发送 JSON 响应."""
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse_events(self, events: list[str]) -> None:
        """发送 SSE 事件."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for event in events:
            self.wfile.write(event.encode("utf-8"))
            self.wfile.flush()

    def _determine_response_model(self, requested_model: str) -> tuple[str, bool]:
        """根据模式确定响应模型和拒绝状态.

        返回 (响应模型, 是否拒绝):
        - honest:      正常 -> DEFAULT_MODEL; 无效模型 -> 拒绝 (404)
        - fallback:    任何模型 -> DEFAULT_MODEL (无效模型也成功返回)
        - inconsistent:正常 -> DEFAULT/ALT 轮换; 无效模型 -> 拒绝 (404)
        """
        global NORMAL_COUNTER

        is_invalid = requested_model.startswith("llmtrace-invalid-")

        if MODE == "honest":
            if is_invalid:
                return "", True
            return DEFAULT_MODEL, False

        elif MODE == "fallback":
            # 无论什么模型都返回默认模型
            return DEFAULT_MODEL, False

        elif MODE == "inconsistent":
            if is_invalid:
                return "", True
            # 只对正常模型请求轮换，保证基线漂移可复现
            NORMAL_COUNTER += 1
            if NORMAL_COUNTER % 3 == 0:
                return ALT_MODEL, False
            return DEFAULT_MODEL, False

        return DEFAULT_MODEL, False

    def _should_include_usage(self) -> bool:
        """决定是否包含 usage 字段."""
        if MODE == "inconsistent":
            return NORMAL_COUNTER % 2 == 0
        return True

    def _should_include_id(self) -> bool:
        """决定是否包含 ID 字段."""
        return True

    def _handle_models(self) -> None:
        """处理模型列表请求."""
        self._send_json({"object": "list", "data": get_model_list()})

    def _handle_openai_completion(self) -> None:
        """处理 OpenAI-compatible 补全请求."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))

        requested_model = body.get("model", "")
        is_stream = body.get("stream", False)
        messages = body.get("messages", [])
        user_content = messages[-1].get("content", "") if messages else ""

        response_model, should_reject = self._determine_response_model(requested_model)

        if should_reject:
            self._send_json(
                {
                    "error": {
                        "message": f"The model `{requested_model}` does not exist",
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                    }
                },
                status=404,
            )
            return

        content = f"Response to: {user_content}"

        if is_stream:
            include_usage = self._should_include_usage()
            include_id = self._should_include_id()
            events = generate_openai_stream_events(response_model, content, include_usage, include_id)
            self._send_sse_events(events)
        else:
            include_usage = self._should_include_usage()
            include_id = self._should_include_id()
            resp = generate_openai_response(response_model, content, include_usage, include_id)
            self._send_json(resp)

    def _handle_anthropic_messages(self) -> None:
        """处理 Anthropic-compatible 消息请求."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))

        requested_model = body.get("model", "")
        is_stream = body.get("stream", False)
        messages = body.get("messages", [])
        user_content = messages[-1].get("content", "") if messages else ""

        response_model, should_reject = self._determine_response_model(requested_model)

        if should_reject:
            self._send_json(
                {
                    "error": {
                        "type": "not_found_error",
                        "message": f"model: {requested_model}",
                    }
                },
                status=404,
            )
            return

        content = f"Response to: {user_content}"

        if is_stream:
            include_usage = self._should_include_usage()
            include_id = self._should_include_id()
            events = generate_anthropic_stream_events(response_model, content, include_usage, include_id)
            self._send_sse_events(events)
        else:
            include_usage = self._should_include_usage()
            include_id = self._should_include_id()
            resp = generate_anthropic_response(response_model, content, include_usage, include_id)
            self._send_json(resp)

    def _detect_protocol(self) -> str:
        """检测协议类型."""
        path = urlparse(self.path).path
        if path.endswith("/chat/completions"):
            return "openai"
        elif path.endswith("/messages"):
            return "anthropic"
        return "unknown"

    def do_GET(self) -> None:
        """处理 GET 请求."""
        parsed = urlparse(self.path)
        # 调试端点：返回实际接收到的审计请求日志（本身不计入请求数）
        if parsed.path == "/debug/requests":
            self._send_json({"count": len(get_request_log()), "requests": get_request_log()})
            return
        _log_request("GET", parsed.path)
        if parsed.path in ("/v1/models", "/models"):
            self._handle_models()
        else:
            self._send_json({"error": "Not Found"}, status=404)

    def do_POST(self) -> None:
        """处理 POST 请求."""
        parsed = urlparse(self.path)
        _log_request("POST", parsed.path)
        protocol = self._detect_protocol()
        if protocol == "openai":
            self._handle_openai_completion()
        elif protocol == "anthropic":
            self._handle_anthropic_messages()
        else:
            self._send_json({"error": "Not Found"}, status=404)


def main() -> None:
    """入口函数."""
    parser = argparse.ArgumentParser(description="LLMTrace Mock Proxy Server")
    parser.add_argument(
        "--mode",
        choices=["honest", "fallback", "inconsistent"],
        default="honest",
        help="服务器模式 (默认: honest)",
    )
    parser.add_argument("--port", type=int, default=8080, help="监听端口 (默认: 8080)")
    args = parser.parse_args()

    global MODE, PORT
    MODE = args.mode
    PORT = args.port

    reset_state()

    server = ThreadingHTTPServer(("127.0.0.1", PORT), MockProxyHandler)
    print(f"Mock Proxy Server 启动: http://127.0.0.1:{PORT}")
    print(f"模式: {MODE}")
    print("支持端点:")
    print(f"  GET  http://127.0.0.1:{PORT}/v1/models")
    print(f"  POST http://127.0.0.1:{PORT}/v1/chat/completions  (OpenAI-compatible)")
    print(f"  POST http://127.0.0.1:{PORT}/v1/messages  (Anthropic-compatible)")
    print(f"  GET  http://127.0.0.1:{PORT}/debug/requests  (请求日志)")
    print()
    print("按 Ctrl+C 停止服务器")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")
        server.server_close()


if __name__ == "__main__":
    main()
