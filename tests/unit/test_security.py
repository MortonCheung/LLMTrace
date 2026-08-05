"""Unit tests for the LLMTrace security redaction module."""

from __future__ import annotations

import os
from unittest.mock import patch

from llmtrace.security.redaction import (
    check_api_key,
    redact_headers,
    redact_json_body,
    redact_url,
    sanitize_for_html,
)

# --------------------------------------------------------------------------- #
# redact_headers
# --------------------------------------------------------------------------- #


class TestRedactHeaders:
    """Tests for redact_headers()."""

    def test_authorization_header_is_redacted(self):
        headers = {"Authorization": "Bearer sk-1234567890abcdefghij"}
        result = redact_headers(headers)
        assert result["Authorization"] == "Bear***ghij"

    def test_authorization_header_short_value_is_fully_redacted(self):
        headers = {"Authorization": "abc123"}
        result = redact_headers(headers)
        assert result["Authorization"] == "[REDACTED]"

    def test_x_api_key_header_is_redacted(self):
        headers = {"x-api-key": "sk-abcdefghijklmnopqrst"}
        result = redact_headers(headers)
        # "sk-abcdefghijklmnopqrst" -> length 22, first 4 = "sk-a", last 4 = "qrst"
        assert result["x-api-key"] == "sk-a***qrst"

    def test_api_key_header_is_redacted(self):
        headers = {"api-key": "my-secret-api-key-value"}
        result = redact_headers(headers)
        assert result["api-key"] == "my-s***alue"

    def test_cookie_header_is_redacted(self):
        headers = {"Cookie": "session_id=abcdef1234567890abcd"}
        result = redact_headers(headers)
        assert result["Cookie"] == "sess***abcd"

    def test_set_cookie_header_is_redacted(self):
        headers = {"Set-Cookie": "session=xyz1234567890; HttpOnly"}
        result = redact_headers(headers)
        assert result["Set-Cookie"] == "sess***Only"

    def test_normal_headers_are_not_redacted(self):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "test-agent",
        }
        result = redact_headers(headers)
        assert result["Content-Type"] == "application/json"
        assert result["Accept"] == "application/json"
        assert result["User-Agent"] == "test-agent"

    def test_case_insensitive_header_matching(self):
        headers = {
            "AUTHORIZATION": "Bearer secret-token-12345",
            "X-API-KEY": "key-abcdefghijklmnop",
            "API-KEY": "another-key-value-here",
            "COOKIE": "sid=abcdef1234567890",
            "SET-COOKIE": "token=xyz1234567890",
        }
        result = redact_headers(headers)
        assert result["AUTHORIZATION"] == "Bear***2345"
        assert result["X-API-KEY"] == "key-***mnop"
        assert result["API-KEY"] == "anot***here"
        assert result["COOKIE"] == "sid=***7890"
        assert result["SET-COOKIE"] == "toke***7890"

    def test_mixed_sensitive_and_normal_headers(self):
        headers = {
            "Authorization": "Bearer token12345",
            "Content-Type": "application/json",
            "x-api-key": "sk-1234567890abcdef",
            "Accept": "text/html",
        }
        result = redact_headers(headers)
        assert result["Authorization"] == "Bear***2345"
        assert result["Content-Type"] == "application/json"
        assert result["x-api-key"] == "sk-1***cdef"
        assert result["Accept"] == "text/html"

    def test_empty_headers(self):
        result = redact_headers({})
        assert result == {}

    def test_original_headers_not_mutated(self):
        headers = {"Authorization": "Bearer original-token"}
        original = dict(headers)
        redact_headers(headers)
        assert headers == original


# --------------------------------------------------------------------------- #
# redact_url
# --------------------------------------------------------------------------- #


class TestRedactUrl:
    """Tests for redact_url()."""

    def test_url_with_token_param_is_redacted(self):
        url = "https://api.example.com/v1/chat?token=sk-abc123&model=gpt-4"
        result = redact_url(url)
        assert "token=[REDACTED]" in result
        assert "sk-abc123" not in result
        assert "model=gpt-4" in result

    def test_url_with_key_param_is_redacted(self):
        url = "https://api.example.com/v1/endpoint?key=my-secret-key&limit=10"
        result = redact_url(url)
        assert "key=[REDACTED]" in result
        assert "my-secret-key" not in result
        assert "limit=10" in result

    def test_url_with_secret_param_is_redacted(self):
        url = "https://api.example.com/v1/auth?secret=supersecret&user=admin"
        result = redact_url(url)
        assert "secret=[REDACTED]" in result
        assert "supersecret" not in result
        assert "user=admin" in result

    def test_url_with_signature_param_is_redacted(self):
        url = "https://api.example.com/v1/webhook?signature=abc123def456"
        result = redact_url(url)
        assert "signature=[REDACTED]" in result
        assert "abc123def456" not in result

    def test_url_with_api_key_param_is_redacted(self):
        url = "https://api.example.com/v1/data?api_key=sk-mykey&format=json"
        result = redact_url(url)
        assert "api_key=[REDACTED]" in result
        assert "sk-mykey" not in result
        assert "format=json" in result

    def test_url_with_apikey_param_is_redacted(self):
        url = "https://api.example.com/v1/data?apikey=sk-mykey&format=json"
        result = redact_url(url)
        assert "apikey=[REDACTED]" in result
        assert "sk-mykey" not in result

    def test_normal_params_are_not_redacted(self):
        url = "https://api.example.com/v1/search?q=hello&page=1&limit=10"
        result = redact_url(url)
        assert result == url

    def test_url_without_query_params_is_unchanged(self):
        url = "https://api.example.com/v1/health"
        result = redact_url(url)
        assert result == url

    def test_url_without_query_params_but_with_fragment(self):
        url = "https://api.example.com/v1/docs#section"
        result = redact_url(url)
        assert result == url

    def test_case_insensitive_query_param_matching(self):
        url = "https://api.example.com/v1?TOKEN=secret123&Key=value&SECRET=hidden"
        result = redact_url(url)
        assert "TOKEN=[REDACTED]" in result
        assert "Key=[REDACTED]" in result
        assert "SECRET=[REDACTED]" in result
        assert "secret123" not in result
        assert "hidden" not in result

    def test_mixed_sensitive_and_normal_params(self):
        url = "https://api.example.com/v1/chat?model=gpt-4&token=sk-abc&user=me"
        result = redact_url(url)
        assert "token=[REDACTED]" in result
        assert "model=gpt-4" in result
        assert "user=me" in result
        assert "sk-abc" not in result


# --------------------------------------------------------------------------- #
# redact_json_body
# --------------------------------------------------------------------------- #


class TestRedactJsonBody:
    """Tests for redact_json_body()."""

    def test_api_key_field_is_redacted(self):
        body = {"api_key": "sk-1234567890abcdef", "model": "gpt-4"}
        result = redact_json_body(body)
        assert result["api_key"] == "[REDACTED]"
        assert result["model"] == "gpt-4"

    def test_apikey_field_is_redacted(self):
        body = {"apikey": "sk-my-secret-key", "user": "admin"}
        result = redact_json_body(body)
        assert result["apikey"] == "[REDACTED]"
        assert result["user"] == "admin"

    def test_token_field_is_redacted(self):
        body = {"token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", "data": "payload"}
        result = redact_json_body(body)
        assert result["token"] == "[REDACTED]"
        assert result["data"] == "payload"

    def test_secret_field_is_redacted(self):
        body = {"secret": "my-hidden-secret-value", "public": "info"}
        result = redact_json_body(body)
        assert result["secret"] == "[REDACTED]"
        assert result["public"] == "info"

    def test_password_field_is_redacted(self):
        body = {"username": "admin", "password": "super-secret-pw"}
        result = redact_json_body(body)
        assert result["password"] == "[REDACTED]"
        assert result["username"] == "admin"

    def test_key_field_is_redacted(self):
        body = {"key": "sk-my-api-key", "value": "some-data"}
        result = redact_json_body(body)
        assert result["key"] == "[REDACTED]"
        assert result["value"] == "some-data"

    def test_nested_json_objects_are_recursively_redacted(self):
        body = {
            "request": {
                "api_key": "sk-nested-key",
                "payload": {
                    "token": "nested-token",
                    "message": "hello",
                },
            },
            "user": "admin",
        }
        result = redact_json_body(body)
        assert result["request"]["api_key"] == "[REDACTED]"
        assert result["request"]["payload"]["token"] == "[REDACTED]"
        assert result["request"]["payload"]["message"] == "hello"
        assert result["user"] == "admin"

    def test_lists_containing_dicts_with_sensitive_keys_are_redacted(self):
        body = {
            "messages": [
                {"role": "user", "token": "tok-1"},
                {"role": "assistant", "api_key": "sk-2"},
                {"role": "system", "content": "plain"},
            ],
        }
        result = redact_json_body(body)
        assert result["messages"][0]["token"] == "[REDACTED]"
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["api_key"] == "[REDACTED]"
        assert result["messages"][1]["role"] == "assistant"
        assert result["messages"][2]["role"] == "system"
        assert result["messages"][2]["content"] == "plain"

    def test_deeply_nested_lists_are_redacted(self):
        body = {
            "data": [
                [
                    {"secret": "deep-secret", "value": 42},
                    {"value": 99},
                ],
            ],
        }
        result = redact_json_body(body)
        assert result["data"][0][0]["secret"] == "[REDACTED]"
        assert result["data"][0][0]["value"] == 42
        assert result["data"][0][1]["value"] == 99

    def test_normal_fields_are_not_redacted(self):
        body = {
            "name": "test",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": True,
            "messages": ["hello", "world"],
        }
        result = redact_json_body(body)
        assert result == body

    def test_none_input_returns_none(self):
        result = redact_json_body(None)
        assert result is None

    def test_empty_dict(self):
        result = redact_json_body({})
        assert result == {}

    def test_case_insensitive_sensitive_key_matching(self):
        body = {"API_KEY": "sk-upper", "Token": "tok-upper", "SECRET": "sec-upper"}
        result = redact_json_body(body)
        assert result["API_KEY"] == "[REDACTED]"
        assert result["Token"] == "[REDACTED]"
        assert result["SECRET"] == "[REDACTED]"

    def test_original_body_not_mutated(self):
        body = {"api_key": "sk-original", "data": {"token": "inner-token"}}
        original = {"api_key": "sk-original", "data": {"token": "inner-token"}}
        redact_json_body(body)
        assert body == original


# --------------------------------------------------------------------------- #
# check_api_key
# --------------------------------------------------------------------------- #


class TestCheckApiKey:
    """Tests for check_api_key()."""

    def test_returns_value_when_env_var_exists(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key-12345"}, clear=True):
            result = check_api_key("OPENAI_API_KEY")
            assert result == "sk-test-key-12345"

    def test_returns_none_when_env_var_does_not_exist(self):
        with patch.dict(os.environ, {}, clear=True):
            result = check_api_key("NONEXISTENT_VAR")
            assert result is None

    def test_returns_none_when_env_var_is_empty_string(self):
        with patch.dict(os.environ, {"EMPTY_VAR": ""}, clear=True):
            result = check_api_key("EMPTY_VAR")
            assert result is None

    def test_returns_value_when_env_var_is_whitespace_only(self):
        with patch.dict(os.environ, {"WHITESPACE_VAR": "   "}, clear=True):
            result = check_api_key("WHITESPACE_VAR")
            assert result == "   "


# --------------------------------------------------------------------------- #
# sanitize_for_html
# --------------------------------------------------------------------------- #


class TestSanitizeForHtml:
    """Tests for sanitize_for_html()."""

    def test_lt_and_gt_are_escaped(self):
        result = sanitize_for_html("<script>alert('xss')</script>")
        assert "<" not in result
        assert ">" not in result
        assert "&lt;" in result
        assert "&gt;" in result

    def test_ampersand_is_escaped(self):
        result = sanitize_for_html("a & b")
        assert "&amp;" in result
        # Make sure the raw & is gone (except the ones we intentionally added)
        assert result == "a &amp; b"

    def test_double_quote_is_escaped(self):
        result = sanitize_for_html('hello "world"')
        assert "&quot;" in result
        assert '"' not in result

    def test_single_quote_is_escaped(self):
        result = sanitize_for_html("it's a test")
        assert "&#x27;" in result
        assert "'" not in result

    def test_normal_text_is_unchanged(self):
        text = "Hello, this is normal text without any HTML special characters."
        result = sanitize_for_html(text)
        assert result == text

    def test_empty_string(self):
        result = sanitize_for_html("")
        assert result == ""

    def test_multiple_special_characters(self):
        result = sanitize_for_html('<a href="http://example.com?a=1&b=2">link</a>')
        assert "&lt;" in result
        assert "&gt;" in result
        assert "&quot;" in result
        assert "&amp;" in result
        assert "<" not in result
        assert ">" not in result
        assert '"' not in result
        assert result.count("&amp;") >= 1  # at minimum the & in the href

    def test_ampersand_does_not_double_escape(self):
        result = sanitize_for_html("&")
        assert result == "&amp;"
        # Running it again should not change already-escaped chars
        # because the introduced & triggers re-escape
        result2 = sanitize_for_html(result)
        assert result2 == "&amp;amp;"
