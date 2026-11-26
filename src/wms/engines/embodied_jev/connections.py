"""Private credential resolution and bounded HTTP for upstream DecisionPolicy.

No API request occurs while resolving or installing a connection. Public runtime
profiles contain credential references only; the resolved key lives in memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import functools
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping
from urllib.parse import urlsplit

import httpx


class PolicyCallError(RuntimeError):
    """A safe, machine-readable failure with no provider body or credential."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(f"embodied_jev: {code}")


@dataclass(frozen=True)
class ResolvedConnection:
    provider: str
    url: str
    model: str
    api_key: str = field(repr=False)
    json_mode: bool = True

    def public_config(self) -> dict[str, Any]:
        return {"provider": self.provider, "url": self.url, "model": self.model,
                "json_mode": self.json_mode, "key_configured": bool(self.api_key)}

    def _upstream_settings(self) -> dict[str, Any]:
        return {"url": self.url, "model": self.model, "key": self.api_key,
                "json_mode": self.json_mode}


def build_connection(runtime: Mapping[str, Any], provider: str,
                     profile: str | None = None, base_dir: str | Path | None = None,
                     *, environ: Mapping[str, str] | None = None) -> ResolvedConnection:
    """Resolve profiles[profile or provider] without changing runtime.

    Profiles use url/base_url, model, json_mode and either api_key_env or
    credentials_file + api_key_field. Credential files must end in .local.json
    (ignored by the platform) and contain a flat map of field names to secrets.
    Relative credential paths are anchored to the runtime file's parent.
    """
    if provider == "baseline":
        return ResolvedConnection("baseline", "", "baseline", "")
    if provider not in {"chat", "jev", "local", "claude"}:
        raise PolicyCallError("unsupported_provider")
    profiles = runtime.get("profiles", {})
    selected = profiles.get(profile or provider) if isinstance(profiles, Mapping) else None
    if not isinstance(selected, Mapping):
        raise PolicyCallError("connection_profile_missing")
    forbidden = {"key", "api_key", "token", "secret", "password", "authorization", "headers"}
    if any(str(key).lower() in forbidden for key in selected):
        raise PolicyCallError("inline_credentials_forbidden")
    if selected.get("provider", provider) != provider:
        raise PolicyCallError("connection_provider_mismatch")
    raw_url = selected.get("url", selected.get("base_url", ""))
    model = selected.get("model")
    if not isinstance(raw_url, str) or not isinstance(model, str) or not model.strip():
        raise PolicyCallError("connection_invalid")
    url = raw_url.strip().rstrip("/")
    try:
        parsed = urlsplit(url)
        valid = (parsed.scheme in {"http", "https"} and parsed.hostname
                 and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment)
        parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise PolicyCallError("connection_url_invalid")
    if provider == "chat" and not url.endswith("/chat/completions"):
        url += "/chat/completions"
    if provider == "claude" and not url.endswith("/messages"):
        url += "/messages"
    if provider == "jev" and url != "https://api.typesafe.ai/v1/systemone":
        raise PolicyCallError("jev_endpoint_invalid")
    json_mode = selected.get("json_mode", True)
    if type(json_mode) is not bool:
        raise PolicyCallError("json_mode_invalid")
    env_name = selected.get("api_key_env")
    if env_name is not None:
        if not isinstance(env_name, str) or not env_name.strip():
            raise PolicyCallError("credential_reference_invalid")
        secret = (os.environ if environ is None else environ).get(env_name, "")
    else:
        filename = selected.get("credentials_file")
        key_field = selected.get("api_key_field", profile or provider)
        if not isinstance(filename, str) or not isinstance(key_field, str) or not key_field:
            raise PolicyCallError("credential_reference_missing")
        credential_path = Path(filename)
        if not credential_path.is_absolute():
            if base_dir is None:
                raise PolicyCallError("credential_base_dir_missing")
            credential_path = Path(base_dir) / credential_path
        if not credential_path.name.endswith(".local.json"):
            raise PolicyCallError("credential_file_must_be_local_json")
        try:
            # This file is intentionally never logged, returned, or copied.
            if credential_path.stat().st_size > 65536:
                raise PolicyCallError("credential_file_too_large")
            values = json.loads(credential_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError, UnicodeError):
            raise PolicyCallError("credential_file_unreadable") from None
        secret = values.get(key_field, "") if isinstance(values, dict) else ""
    if not isinstance(secret, str) or not secret.strip():
        raise PolicyCallError("credential_missing")
    secret = secret.strip()
    if any(char in secret for char in "\r\n\0"):
        raise PolicyCallError("credential_invalid")
    # Avoid ever echoing a key accidentally pasted into an exportable field.
    if secret in url or secret in model:
        raise PolicyCallError("credential_in_public_field")
    return ResolvedConnection(provider, url, model.strip(), secret, json_mode)


def _integer(params: Mapping[str, Any], key: str, default: int, maximum: int) -> int:
    value = params.get(key, default)
    if type(value) is not int or not 0 <= value <= maximum:
        raise PolicyCallError(f"invalid_{key}")
    return value


def _number(params: Mapping[str, Any], key: str, default: float,
            minimum: float, maximum: float) -> float:
    value = params.get(key, default)
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise PolicyCallError(f"invalid_{key}")
    return float(value)


class RequestController:
    """Serial HTTP admission; retries count against the same hard request budget."""

    def __init__(self, connection: ResolvedConnection, params: Mapping[str, Any],
                 *, transport: httpx.BaseTransport | None = None,
                 cancel: threading.Event | None = None):
        self.max_api_calls = _integer(params, "max_api_calls", 0, 1000000)
        self.timeout_seconds = _number(params, "timeout_seconds", 30, .01, 300)
        self.max_429_retries = _integer(params, "max_429_retries", 1, 3)
        self.retry_backoff_seconds = _number(params, "retry_backoff_seconds", 1, 0, 10)
        self._connection = connection
        self._cancel = cancel or threading.Event()
        self._serial = threading.Lock()
        self._state_lock = threading.Lock()
        self._client = httpx.Client(transport=transport, follow_redirects=False,
                                   limits=httpx.Limits(max_connections=1, max_keepalive_connections=1))
        self._closed = False
        self._policy: Any = None
        self.api_calls: list[dict[str, Any]] = []
        self.last_error: str | None = None

    def _fail(self, code: str) -> None:
        with self._state_lock:
            self.last_error = code
        raise PolicyCallError(code) from None

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        with self._serial:
            if self._closed:
                self._fail("connection_closed")
            if url != self._connection.url:
                self._fail("connection_endpoint_changed")
            kwargs["timeout"] = self.timeout_seconds
            kwargs["follow_redirects"] = False
            if self._connection.provider == "chat":
                kwargs["json"] = {**kwargs.get("json", {}), "stream": False}
            for retry in range(self.max_429_retries + 1):
                if self._cancel.is_set():
                    self._fail("cancelled")
                with self._state_lock:
                    exhausted = len(self.api_calls) >= self.max_api_calls
                    if not exhausted:
                        row = {"attempt": len(self.api_calls) + 1, "retry": retry,
                               "input_tokens": None, "output_tokens": None, "latency_ms": None,
                               "http_status": None, "error_code": None, "model": None}
                        self.api_calls.append(row)
                if exhausted:
                    self._fail("api_budget_exhausted")
                started = time.perf_counter()
                try:
                    response = self._client.post(url, **kwargs)
                except httpx.TimeoutException:
                    row.update(latency_ms=(time.perf_counter() - started) * 1000, error_code="api_timeout")
                    self._fail("api_timeout")
                except httpx.HTTPError:
                    row.update(latency_ms=(time.perf_counter() - started) * 1000, error_code="api_transport_error")
                    self._fail("api_transport_error")
                row.update(latency_ms=(time.perf_counter() - started) * 1000, http_status=response.status_code)
                self._record_usage(row, response)
                if response.is_success:
                    with self._state_lock:
                        self.last_error = None
                    return response
                row["error_code"] = f"http_{response.status_code}"
                if response.status_code != 429 or retry >= self.max_429_retries:
                    self._fail(row["error_code"])
                # Do not wait when the next request cannot be admitted anyway.
                if len(self.api_calls) >= self.max_api_calls:
                    self._fail("api_budget_exhausted")
                delay = self.retry_backoff_seconds * (2 ** retry)
                try:
                    retry_after = float(response.headers.get("Retry-After", "0"))
                    if math.isfinite(retry_after):
                        delay = max(delay, retry_after)
                except ValueError:
                    pass
                if self._cancel.wait(min(10, max(0, delay))):
                    self._fail("cancelled")
        raise AssertionError("unreachable")

    def _record_usage(self, row: dict[str, Any], response: httpx.Response) -> None:
        try:
            body = response.json()
        except ValueError:
            return
        if not isinstance(body, dict):
            return
        usage = body.get("usage")
        if isinstance(usage, dict):
            for dest, aliases in (("input_tokens", ("prompt_tokens", "input_tokens")),
                                  ("output_tokens", ("completion_tokens", "output_tokens"))):
                value = next((usage[name] for name in aliases if name in usage), None)
                if type(value) is int and value >= 0:
                    row[dest] = value
        model = body.get("model")
        if isinstance(model, str) and 0 < len(model) <= 256:
            row["model"] = model.replace(self._connection.api_key, "[redacted]")

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            return {"max_api_calls": self.max_api_calls, "actual_http_requests": len(self.api_calls),
                    "policy_logical_calls": getattr(self._policy, "calls", 0),
                    "remaining_api_calls": max(0, self.max_api_calls - len(self.api_calls)),
                    "api_calls": copy.deepcopy(self.api_calls), "last_error": self.last_error,
                    "timeout_seconds": self.timeout_seconds, "max_429_retries": self.max_429_retries,
                    "max_concurrency": 1}

    def close(self) -> None:
        with self._serial:
            if not self._closed:
                self._client.close()
                self._closed = True


def install_policy(session: Any, connection: ResolvedConnection, params: Mapping[str, Any],
                   *, transport: httpx.BaseTransport | None = None,
                   policy_class: type | None = None) -> RequestController:
    """Replace an unstarted session's policy, keeping upstream choice validation.

    The optional transport/policy_class hooks are for offline tests. Construct the
    Session with its true provider and an empty-key placeholder connection, then
    install before start(). No global monkeypatch or environment mutation occurs.
    """
    if getattr(session, "started", None) is not None or getattr(session, "status", "idle") == "running":
        raise PolicyCallError("policy_install_requires_idle_session")
    if policy_class is None:
        from embodied_jev.policies import DecisionPolicy
        policy_class = DecisionPolicy
    controller = RequestController(connection, params, transport=transport,
                                   cancel=getattr(session, "cancel", None))

    def post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
        return controller.post(url, **kwargs)

    def close(self: Any) -> None:
        controller.close()

    def guarded(method: Any) -> Any:
        @functools.wraps(method)
        def invoke(self: Any, *args: Any, **kwargs: Any) -> Any:
            with controller._state_lock:
                requests_before = len(controller.api_calls)
            try:
                return method(self, *args, **kwargs)
            except PolicyCallError:
                raise
            except Exception:
                with controller._state_lock:
                    if len(controller.api_calls) > requests_before and controller.api_calls[-1]["error_code"] is None:
                        controller.api_calls[-1]["error_code"] = "decision_protocol_error"
                controller._fail("decision_protocol_error")
        return invoke

    methods = {"_post": post, "close": close}
    for name in ("choose", "choose_plan", "choose_channels"):
        if hasattr(policy_class, name):
            methods[name] = guarded(getattr(policy_class, name))
    controlled_class = type("BoundedDecisionPolicy", (policy_class,), methods)
    try:
        policy = controlled_class(connection.provider, connection._upstream_settings())
        previous = getattr(session, "policy", None)
        if previous is not None:
            previous.close()
        session.policy = policy
        controller._policy = policy
    except Exception:
        controller.close()
        raise PolicyCallError("policy_install_failed") from None
    return controller



