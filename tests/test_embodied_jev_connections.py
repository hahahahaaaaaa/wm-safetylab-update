"""Offline contracts; no real credentials or remote API requests."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from wms.engines.embodied_jev.connections import (
    PolicyCallError, RequestController, ResolvedConnection, build_connection, install_policy,
)


@pytest.fixture
def connection():
    return ResolvedConnection("chat", "https://model.invalid/v1/chat/completions", "test-model", "test-secret-only")


def runtime(profile):
    return {"profiles": {"chat": {"base_url": "https://model.invalid/v1/", "model": "test-model", **profile}}}


def body(choice="move", usage=True):
    result = {"model": "returned-model", "choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps({"choice": choice})}}]}
    if usage:
        result["usage"] = {"prompt_tokens": 12, "completion_tokens": 3}
    return result


def test_env_reference_does_not_mutate_or_expose_secret():
    config = runtime({"api_key_env": "WMS_TEST_KEY"})
    before = copy.deepcopy(config)
    connection = build_connection(config, "chat", environ={"WMS_TEST_KEY": "test-secret-only"})
    assert config == before
    assert connection.url == "https://model.invalid/v1/chat/completions"
    assert "test-secret-only" not in repr(connection)
    assert "test-secret-only" not in json.dumps(connection.public_config())


def test_flat_ignored_file_reference_and_named_profile(tmp_path):
    path = tmp_path / "credentials.local.json"
    path.write_text(json.dumps({"chat": "test-secret-only"}), encoding="utf-8")
    config = {"profiles": {"alternate": {"provider": "chat", "url": "https://model.invalid/v1/chat/completions",
              "model": "test-model", "credentials_file": path.name, "api_key_field": "chat"}}}
    connection = build_connection(config, "chat", profile="alternate", base_dir=tmp_path)
    assert connection.api_key == "test-secret-only"
    assert connection.url.endswith("/chat/completions")


@pytest.mark.parametrize("profile,code", [
    ({"api_key": "do-not-echo"}, "inline_credentials_forbidden"),
    ({"api_key_env": "WMS_TEST_KEY"}, "credential_missing"),
    ({"credentials_file": "credentials.json"}, "credential_file_must_be_local_json"),
    ({"credentials_file": "missing.local.json"}, "credential_file_unreadable"),
    ({"api_key_env": "WMS_TEST_KEY", "url": "https://user:secret@model.invalid"}, "connection_url_invalid"),
])
def test_invalid_credentials_are_safe(tmp_path, profile, code):
    with pytest.raises(PolicyCallError) as caught:
        build_connection(runtime(profile), "chat", base_dir=tmp_path, environ={})
    assert caught.value.code == code
    assert "do-not-echo" not in str(caught.value)


def test_missing_explicit_env_does_not_read_file(tmp_path):
    config = runtime({"api_key_env": "WMS_TEST_KEY", "credentials_file": "unknown.local.json"})
    with pytest.raises(PolicyCallError, match="credential_missing"):
        build_connection(config, "chat", base_dir=tmp_path, environ={})


def test_zero_budget_sends_nothing(connection):
    sent = []
    controller = RequestController(connection, {}, transport=httpx.MockTransport(lambda request: sent.append(request)))
    with pytest.raises(PolicyCallError, match="api_budget_exhausted"):
        controller.post(connection.url, json={})
    assert sent == []
    assert controller.snapshot()["actual_http_requests"] == 0
    controller.close()


def test_429_retry_records_each_attempt_and_forces_stream_false(connection):
    requests = []
    def reply(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "ignored"})
        return httpx.Response(200, json=body())
    controller = RequestController(connection, {"max_api_calls": 2, "timeout_seconds": 7,
        "max_429_retries": 1, "retry_backoff_seconds": 0}, transport=httpx.MockTransport(reply))
    response = controller.post(connection.url, json={"stream": True}, timeout=999)
    assert response.status_code == 200
    assert all(json.loads(request.content)["stream"] is False for request in requests)
    assert requests[0].extensions["timeout"]["read"] == 7
    state = controller.snapshot()
    assert state["actual_http_requests"] == 2
    assert state["api_calls"][0]["error_code"] == "http_429"
    assert state["api_calls"][0]["input_tokens"] is None
    assert state["api_calls"][1]["input_tokens"] == 12
    assert state["api_calls"][1]["output_tokens"] == 3
    assert state["api_calls"][1]["model"] == "returned-model"
    with pytest.raises(PolicyCallError, match="api_budget_exhausted"):
        controller.post(connection.url, json={})
    controller.close()


def test_retry_never_exceeds_request_budget(connection):
    controller = RequestController(connection, {"max_api_calls": 1, "max_429_retries": 3},
        transport=httpx.MockTransport(lambda request: httpx.Response(429)))
    with pytest.raises(PolicyCallError, match="api_budget_exhausted"):
        controller.post(connection.url, json={})
    assert controller.snapshot()["actual_http_requests"] == 1
    controller.close()


def test_429_retries_are_finite(connection):
    controller = RequestController(connection, {"max_api_calls": 20, "max_429_retries": 1,
        "retry_backoff_seconds": 0}, transport=httpx.MockTransport(lambda request: httpx.Response(429)))
    with pytest.raises(PolicyCallError, match="http_429"):
        controller.post(connection.url, json={})
    assert controller.snapshot()["actual_http_requests"] == 2
    controller.close()


def test_cancellation_blocks_retry(connection):
    cancel = threading.Event()
    def reply(request):
        cancel.set()
        return httpx.Response(429, headers={"Retry-After": "10"})
    controller = RequestController(connection, {"max_api_calls": 2}, cancel=cancel,
                                   transport=httpx.MockTransport(reply))
    with pytest.raises(PolicyCallError, match="cancelled"):
        controller.post(connection.url, json={})
    assert controller.snapshot()["actual_http_requests"] == 1
    controller.close()


def test_timeout_is_not_retried_or_disclosed(connection):
    def reply(request):
        raise httpx.ReadTimeout("private provider secret and payload", request=request)
    controller = RequestController(connection, {"max_api_calls": 5}, transport=httpx.MockTransport(reply))
    with pytest.raises(PolicyCallError) as caught:
        controller.post(connection.url, json={})
    assert caught.value.code == "api_timeout"
    assert "private" not in str(caught.value)
    assert controller.snapshot()["actual_http_requests"] == 1
    assert controller.snapshot()["api_calls"][0]["error_code"] == "api_timeout"
    controller.close()


def test_unknown_usage_remains_null_and_snapshots_are_detached(connection):
    controller = RequestController(connection, {"max_api_calls": 1},
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body(usage=False))))
    controller.post(connection.url, json={})
    state = controller.snapshot()
    assert state["api_calls"][0]["input_tokens"] is None
    assert state["api_calls"][0]["output_tokens"] is None
    state["api_calls"].clear()
    assert len(controller.snapshot()["api_calls"]) == 1
    controller.close()


def test_requests_are_serial(connection):
    active = 0
    peak = 0
    def reply(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        time.sleep(.02)
        active -= 1
        return httpx.Response(200, json=body())
    controller = RequestController(connection, {"max_api_calls": 2}, transport=httpx.MockTransport(reply))
    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(lambda _: controller.post(connection.url, json={}), range(2)))
    assert len(responses) == 2
    assert peak == 1
    controller.close()


@pytest.fixture
def upstream_policy():
    # Integration tests run when the upstream source is supplied on PYTHONPATH;
    # credential/controller unit tests have no checkout dependency.
    return pytest.importorskip("embodied_jev.policies").DecisionPolicy


def test_upstream_choice_protocol_is_reused(connection, upstream_policy):
    prior = SimpleNamespace(close=lambda: None)
    session = SimpleNamespace(policy=prior, cancel=threading.Event(), started=None, status="idle")
    controller = install_policy(session, connection, {"max_api_calls": 2, "retry_backoff_seconds": 0},
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body())), policy_class=upstream_policy)
    result = session.policy.choose_plan({"observation": "example"}, "Choose an action", {"move": "move", "hold": "hold"})
    assert result["choice"] == "move"
    assert result["probabilities"] == {}
    assert result["selected_probability"] is None
    assert session.policy.provider == "chat"
    assert controller.snapshot()["policy_logical_calls"] == 1
    assert controller.snapshot()["actual_http_requests"] == 1
    session.policy.close()


def test_invalid_choice_fails_without_fallback(connection, upstream_policy):
    session = SimpleNamespace(policy=None, cancel=threading.Event(), started=None, status="idle")
    controller = install_policy(session, connection, {"max_api_calls": 1},
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body(choice="unoffered"))),
        policy_class=upstream_policy)
    with pytest.raises(PolicyCallError, match="decision_protocol_error"):
        session.policy.choose_plan({}, "Choose", {"move": "move", "hold": "hold"})
    assert controller.snapshot()["last_error"] == "decision_protocol_error"
    assert controller.snapshot()["actual_http_requests"] == 1
    session.policy.close()


def test_policy_install_refuses_active_session(connection, upstream_policy):
    with pytest.raises(PolicyCallError, match="idle_session"):
        install_policy(SimpleNamespace(started=1, status="running"), connection, {}, policy_class=upstream_policy)


def test_http_auth_failure_has_no_retry_or_body_leak(connection):
    controller = RequestController(connection, {"max_api_calls": 5}, transport=httpx.MockTransport(
        lambda request: httpx.Response(401, json={"error": "test-secret-only"})))
    with pytest.raises(PolicyCallError, match="http_401") as caught:
        controller.post(connection.url, json={})
    assert caught.value.__suppress_context__ is True
    assert "test-secret-only" not in json.dumps(controller.snapshot())
    assert controller.snapshot()["actual_http_requests"] == 1
    controller.close()


def test_hierarchical_chat_channels_reuse_upstream(connection, upstream_policy):
    session = SimpleNamespace(policy=None, cancel=threading.Event(), started=None, status="idle")
    choices = {"x": "positive", "y": "hold", "z": "negative", "gripper": "close"}
    response = {"model": "returned-model", "choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps(choices)}}]}
    controller = install_policy(session, connection, {"max_api_calls": 1},
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response)),
        policy_class=upstream_policy)
    questions = {axis: {"type": "choice", "instructions": "Choose", "criteria": {
        choice: choice for choice in (("open", "hold", "close") if axis == "gripper"
                                     else ("negative", "hold", "positive"))}} for axis in choices}
    result = session.policy.choose_channels({}, questions)
    assert {name: answer["choice"] for name, answer in result.items()} == choices
    assert controller.snapshot()["actual_http_requests"] == 1
    assert controller.snapshot()["policy_logical_calls"] == 1
    assert all(answer["selected_probability"] is None for answer in result.values())
    session.policy.close()


def test_baseline_needs_no_profile_or_credentials(upstream_policy):
    connection = build_connection({}, "baseline", environ={})
    assert connection == ResolvedConnection("baseline", "", "baseline", "")
    session = SimpleNamespace(policy=None, cancel=threading.Event(), started=None, status="idle")
    sent = []
    controller = install_policy(session, connection, {}, policy_class=upstream_policy,
        transport=httpx.MockTransport(lambda request: sent.append(request)))
    result = session.policy.choose_plan({}, "Choose", {"move": "move", "hold": "hold"}, baseline_choice="hold")
    assert result["choice"] == "hold"
    assert result["model_call"] is False
    assert sent == []
    assert controller.snapshot()["actual_http_requests"] == 0
    assert controller.snapshot()["policy_logical_calls"] == 0
    assert controller.snapshot()["last_error"] is None
    session.policy.close()
