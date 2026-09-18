import base64
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from blockbuster import BlockBuster
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai.chat_models.codex import (  # noqa: PLC2701
    CHATGPT_CODEX_BASE_URL,
    _ChatOpenAICodex,
)
from pydantic import BaseModel

from agent.utils import model, openai_oauth


@contextmanager
def detect_blocking_calls() -> Iterator[None]:
    """Leak-proof ``blockbuster_ctx``: deactivates even when the body raises."""
    blockbuster = BlockBuster()
    blockbuster.activate()
    try:
        yield
    finally:
        blockbuster.deactivate()


@pytest.fixture(autouse=True)
def _clean_oauth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OPEN_SWE_OPENAI_OAUTH_BROKER_URL",
        "OPEN_SWE_OPENAI_OAUTH_BROKER_TOKEN",
        "OPEN_SWE_CODEX_AUTH_FILE",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    model._MODEL_CACHE.clear()


def _configure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPEN_SWE_OPENAI_OAUTH_BROKER_URL", "http://127.0.0.1:3210/token")
    monkeypatch.setenv("OPEN_SWE_OPENAI_OAUTH_BROKER_TOKEN", "broker-secret")


def test_oauth_model_uses_dedicated_account_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch)
    captured: dict[str, Any] = {}

    def fake_model(model_name: str, **kwargs: Any) -> str:
        captured["model_name"] = model_name
        captured.update(kwargs)
        return "MODEL"

    with (
        patch.object(model, "build_openai_oauth_model", fake_model),
        detect_blocking_calls(),
    ):
        result = model.make_model("openai:gpt-5.6-sol", use_gateway=False, max_tokens=123)

    assert result == "MODEL"
    assert captured["model_name"] == "gpt-5.6-sol"
    assert "base_url" not in captured
    assert "max_tokens" not in captured


@pytest.mark.filterwarnings("ignore:.*experimental and unofficial.*:UserWarning")
def test_oauth_model_enforces_account_backend_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch)

    result = openai_oauth.build_desktop_openai_oauth_model("gpt-5.6-sol")

    assert isinstance(result, _ChatOpenAICodex)
    assert str(result.openai_api_base).rstrip("/") == CHATGPT_CODEX_BASE_URL
    assert result.use_responses_api is True
    assert result.store is False
    assert result.streaming is True
    assert result.originator == "open_swe_desktop"


def test_oauth_rejects_non_loopback_broker(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch)
    monkeypatch.setenv("OPEN_SWE_OPENAI_OAUTH_BROKER_URL", "https://example.com/token")
    assert openai_oauth.desktop_openai_oauth_available() is False


@pytest.mark.filterwarnings("ignore:.*experimental and unofficial.*:UserWarning")
async def test_token_provider_authenticates_to_broker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch)
    requests: list[tuple[str, dict[str, str]]] = []
    payloads = iter(
        [
            {"access_token": "access-token-a", "account_id": "account-a"},
            {"access_token": "access-token-b", "account_id": "account-b"},
        ]
    )

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return next(payloads)

    class Client:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def get(self, url: str, *, headers: dict[str, str]):
            requests.append((url, headers))
            return Response()

    oauth_model = openai_oauth.build_desktop_openai_oauth_model("gpt-5.6-sol")
    monkeypatch.setattr(openai_oauth.httpx2, "AsyncClient", Client)
    provider = oauth_model.token_provider  # type: ignore[attr-defined]
    first_token = await provider.aget_token()
    second_token = await provider.aget_token()

    assert first_token.access_token == "access-token-a"
    assert first_token.account_id == "account-a"
    assert second_token.access_token == "access-token-b"
    assert second_token.account_id == "account-b"
    assert provider.get_access_token() == "access-token-b"
    assert requests == [
        (
            "http://127.0.0.1:3210/token",
            {"Authorization": "Bearer broker-secret"},
        ),
        (
            "http://127.0.0.1:3210/token",
            {"Authorization": "Bearer broker-secret"},
        ),
    ]

    payload = oauth_model._get_request_payload(  # type: ignore[attr-defined]
        [SystemMessage("agent instructions"), HumanMessage("hello")]
    )
    assert payload["instructions"] == "agent instructions"
    assert all(item.get("role") != "system" for item in payload["input"])


def _jwt(claims: dict[str, object]) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"eyJhbGciOiJub25lIn0.{payload}.signature"


def _write_codex_auth(path: Path, tokens: dict[str, object] | None) -> None:
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt" if tokens else "apikey",
                "OPENAI_API_KEY": None,
                "tokens": tokens,
                "last_refresh": "2026-09-17T15:11:21Z",
            }
        )
    )
    # Rewrites within one mtime tick must still be noticed.
    stamp = path.stat().st_mtime_ns + 1_000_000
    os.utime(path, ns=(stamp, stamp))


def _chatgpt_tokens(access_token: str, account_id: str | None = "acct-1") -> dict[str, object]:
    return {
        "id_token": "id.token.x",
        "access_token": access_token,
        "refresh_token": "refresh-token",
        "account_id": account_id,
    }


@pytest.mark.filterwarnings("ignore:.*experimental and unofficial.*:UserWarning")
def test_codex_auth_file_backs_openai_models_without_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPEN_SWE_CODEX_AUTH_FILE", str(tmp_path / "missing" / "auth.json"))

    result = model.make_model("openai:gpt-5.6-sol", use_gateway=False, max_tokens=123)

    assert isinstance(result, _ChatOpenAICodex)
    assert str(result.openai_api_base).rstrip("/") == CHATGPT_CODEX_BASE_URL
    assert result.originator == "open_swe"
    assert result.max_tokens is None
    provider = result.token_provider  # type: ignore[attr-defined]
    assert isinstance(provider, openai_oauth._CodexCliTokenProvider)
    # The file is only consulted per request, so a missing login fails the
    # request with a hint rather than failing model construction.
    with pytest.raises(RuntimeError, match="codex login"):
        provider.get_token()


@pytest.mark.filterwarnings("ignore:.*experimental and unofficial.*:UserWarning")
async def test_codex_auth_file_tokens_follow_the_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    auth_file = tmp_path / "auth.json"
    monkeypatch.setenv("OPEN_SWE_CODEX_AUTH_FILE", str(auth_file))
    future = int(time.time()) + 3600
    _write_codex_auth(auth_file, _chatgpt_tokens(_jwt({"exp": future})))
    oauth_model = openai_oauth.build_openai_oauth_model("gpt-5.6-sol")
    provider = oauth_model.token_provider  # type: ignore[attr-defined]

    with detect_blocking_calls():
        first = await provider.aget_token()
    assert first.account_id == "acct-1"
    assert first.access_token == _jwt({"exp": future})

    # Codex CLI refreshed and dropped the explicit account id: the new token is
    # served and the account comes from the JWT claim instead.
    claims = {
        "exp": future,
        "https://api.openai.com/auth": {"chatgpt_account_id": "acct-2", "chatgpt_plan_type": "pro"},
    }
    _write_codex_auth(auth_file, _chatgpt_tokens(_jwt(claims), account_id=None))
    with detect_blocking_calls():
        second = await provider.aget_token()
    assert second.access_token == _jwt(claims)
    assert second.account_id == "acct-2"
    assert second.plan_type == "pro"
    assert provider.get_access_token() == _jwt(claims)

    payload = oauth_model._get_request_payload(  # type: ignore[attr-defined]
        [SystemMessage("agent instructions"), HumanMessage("hello")]
    )
    assert payload["instructions"] == "agent instructions"


def test_codex_auth_file_expired_login_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    auth_file = tmp_path / "auth.json"
    monkeypatch.setenv("OPEN_SWE_CODEX_AUTH_FILE", str(auth_file))
    _write_codex_auth(auth_file, _chatgpt_tokens(_jwt({"exp": int(time.time()) - 60})))

    provider = openai_oauth._CodexCliTokenProvider(auth_file)

    with pytest.raises(RuntimeError, match="expired"):
        provider.get_token()


def test_codex_auth_file_without_chatgpt_login_is_rejected(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    _write_codex_auth(auth_file, None)

    provider = openai_oauth._CodexCliTokenProvider(auth_file)

    with pytest.raises(RuntimeError, match="no ChatGPT login"):
        provider.get_token()


@pytest.mark.filterwarnings("ignore:.*experimental and unofficial.*:UserWarning")
def test_desktop_broker_takes_precedence_over_codex_auth_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure(monkeypatch)
    monkeypatch.setenv("OPEN_SWE_CODEX_AUTH_FILE", str(tmp_path / "auth.json"))

    oauth_model = openai_oauth.build_openai_oauth_model("gpt-5.6-sol")

    assert oauth_model.originator == "open_swe_desktop"  # type: ignore[attr-defined]


def test_codex_auth_file_counts_as_openai_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert openai_oauth.openai_oauth_available() is False
    monkeypatch.setenv("OPEN_SWE_CODEX_AUTH_FILE", "~/.codex/auth.json")
    assert openai_oauth.openai_oauth_available() is True
    assert openai_oauth.codex_auth_file() == Path.home() / ".codex" / "auth.json"


class _Decision(BaseModel):
    route: str


@pytest.mark.filterwarnings("ignore:.*experimental and unofficial.*:UserWarning")
def test_codex_models_serve_structured_output_through_tool_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPEN_SWE_CODEX_AUTH_FILE", str(tmp_path / "auth.json"))
    oauth_model = openai_oauth.build_openai_oauth_model("gpt-5.6-sol")

    structured = oauth_model.with_structured_output(_Decision)

    bound_kwargs = structured.first.kwargs  # type: ignore[attr-defined]
    assert "response_format" not in bound_kwargs
    assert [tool["function"]["name"] for tool in bound_kwargs["tools"]] == ["_Decision"]
    assert bound_kwargs["tool_choice"]["function"]["name"] == "_Decision"
