"""OpenAI models on a ChatGPT login instead of an API key.

Both credential sources feed ``_ChatOpenAICodex`` and are applied by
:func:`agent.utils.model.make_model` when ``OPENAI_API_KEY`` is unset and the
LangSmith gateway is off:

- the desktop app's loopback token broker, which owns the OAuth login and refresh;
- a Codex CLI ``auth.json`` named by ``OPEN_SWE_CODEX_AUTH_FILE``, so a ``codex login``
  on the host serves Open SWE too. Refreshing stays with Codex CLI: rotating its
  refresh token from here would invalidate the CLI's own session, and its access
  tokens last days, so re-reading the file whenever it changes is enough.
"""

import asyncio
import json
import threading
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx2
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.runnables import Runnable
from langchain_openai.chat_models.base import (
    _DictOrPydantic,  # noqa: PLC2701
    _DictOrPydanticClass,  # noqa: PLC2701
)
from langchain_openai.chat_models.codex import _ChatOpenAICodex  # noqa: PLC2701
from langchain_openai.chatgpt_oauth import (
    CHATGPT_AUTH_CLAIMS_NAMESPACE,
    _ChatGPTOAuthTokenProvider,  # noqa: PLC2701
    _ChatGPTToken,  # noqa: PLC2701
    decode_jwt_claims,
)

from agent.config import ENV

_BROKER_MANAGED_REFRESH_TOKEN = "managed-by-desktop-broker"
_CODEX_CLI_MANAGED_REFRESH_TOKEN = "managed-by-codex-cli"
_DESKTOP_ORIGINATOR = "open_swe_desktop"
_CODEX_CLI_ORIGINATOR = "open_swe"
# ``_ChatGPTToken`` insists on an expiry; an access token without an ``exp`` claim
# is trusted until the backend rejects it.
_NO_EXPIRY = datetime.max.replace(tzinfo=UTC)


class _OpenSWEChatOpenAICodex(_ChatOpenAICodex):
    """``_ChatOpenAICodex`` whose structured output survives the Codex backend.

    The backend applies a ``json_schema`` text format, but the streamed completion
    (streaming is forced) does not echo ``text.format``, so langchain never fills
    ``parsed`` and its parser rejects the valid JSON it received. Tool calling
    works, so ``json_schema`` requests are served through ``function_calling``.
    """

    def with_structured_output(
        self,
        schema: _DictOrPydanticClass | None = None,
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = "json_schema",
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        if method == "json_schema":
            method = "function_calling"
        return super().with_structured_output(schema, method=method, **kwargs)


def _account_id(value: object) -> str | None:
    """``value`` when it is a ChatGPT account id safe to send as a header."""
    if not isinstance(value, str) or not value or len(value) > 512:
        return None
    if "\r" in value or "\n" in value:
        return None
    return value


# --- Desktop token broker ------------------------------------------------------------


def _broker_config() -> tuple[str, str] | None:
    url = ENV.OPEN_SWE_OPENAI_OAUTH_BROKER_URL.get()
    token = ENV.OPEN_SWE_OPENAI_OAUTH_BROKER_TOKEN.get()
    if not url or not token:
        return None
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.path != "/token":
        return None
    return url, token


def desktop_openai_oauth_available() -> bool:
    return _broker_config() is not None


class _DesktopOpenAIOAuthTokenProvider(_ChatGPTOAuthTokenProvider):
    def __init__(self, broker_url: str, broker_token: str) -> None:
        self._broker_url = broker_url
        self._broker_token = broker_token
        self._current_token: _ChatGPTToken | None = None

    def get_token(self) -> _ChatGPTToken:
        if self._current_token is None:
            raise RuntimeError("Local OpenAI credentials have not been fetched asynchronously")
        return self._current_token

    async def aget_token(self) -> _ChatGPTToken:
        async with httpx2.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                self._broker_url,
                headers={"Authorization": f"Bearer {self._broker_token}"},
            )
        response.raise_for_status()
        payload = response.json()
        access_token = payload.get("access_token") if isinstance(payload, dict) else None
        account_id = _account_id(payload.get("account_id")) if isinstance(payload, dict) else None
        if not isinstance(access_token, str) or not access_token:
            raise ValueError("Local OpenAI credential broker returned no access token")
        if account_id is None:
            raise ValueError("Local OpenAI credential broker returned no account ID")
        token = _ChatGPTToken(
            access_token=access_token,
            refresh_token=_BROKER_MANAGED_REFRESH_TOKEN,
            expires_at=datetime.now(UTC) + timedelta(hours=1),
            account_id=account_id,
        )
        self._current_token = token
        return token

    def get_access_token(self) -> str:
        return self.get_token().access_token

    async def aget_access_token(self) -> str:
        return (await self.aget_token()).access_token


def build_desktop_openai_oauth_model(model_name: str, **kwargs: Any) -> BaseChatModel:
    config = _broker_config()
    if config is None:
        raise ValueError("Local OpenAI credentials are unavailable")
    provider = _DesktopOpenAIOAuthTokenProvider(*config)
    return _OpenSWEChatOpenAICodex(
        model=model_name,
        token_provider=provider,
        originator=_DESKTOP_ORIGINATOR,
        **kwargs,
    )


# --- Codex CLI auth file -----------------------------------------------------------


def codex_auth_file() -> Path | None:
    """The Codex CLI ``auth.json`` OpenAI models may run on, when configured."""
    value = ENV.OPEN_SWE_CODEX_AUTH_FILE.optional()
    return Path(value).expanduser() if value else None


def codex_cli_oauth_available() -> bool:
    return codex_auth_file() is not None


def openai_oauth_available() -> bool:
    """Whether OpenAI models can run on a ChatGPT login instead of ``OPENAI_API_KEY``."""
    return desktop_openai_oauth_available() or codex_cli_oauth_available()


def _codex_login_hint(path: Path) -> str:
    return f"run `codex login` on this machine so {path} holds a ChatGPT login"


def _parse_codex_auth_file(path: Path) -> _ChatGPTToken:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Codex CLI auth file {path} could not be read: {exc}") from exc
    tokens = raw.get("tokens") if isinstance(raw, dict) else None
    access_token = tokens.get("access_token") if isinstance(tokens, dict) else None
    if not isinstance(tokens, dict) or not isinstance(access_token, str) or not access_token:
        raise RuntimeError(
            f"Codex CLI auth file {path} holds no ChatGPT login (an API-key login does not "
            f"apply); {_codex_login_hint(path)}"
        )
    claims: Mapping[str, object] = decode_jwt_claims(access_token)
    auth_claims = claims.get(CHATGPT_AUTH_CLAIMS_NAMESPACE)
    if not isinstance(auth_claims, dict):
        auth_claims = {}
    account_id = _account_id(tokens.get("account_id")) or _account_id(
        auth_claims.get("chatgpt_account_id")
    )
    if account_id is None:
        raise RuntimeError(
            f"Codex CLI auth file {path} names no ChatGPT account; {_codex_login_hint(path)}"
        )
    exp = claims.get("exp")
    expires_at = datetime.fromtimestamp(exp, tz=UTC) if isinstance(exp, int | float) else _NO_EXPIRY
    plan_type = auth_claims.get("chatgpt_plan_type")
    return _ChatGPTToken(
        access_token=access_token,
        refresh_token=_CODEX_CLI_MANAGED_REFRESH_TOKEN,
        expires_at=expires_at,
        account_id=account_id,
        plan_type=plan_type if isinstance(plan_type, str) else None,
    )


class _CodexCliTokenProvider(_ChatGPTOAuthTokenProvider):
    """Serves the ChatGPT tokens Codex CLI keeps in ``auth.json``.

    The file is parsed again whenever its size or mtime changes, so a refresh or a
    new login by Codex CLI is picked up without restarting the server.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._cached: tuple[tuple[int, int], _ChatGPTToken] | None = None

    def _load(self) -> _ChatGPTToken:
        try:
            stat = self._path.stat()
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Codex CLI auth file {self._path} does not exist; {_codex_login_hint(self._path)}"
            ) from exc
        key = (stat.st_mtime_ns, stat.st_size)
        with self._lock:
            if self._cached is None or self._cached[0] != key:
                self._cached = (key, _parse_codex_auth_file(self._path))
            token = self._cached[1]
        if token.is_expired():
            raise RuntimeError(
                f"The ChatGPT login in Codex CLI auth file {self._path} has expired; run "
                "`codex login` (or any Codex command) so Codex CLI refreshes it"
            )
        return token

    def get_token(self) -> _ChatGPTToken:
        return self._load()

    async def aget_token(self) -> _ChatGPTToken:
        return await asyncio.to_thread(self._load)

    def get_access_token(self) -> str:
        return self.get_token().access_token

    async def aget_access_token(self) -> str:
        return (await self.aget_token()).access_token


def build_codex_cli_oauth_model(model_name: str, **kwargs: Any) -> BaseChatModel:
    path = codex_auth_file()
    if path is None:
        raise ValueError("OPEN_SWE_CODEX_AUTH_FILE is not set")
    return _OpenSWEChatOpenAICodex(
        model=model_name,
        token_provider=_CodexCliTokenProvider(path),
        originator=_CODEX_CLI_ORIGINATOR,
        **kwargs,
    )


def build_openai_oauth_model(model_name: str, **kwargs: Any) -> BaseChatModel:
    """A Codex-backend model on whichever ChatGPT login is configured.

    The desktop broker wins over the Codex CLI file: the app's own sign-in is the
    explicit choice, and it refreshes its tokens itself.
    """
    if desktop_openai_oauth_available():
        return build_desktop_openai_oauth_model(model_name, **kwargs)
    return build_codex_cli_oauth_model(model_name, **kwargs)
