"""Supported models and reasoning efforts surfaced in the profile editor."""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache, lru_cache
from importlib import import_module
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from agent.config import ENV


class ModelOption(TypedDict):
    id: str
    label: str
    efforts: list[str]
    default_effort: str
    supports_images: bool
    can_be_default: NotRequired[bool]
    context_window: NotRequired[int | None]


BUILTIN_MODELS: list[ModelOption] = [
    {
        "id": "anthropic:claude-opus-5",
        "label": "Opus 5",
        "efforts": ["low", "medium", "high", "xhigh", "max"],
        "default_effort": "high",
        "supports_images": True,
    },
    {
        "id": "anthropic:claude-sonnet-5",
        "label": "Sonnet 5",
        "efforts": ["low", "medium", "high", "xhigh", "max"],
        "default_effort": "high",
        "supports_images": True,
    },
    {
        "id": "anthropic:claude-fable-5-1",
        "label": "Fable 5.1",
        "efforts": ["low", "medium", "high", "xhigh", "max"],
        "default_effort": "high",
        "supports_images": True,
        "can_be_default": False,
    },
    {
        "id": "anthropic:claude-haiku-4-5",
        "label": "Haiku 4.5",
        # Haiku 4.5 predates the adaptive-thinking/effort params the other
        # Claude entries rely on, so it is offered without reasoning.
        "efforts": ["none"],
        "default_effort": "none",
        "supports_images": True,
    },
    {
        "id": "openai:gpt-6-astra",
        "label": "GPT-6 Astra",
        "efforts": ["low", "medium", "high", "xhigh", "max"],
        "default_effort": "low",
        "supports_images": True,
    },
    {
        "id": "openai:gpt-5.6-sol",
        "label": "GPT-5.6 Sol",
        "efforts": ["none", "low", "medium", "high", "xhigh"],
        "default_effort": "xhigh",
        "supports_images": True,
    },
    {
        "id": "openai:gpt-5.6-terra",
        "label": "GPT-5.6 Terra",
        "efforts": ["none", "low", "medium", "high", "xhigh"],
        "default_effort": "xhigh",
        "supports_images": True,
    },
    {
        "id": "openai:gpt-5.6-luna",
        "label": "GPT-5.6 Luna",
        "efforts": ["none", "low", "medium", "high", "xhigh", "max"],
        "default_effort": "xhigh",
        "supports_images": True,
    },
    {
        "id": "google_genai:gemini-3.8-flash",
        "label": "Gemini 3.8 Flash",
        "efforts": ["minimal", "low", "medium", "high"],
        "default_effort": "medium",
        "supports_images": True,
    },
    {
        "id": "fireworks:accounts/fireworks/models/kimi-k3",
        "label": "Kimi K3",
        # K3 always reasons and only accepts low/high/max — "medium" is rejected.
        "efforts": ["low", "high", "max"],
        "default_effort": "high",
        "supports_images": False,
    },
    {
        "id": "fireworks:accounts/fireworks/models/deepseek-v4-pro",
        "label": "DeepSeek V4 Pro",
        "efforts": ["none", "low", "medium", "high", "xhigh", "max"],
        "default_effort": "high",
        "supports_images": False,
    },
    {
        "id": "fireworks:accounts/fireworks/models/glm-5p3",
        "label": "GLM 5.3",
        "efforts": ["none", "high", "max"],
        "default_effort": "high",
        "supports_images": False,
    },
    {
        "id": "fireworks:accounts/fireworks/models/glm-5p3-flash",
        "label": "GLM-5.3 Flash",
        "efforts": ["low", "high", "max"],
        "default_effort": "high",
        "supports_images": True,
    },
]

BUILTIN_MODEL_IDS: frozenset[str] = frozenset(m["id"] for m in BUILTIN_MODELS)

# Models added, overridden or hidden through OPEN_SWE_EXTRA_MODELS_FILE (see load_extra_models).
EXTRA_MODEL_EFFORTS: list[str] = ["none", "low", "medium", "high"]
EXTRA_MODEL_DEFAULT_EFFORT: str = "medium"
# Every effort a built-in model offers is one the provider kwargs know how to send;
# per provider, so an effort another provider's models understand isn't accepted
# for one that would silently drop it (``minimal`` on ``openai:``).
_EFFORTS_BY_PROVIDER: dict[str, frozenset[str]] = {
    provider: frozenset(
        effort
        for model in BUILTIN_MODELS
        if model["id"].partition(":")[0] == provider
        for effort in model["efforts"]
    )
    for provider in {model["id"].partition(":")[0] for model in BUILTIN_MODELS}
}
KNOWN_EFFORTS: frozenset[str] = frozenset().union(*_EFFORTS_BY_PROVIDER.values())
_EXTRA_MODEL_FIELDS: frozenset[str] = frozenset(
    {
        "id",
        "label",
        "efforts",
        "default_effort",
        "supports_images",
        "can_be_default",
        "context_window",
        "hidden",
    }
)


@dataclass(frozen=True)
class ExtraModels:
    """What an extra-models file declares: new or overriding entries, and hidden built-ins."""

    models: list[ModelOption]
    hidden: frozenset[str]


def _extra_model_label(model_id: str) -> str:
    return model_id.partition(":")[2].rsplit("/", 1)[-1]


def _extra_model_id(entry: Mapping[str, object], where: str) -> str:
    model_id = entry.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError(f'{where} needs an "id" in provider:model form')
    model_id = model_id.strip()
    provider, _, name = model_id.partition(":")
    if not provider or not name:
        raise ValueError(f"{where}: id {model_id!r} is not in provider:model form")
    return model_id


def _extra_model_efforts(
    entry: Mapping[str, object], where: str, base: ModelOption | None
) -> tuple[list[str], str]:
    efforts_value = entry.get("efforts", base["efforts"] if base else EXTRA_MODEL_EFFORTS)
    if (
        not isinstance(efforts_value, list)
        or not efforts_value
        or not all(isinstance(effort, str) for effort in efforts_value)
    ):
        raise ValueError(f'{where}: "efforts" must be a list of at least one effort name')
    efforts = [str(effort) for effort in efforts_value]
    provider = str(entry["id"]).partition(":")[0]
    allowed = _EFFORTS_BY_PROVIDER.get(provider, KNOWN_EFFORTS)
    for effort in efforts:
        if effort not in allowed:
            raise ValueError(
                f'{where}: unknown effort "{effort}" for {provider}: models; '
                f"use one of {sorted(allowed)}"
            )
    default_effort = entry.get("default_effort")
    if default_effort is None:
        if base is not None and base["default_effort"] in efforts:
            default_effort = base["default_effort"]
        elif EXTRA_MODEL_DEFAULT_EFFORT in efforts:
            default_effort = EXTRA_MODEL_DEFAULT_EFFORT
        else:
            default_effort = efforts[0]
    if not isinstance(default_effort, str) or default_effort not in efforts:
        raise ValueError(f'{where}: "default_effort" must be one of its "efforts" {efforts}')
    return efforts, default_effort


def _hidden_model_id(fields: Mapping[str, object], where: str) -> str:
    model_id = _extra_model_id(fields, where)
    if model_id not in BUILTIN_MODEL_IDS:
        raise ValueError(f'{where}: "hidden" only hides a built-in model, and {model_id!r} is none')
    if set(fields) - {"id", "hidden"}:
        raise ValueError(f'{where}: a "hidden" entry takes no other field than "id"')
    return model_id


def _parse_extra_model(fields: Mapping[str, object], where: str) -> ModelOption:
    model_id = _extra_model_id(fields, where)
    base = next((m for m in BUILTIN_MODELS if m["id"] == model_id), None)
    efforts, default_effort = _extra_model_efforts(fields, where, base)
    label = fields.get("label", base["label"] if base else _extra_model_label(model_id))
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f'{where}: "label" must be a non-empty string')
    supports_images = fields.get("supports_images", base["supports_images"] if base else False)
    if not isinstance(supports_images, bool):
        raise ValueError(f'{where}: "supports_images" must be true or false')
    option: ModelOption = {
        "id": model_id,
        "label": label.strip(),
        "efforts": efforts,
        "default_effort": default_effort,
        "supports_images": supports_images,
    }
    can_be_default = fields.get("can_be_default", base.get("can_be_default") if base else None)
    if can_be_default is not None:
        if not isinstance(can_be_default, bool):
            raise ValueError(f'{where}: "can_be_default" must be true or false')
        option["can_be_default"] = can_be_default
    context_window = fields.get("context_window")
    if context_window is not None:
        if (
            isinstance(context_window, bool)
            or not isinstance(context_window, int)
            or context_window <= 0
        ):
            raise ValueError(f'{where}: "context_window" must be a positive integer of tokens')
        option["context_window"] = context_window
    return option


def load_extra_models(path: Path) -> ExtraModels:
    """Parse the JSON list of extra selectable models at ``path``.

    Each entry needs an ``id`` in ``provider:model`` form; ``label``, ``efforts``,
    ``default_effort``, ``supports_images``, ``can_be_default`` and
    ``context_window`` are optional. An entry naming a built-in model overrides
    it field by field, and ``{"id": ..., "hidden": true}`` drops a built-in model
    from the catalog. This is how models served by an OpenAI-compatible gateway
    such as LiteLLM (``openai:<name>`` with ``OPENAI_BASE_URL`` pointing at the
    gateway) become selectable. Every problem raises ``ValueError`` naming the
    file and entry, so a typo fails startup loudly instead of silently dropping a
    model.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"Extra models file {path} does not exist") from exc
    try:
        raw: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Extra models file {path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError(f"Extra models file {path} must contain a JSON list of models")
    models: list[ModelOption] = []
    hidden: set[str] = set()
    seen: set[str] = set()
    for index, entry in enumerate(cast(list[object], raw)):
        where = f"{path} entry {index}"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{where} must be an object")
        fields = cast(Mapping[str, object], entry)
        for field in fields:
            if field not in _EXTRA_MODEL_FIELDS:
                raise ValueError(
                    f'{where}: unknown field "{field}"; allowed fields are '
                    f"{sorted(_EXTRA_MODEL_FIELDS)}"
                )
        hide = fields.get("hidden", False)
        if not isinstance(hide, bool):
            raise ValueError(f'{where}: "hidden" must be true or false')
        if hide:
            model_id = _hidden_model_id(fields, where)
            hidden.add(model_id)
        else:
            option = _parse_extra_model(fields, where)
            model_id = option["id"]
            models.append(option)
        if model_id in seen:
            raise ValueError(f"{where}: {model_id!r} is listed twice")
        seen.add(model_id)
    return ExtraModels(models=models, hidden=frozenset(hidden))


def _extra_models_from_env() -> ExtraModels:
    configured = ENV.OPEN_SWE_EXTRA_MODELS_FILE.optional()
    if configured is None:
        return ExtraModels(models=[], hidden=frozenset())
    return load_extra_models(Path(configured).expanduser())


def _declared_context_windows(models: Sequence[ModelOption]) -> dict[str, int]:
    windows: dict[str, int] = {}
    for model in models:
        context_window = model.get("context_window")
        if context_window is not None:
            windows[model["id"]] = context_window
    return windows


def _catalog_entry(model: ModelOption) -> ModelOption:
    """The entry as the catalog carries it: the context window lives in the
    override map below (like the Codex ones) and reaches the pickers through
    ``models_with_profile_context_windows``."""
    option: ModelOption = {
        "id": model["id"],
        "label": model["label"],
        "efforts": model["efforts"],
        "default_effort": model["default_effort"],
        "supports_images": model["supports_images"],
    }
    if "can_be_default" in model:
        option["can_be_default"] = model["can_be_default"]
    return option


_EXTRA: ExtraModels = _extra_models_from_env()
_BUILTIN_OVERRIDES: dict[str, ModelOption] = {
    m["id"]: m for m in _EXTRA.models if m["id"] in BUILTIN_MODEL_IDS
}
EXTRA_MODELS: list[ModelOption] = [
    _catalog_entry(m) for m in _EXTRA.models if m["id"] not in BUILTIN_MODEL_IDS
]
EXTRA_MODEL_IDS: frozenset[str] = frozenset(m["id"] for m in EXTRA_MODELS)
SUPPORTED_MODELS: list[ModelOption] = [
    *(
        _catalog_entry(_BUILTIN_OVERRIDES.get(m["id"], m))
        for m in BUILTIN_MODELS
        if m["id"] not in _EXTRA.hidden
    ),
    *EXTRA_MODELS,
]
SUPPORTED_MODEL_IDS: frozenset[str] = frozenset(m["id"] for m in SUPPORTED_MODELS)

FABLE_MODEL_IDS: frozenset[str] = frozenset(
    m["id"] for m in SUPPORTED_MODELS if m["id"].startswith("anthropic:claude-fable")
)
NON_DEFAULT_MODEL_IDS: frozenset[str] = frozenset(
    m["id"] for m in SUPPORTED_MODELS if not m.get("can_be_default", True)
)

DEPRECATED_MODEL_IDS: frozenset[str] = frozenset(
    {
        "anthropic:claude-opus-4-8",
        "anthropic:claude-fable-5",
        "openai:gpt-5.5",
        "google_genai:gemini-3.5-flash",
        "google_genai:gemini-3.6-flash",
        "google_genai:gemini-3.7-flash",
        "fireworks:accounts/fireworks/models/kimi-k2p7-code",
        "fireworks:accounts/fireworks/models/kimi-k3-code",
        "fireworks:accounts/fireworks/models/glm-5p2",
    }
)

DEPRECATED_MODEL_REPLACEMENTS: dict[str, str] = dict.fromkeys(DEPRECATED_MODEL_IDS, "")

ProfileLoader = Callable[[str], Mapping[str, object]]

# LangChain partner packages expose ``_get_default_model_profile`` — the same
# accessor that populates ``ChatModel.profile`` from bundled models.dev data.
_PROFILE_LOADER_MODULES: dict[str, str] = {
    "anthropic": "langchain_anthropic.chat_models",
    "fireworks": "langchain_fireworks.chat_models",
    "google_genai": "langchain_google_genai.chat_models",
    "openai": "langchain_openai.chat_models.base",
}
CODEX_CONTEXT_WINDOW_OVERRIDES: dict[str, int] = {
    "openai:gpt-6-astra": 272_000,
    "openai:gpt-5.6-sol": 272_000,
    "openai:gpt-5.6-terra": 272_000,
    "openai:gpt-5.6-luna": 272_000,
}
_PROFILE_CONTEXT_WINDOW_FALLBACKS: dict[str, int] = {
    "fireworks:accounts/fireworks/models/kimi-k3": 1_048_576,
    "fireworks:accounts/fireworks/models/glm-5p3": 1_048_576,
    "fireworks:accounts/fireworks/models/glm-5p3-flash": 1_048_576,
}
# Extra models are unknown to the partner packages' models.dev profiles, so their
# file-declared context window is the only source and overrides like the Codex ones.
_CONTEXT_WINDOW_OVERRIDES: dict[str, int] = {
    **CODEX_CONTEXT_WINDOW_OVERRIDES,
    **_declared_context_windows(_EXTRA.models),
}


@cache
def _profile_loader(provider: str) -> ProfileLoader | None:
    module_path = _PROFILE_LOADER_MODULES.get(provider)
    if module_path is None:
        return None
    try:
        module = import_module(module_path)
    except ImportError:
        return None
    loader = getattr(module, "_get_default_model_profile", None)
    if not callable(loader):
        return None
    return cast(ProfileLoader, loader)


def model_profile_with_context_override(model_id: str) -> dict[str, object] | None:
    context_window = _CONTEXT_WINDOW_OVERRIDES.get(model_id)
    if context_window is None:
        return None
    provider, _, model_name = model_id.partition(":")
    loader = _profile_loader(provider)
    profile = dict(loader(model_name)) if loader is not None else {}
    profile["max_input_tokens"] = context_window
    return profile


@lru_cache(maxsize=512)
def model_profile_context_window(model_id: str) -> int | None:
    context_window = _CONTEXT_WINDOW_OVERRIDES.get(model_id)
    if context_window is not None:
        return context_window
    provider, _, model_name = model_id.partition(":")
    if not provider or not model_name:
        return None
    loader = _profile_loader(provider)
    if loader is not None:
        profile = loader(model_name)
        context_window = profile.get("max_input_tokens")
        if isinstance(context_window, int) and context_window > 0:
            return context_window
    return _PROFILE_CONTEXT_WINDOW_FALLBACKS.get(model_id)


def models_with_profile_context_windows(models: Sequence[ModelOption]) -> list[ModelOption]:
    enriched: list[ModelOption] = []
    for model in models:
        option: ModelOption = {
            "id": model["id"],
            "label": model["label"],
            "efforts": model["efforts"],
            "default_effort": model["default_effort"],
            "supports_images": model["supports_images"],
        }
        if "can_be_default" in model:
            option["can_be_default"] = model["can_be_default"]
        context_window = model_profile_context_window(model["id"])
        if context_window is not None:
            option["context_window"] = context_window
        enriched.append(option)
    return enriched


def fable_disabled_fallback(effort: object = None) -> tuple[str, str]:
    """Newest supported non-Fable Anthropic model (keeps the Claude family),
    else the global default. Substitutes a Fable selection when Fable is
    disabled workspace-wide, preserving ``effort`` when the fallback supports it."""
    for m in SUPPORTED_MODELS:
        if m["id"].startswith("anthropic:") and m["id"] not in FABLE_MODEL_IDS:
            return m["id"], _fallback_effort_for(m, effort) or m["default_effort"]
    return default_model_pair()


def gate_fable_model(
    model_id: str, effort: str | None, *, fable_enabled: bool
) -> tuple[str, str | None]:
    """ZDR guard: if Fable is disabled but a Fable id was resolved, swap in a
    safe non-Fable model. Non-Fable selections pass through unchanged. Applied
    at every model-construction entrypoint so a disabled Fable model can never
    reach ``make_model``, no matter which layer selected it."""
    if not fable_enabled and isinstance(model_id, str) and model_id in FABLE_MODEL_IDS:
        return fable_disabled_fallback(effort)
    return model_id, effort


def _default_model_id() -> str:
    """The catalog default: Opus on an Anthropic-only deployment, else GPT-5.6 Sol.

    When the extra-models file hides that model, the next selectable model of
    the same provider takes its place, then any selectable model.
    """
    preferred = (
        "anthropic:claude-opus-5"
        if ENV.ANTHROPIC_API_KEY.optional() and not ENV.OPENAI_API_KEY.optional()
        else "openai:gpt-5.6-sol"
    )
    if preferred in SUPPORTED_MODEL_IDS:
        return preferred
    provider = preferred.partition(":")[0]
    selectable = [m["id"] for m in SUPPORTED_MODELS if m.get("can_be_default", True)]
    same_provider = [model_id for model_id in selectable if model_id.startswith(f"{provider}:")]
    return (same_provider or selectable or [preferred])[0]


DEFAULT_MODEL_ID: str = _default_model_id()
DEFAULT_MODEL_EFFORT: str = "medium"


def model_supports_effort(model_id: str, effort: str) -> bool:
    for m in SUPPORTED_MODELS:
        if m["id"] == model_id:
            return effort in m["efforts"]
    return False


def model_supports_images(model_id: str) -> bool:
    for m in SUPPORTED_MODELS:
        if m["id"] == model_id:
            return m["supports_images"]
    return False


def _provider_of(model_id: str) -> str | None:
    provider, _, rest = model_id.partition(":")
    return provider if rest else None


def _claude_family_of(model_id: str) -> str | None:
    provider, _, name = model_id.partition(":")
    if provider != "anthropic" or not name.startswith("claude-"):
        return None
    parts = name.split("-")
    if len(parts) < 2:
        return None
    return "-".join(parts[:2])


def _fallback_effort_for(model: ModelOption, effort: object) -> str | None:
    if not isinstance(effort, str):
        return None
    if effort in model["efforts"]:
        return effort
    if (
        model["id"].startswith("google_genai:")
        and effort == "none"
        and "minimal" in model["efforts"]
    ):
        return "minimal"
    return None


def is_deprecated_model(model_id: object) -> bool:
    return isinstance(model_id, str) and model_id in DEPRECATED_MODEL_IDS


def canonical_model_pair(model_id: object, effort: object = None) -> tuple[str, str] | None:
    return None


def normalize_model_choice(model_id: object, effort: object) -> tuple[str | None, str | None]:
    if (
        not isinstance(model_id, str)
        or model_id not in SUPPORTED_MODEL_IDS
        or not isinstance(effort, str)
        or not model_supports_effort(model_id, effort)
    ):
        return None, None
    return model_id, effort


def provider_fallback_pair(model_id: object, effort: object = None) -> tuple[str, str] | None:
    """Newest supported ``(model_id, effort)`` for the same provider/family.

    Keeps a stored selection on its original provider when its exact id has
    dropped out of the supported set (e.g. an Opus minor-version bump), preferring
    the same Claude family when available instead of falling through to the
    cross-provider global default. Preserves ``effort`` when the fallback model
    supports it, otherwise uses that model's default effort. Returns ``None`` when
    no supported model shares the provider.

    Explicitly deprecated ids inherit the workspace default instead.
    """
    if not isinstance(model_id, str) or model_id in DEPRECATED_MODEL_IDS:
        return None
    provider = _provider_of(model_id)
    if provider is None:
        return None
    family = _claude_family_of(model_id)
    if family is not None:
        for m in SUPPORTED_MODELS:
            if _provider_of(m["id"]) == provider and _claude_family_of(m["id"]) == family:
                return m["id"], _fallback_effort_for(m, effort) or m["default_effort"]
    for m in SUPPORTED_MODELS:
        if _provider_of(m["id"]) == provider:
            return m["id"], _fallback_effort_for(m, effort) or m["default_effort"]
    return None


def default_model_pair() -> tuple[str, str]:
    """Deployment fallback used when neither the workspace nor the instance sets a default."""
    model_id = ENV.LLM_MODEL_ID.get(DEFAULT_MODEL_ID)
    effort = ENV.LLM_REASONING_EFFORT.get()
    for model in SUPPORTED_MODELS:
        if model["id"] == model_id and model.get("can_be_default", True):
            effort = (
                effort
                or _fallback_effort_for(model, DEFAULT_MODEL_EFFORT)
                or model["default_effort"]
            )
            if effort not in model["efforts"]:
                raise ValueError(f"Unsupported LLM_REASONING_EFFORT {effort!r} for {model_id!r}")
            return model_id, effort
    raise ValueError(f"Unsupported default LLM_MODEL_ID: {model_id!r}")


def default_vision_model_pair() -> tuple[str, str]:
    """Default OpenAI/Anthropic model pair to use when image input is required."""
    if (
        DEFAULT_MODEL_ID in SUPPORTED_MODEL_IDS
        and model_supports_images(DEFAULT_MODEL_ID)
        and model_supports_effort(DEFAULT_MODEL_ID, DEFAULT_MODEL_EFFORT)
        and DEFAULT_MODEL_ID.startswith(("openai:", "anthropic:"))
    ):
        return DEFAULT_MODEL_ID, DEFAULT_MODEL_EFFORT
    for model in SUPPORTED_MODELS:
        if model["id"].startswith(("openai:", "anthropic:")) and model["supports_images"]:
            return model["id"], model["default_effort"]
    return default_model_pair()
