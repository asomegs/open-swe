"""Extra selectable models loaded from ``OPEN_SWE_EXTRA_MODELS_FILE`` (agent/dashboard/options.py)."""

import json
import runpy
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest

from agent.dashboard import options
from agent.dashboard.options import load_extra_models


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "extra-models.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_minimal_entry_gets_defaults(tmp_path: Path) -> None:
    path = _write(tmp_path, [{"id": "openai:claude-sonnet-4-5"}])
    assert load_extra_models(path).models == [
        {
            "id": "openai:claude-sonnet-4-5",
            "label": "claude-sonnet-4-5",
            "efforts": ["none", "low", "medium", "high"],
            "default_effort": "medium",
            "supports_images": False,
        }
    ]


def test_explicit_fields_are_kept(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            {
                "id": "openai:vertex/gemini-2.5-pro",
                "label": "Gemini 2.5 Pro (LiteLLM)",
                "efforts": ["low", "high"],
                "default_effort": "high",
                "supports_images": True,
                "can_be_default": False,
                "context_window": 1_000_000,
            }
        ],
    )
    assert load_extra_models(path).models == [
        {
            "id": "openai:vertex/gemini-2.5-pro",
            "label": "Gemini 2.5 Pro (LiteLLM)",
            "efforts": ["low", "high"],
            "default_effort": "high",
            "supports_images": True,
            "can_be_default": False,
            "context_window": 1_000_000,
        }
    ]


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        ({"id": "openai:x"}, "JSON list"),
        (["openai:x"], "entry 0 must be an object"),
        ([{"label": "no id"}], 'entry 0 needs an "id"'),
        ([{"id": "gpt-4o"}], "provider:model"),
        ([{"id": "openai:"}], "provider:model"),
        ([{"id": "openai:a", "efforts": ["minimal"]}], 'unknown effort "minimal" for openai:'),
        ([{"id": "openai:b", "hidden": True}], "only hides a built-in model"),
        ([{"id": "openai:gpt-5.6-sol", "hidden": True, "label": "x"}], "takes no other field"),
        ([{"id": "openai:gpt-5.6-sol", "hidden": "yes"}], '"hidden" must be true or false'),
        ([{"id": "openai:a"}, {"id": "openai:a"}], "listed twice"),
        ([{"id": "openai:a", "labell": "typo"}], 'unknown field "labell"'),
        ([{"id": "openai:a", "efforts": []}], "at least one effort"),
        ([{"id": "openai:a", "efforts": ["turbo"]}], 'unknown effort "turbo"'),
        ([{"id": "openai:a", "efforts": ["low"], "default_effort": "high"}], "default_effort"),
        ([{"id": "openai:a", "context_window": 0}], "context_window"),
        ([{"id": "openai:a", "context_window": "200k"}], "context_window"),
        ([{"id": "openai:a", "supports_images": "yes"}], "supports_images"),
        ([{"id": "openai:a", "label": 3}], "label"),
    ],
)
def test_invalid_entries_are_rejected_with_the_file_named(
    tmp_path: Path, payload: object, fragment: str
) -> None:
    path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match=fragment) as excinfo:
        load_extra_models(path)
    assert str(path) in str(excinfo.value)


def test_missing_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "nope.json"
    with pytest.raises(ValueError, match="does not exist") as excinfo:
        load_extra_models(path)
    assert str(path) in str(excinfo.value)


def test_malformed_json_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "extra-models.json"
    path.write_text("[{'id': 'openai:a'}]", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON") as excinfo:
        load_extra_models(path)
    assert str(path) in str(excinfo.value)


ModelChoice = tuple[str | None, str | None]


def _options_with_env(monkeypatch: pytest.MonkeyPatch, path: str) -> Mapping[str, object]:
    """Module globals of ``options.py`` evaluated with the extra-models file set."""
    monkeypatch.setenv("OPEN_SWE_EXTRA_MODELS_FILE", path)
    return runpy.run_path(options.__file__)


def _supported_ids(loaded: Mapping[str, object]) -> frozenset[str]:
    return cast(frozenset[str], loaded["SUPPORTED_MODEL_IDS"])


def _supported_models(loaded: Mapping[str, object]) -> Sequence[options.ModelOption]:
    return cast(Sequence[options.ModelOption], loaded["SUPPORTED_MODELS"])


def test_extra_models_file_extends_the_supported_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path, [{"id": "openai:my-model", "context_window": 131_072}])
    loaded = _options_with_env(monkeypatch, str(path))
    assert "openai:my-model" in _supported_ids(loaded)
    catalog = _supported_models(loaded)
    assert catalog[-1]["id"] == "openai:my-model"
    # The context window lives in the profile override, not on the catalog entry.
    assert "context_window" not in catalog[-1]
    normalize = cast(Callable[[object, object], ModelChoice], loaded["normalize_model_choice"])
    assert normalize("openai:my-model", "low") == ("openai:my-model", "low")
    context_window = cast(Callable[[str], int | None], loaded["model_profile_context_window"])
    assert context_window("openai:my-model") == 131_072
    enrich = cast(
        Callable[[Sequence[options.ModelOption]], list[options.ModelOption]],
        loaded["models_with_profile_context_windows"],
    )
    assert enrich(catalog)[-1]["context_window"] == 131_072


def test_extra_models_file_expands_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, [{"id": "openai:home-model"}])
    monkeypatch.setenv("HOME", str(tmp_path))
    loaded = _options_with_env(monkeypatch, "~/extra-models.json")
    assert "openai:home-model" in _supported_ids(loaded)


def test_unset_env_keeps_the_builtin_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPEN_SWE_EXTRA_MODELS_FILE", raising=False)
    loaded = runpy.run_path(options.__file__)
    assert _supported_ids(loaded) == cast(frozenset[str], loaded["BUILTIN_MODEL_IDS"])


def test_efforts_are_validated_per_provider(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            {"id": "google_genai:gemini-x", "efforts": ["minimal", "high"]},
            {"id": "groq:llama-x", "efforts": ["none", "max"]},
        ],
    )
    loaded = load_extra_models(path).models
    assert loaded[0]["efforts"] == ["minimal", "high"]
    assert loaded[1]["efforts"] == ["none", "max"]


def test_builtin_entry_is_overridden_field_by_field(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [{"id": "openai:gpt-5.6-sol", "label": "GPT-5.6 Sol (LiteLLM)", "context_window": 200_000}],
    )
    (override,) = load_extra_models(path).models
    assert override["label"] == "GPT-5.6 Sol (LiteLLM)"
    assert override["context_window"] == 200_000
    builtin = next(m for m in options.BUILTIN_MODELS if m["id"] == "openai:gpt-5.6-sol")
    assert override["efforts"] == builtin["efforts"]
    assert override["default_effort"] == builtin["default_effort"]
    assert override["supports_images"] == builtin["supports_images"]


def test_hidden_entries_name_builtin_models(tmp_path: Path) -> None:
    path = _write(tmp_path, [{"id": "anthropic:claude-opus-5", "hidden": True}])
    parsed = load_extra_models(path)
    assert parsed.models == []
    assert parsed.hidden == frozenset({"anthropic:claude-opus-5"})


def test_override_and_hide_reshape_the_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(
        tmp_path,
        [
            {"id": "openai:gpt-5.6-sol", "hidden": True},
            {"id": "openai:gpt-6-astra", "label": "Astra via LiteLLM", "context_window": 100_000},
            {"id": "openai:qwen3-coder"},
        ],
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL_ID", raising=False)
    loaded = _options_with_env(monkeypatch, str(path))
    ids = _supported_ids(loaded)
    assert "openai:gpt-5.6-sol" not in ids
    assert {"openai:gpt-6-astra", "openai:qwen3-coder"} <= ids
    astra = next(m for m in _supported_models(loaded) if m["id"] == "openai:gpt-6-astra")
    assert astra["label"] == "Astra via LiteLLM"
    context_window = cast(Callable[[str], int | None], loaded["model_profile_context_window"])
    assert context_window("openai:gpt-6-astra") == 100_000
    # An overridden built-in stays a built-in; only new ids count as extra models.
    assert cast(frozenset[str], loaded["EXTRA_MODEL_IDS"]) == frozenset({"openai:qwen3-coder"})
    # The hidden built-in default gives way to the next selectable model of its provider.
    assert loaded["DEFAULT_MODEL_ID"] == "openai:gpt-6-astra"
