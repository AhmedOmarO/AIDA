from __future__ import annotations

import os
from .smolagents_utils import require_module


def normalize_model_name(model_name: str) -> tuple[str, str]:
    normalized = model_name.strip()
    explicit_prefixes = {"gemini/": "gemini", "openai/": "openai", "deepseek/": "deepseek"}

    for prefix, provider in explicit_prefixes.items():
        if normalized.startswith(prefix):
            return provider, normalized
    if normalized.startswith("models/"):
        return "gemini", f"gemini/{normalized.split('/', 1)[1]}"
    if normalized.startswith("gemini-"):
        return "gemini", f"gemini/{normalized}"
    if normalized.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai", f"openai/{normalized}"
    if "/" in normalized:
        provider = normalized.split("/", 1)[0]
        return provider, normalized

    raise SystemExit(
        "Could not infer the LLM provider from model name "
        f"'{model_name}'. Use a LiteLLM-style name like "
        "'gemini/gemini-flash-lite-latest' or 'openai/gpt-4.1-mini'."
    )


def resolve_api_key(provider: str) -> str:
    env_var_by_provider = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    if provider not in env_var_by_provider:
        raise SystemExit(f"Unsupported provider '{provider}'.")

    env_var = env_var_by_provider[provider]
    api_key = os.getenv(env_var)
    if not api_key:
        raise SystemExit(f"Set {env_var} in the environment before running this script.")
    return api_key


def build_model(model_name: str):
    smolagents = require_module("smolagents")
    provider, normalized_model_name = normalize_model_name(model_name)
    model_kwargs = {
        "model_id": normalized_model_name,
        "api_key": resolve_api_key(provider),
    }
    if provider == "deepseek":
        model_kwargs["api_base"] = "https://api.deepseek.com"
        model_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return smolagents.LiteLLMModel(**model_kwargs)
