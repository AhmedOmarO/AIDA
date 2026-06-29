from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any


def load_prompt_config(prompts_path: Path | str) -> dict[str, Any]:
    return json.loads(Path(prompts_path).read_text())


def build_code_agent_json_instruction(json_shape: str) -> str:
    return ""


def render_agent_prompt(prompts: dict[str, Any], agent_key: str, **values: Any) -> str:
    config = prompts[agent_key]
    return config["template"].format(
        json_shape=config["json_shape"],
        code_agent_instruction=build_code_agent_json_instruction(config["code_agent_example"]),
        **values,
    ).strip()
