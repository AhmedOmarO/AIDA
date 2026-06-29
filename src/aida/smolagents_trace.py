from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .smolagents_context import DatasetContext


def init_trace(context: DatasetContext, model_name: str | None) -> dict[str, Any]:
    return {
        "flag_id": context.flag_id,
        "goal": context.goal,
        "dataset_path": context.dataset_path_value,
        "model_name": model_name or "unknown",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "calls": [],
    }


def append_trace_call(trace: dict[str, Any], **call: Any) -> None:
    trace["calls"].append(copy.deepcopy(call))


def write_trace_logs(trace: dict[str, Any], log_dir: Path, flag_id: str) -> dict[str, str]:
    def block(text: str, language: str = "") -> str:
        return f"```{language}\n{text}\n```"

    log_dir.mkdir(parents=True, exist_ok=True)
    json_path = log_dir / f"{flag_id}_llm_trace.json"
    md_path = log_dir / f"{flag_id}_llm_trace.md"
    json_path.write_text(json.dumps(trace, indent=2) + "\n")
    lines = [f"# LLM Trace for {flag_id}", "", f"- Generated at: {trace['generated_at']}", f"- Model: `{trace['model_name']}`", f"- Goal: {trace['goal']}", ""]
    for call in trace["calls"]:
        lines += [
            f"## Round {call['round']} - {call['agent_name']}",
            "",
            f"- Call index: `{call['call_index']}`",
            f"- Agent role: `{call['agent_role']}`",
            "",
            "### Logical Input",
            "",
            block(json.dumps(call["logical_input"], indent=2), "json"),
            "",
            "### Prompt",
            "",
            block(call["prompt"], "text"),
            "",
            "### Raw Output",
            "",
            block(call["raw_output"], "json"),
            "",
            "### Parsed Output",
            "",
            block(json.dumps(call["parsed_output"], indent=2), "json"),
            "",
        ]
    md_path.write_text("\n".join(lines))
    return {"json": str(json_path), "markdown": str(md_path)}
