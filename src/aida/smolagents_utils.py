from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
from typing import Any, Iterable


REVIEWER_FEEDBACK_KEYS = ("areas_to_cover", "questions_to_answer")

def require_module(module_name: str, package_name: str | None = None):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover
        package = package_name or module_name
        raise SystemExit(
            f"Missing dependency: {package}. Install the package dependencies first."
        ) from exc


def resolve_dataset_path(root: Path, raw_path: str) -> Path:
    normalized_raw_path = raw_path.strip()
    replacement_rules = [
        ("/FiveThirtyEight/data", "data/fivethirtyeight/data"),
        ("FiveThirtyEight/data", "data/fivethirtyeight/data"),
        ("/FiveThirtyEight", "data/fivethirtyeight"),
        ("FiveThirtyEight", "data/fivethirtyeight"),
        ("insight-bench/data/notebooks/csvs", "data/InsightBench/notebooks/csvs"),
        ("insight-bench/data/notebooks", "data/InsightBench/notebooks"),
        ("insight-bench/data", "data/InsightBench/notebooks"),
        ("insight-bench", "data/InsightBench"),
    ]

    candidates = [
        root / normalized_raw_path,
        root / normalized_raw_path.lstrip("/"),
        root / "public" / normalized_raw_path.lstrip("/"),
        root / "thesis" / normalized_raw_path.lstrip("/"),
    ]
    for source, target in replacement_rules:
        if source in normalized_raw_path:
            replaced = normalized_raw_path.replace(source, target).lstrip("/")
            candidates.extend(
                [
                    root / replaced,
                    root / "public" / replaced,
                    root / "thesis" / replaced,
                ]
            )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Could not resolve dataset path from SSOT entry: {raw_path}")


def resolve_dataset_paths(root: Path, raw_paths: str | Iterable[str]) -> list[Path]:
    if isinstance(raw_paths, str):
        return [resolve_dataset_path(root, raw_paths)]
    return [resolve_dataset_path(root, str(raw_path)) for raw_path in raw_paths]


def clean_json_block(raw_text: str) -> str:
    text = raw_text.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_json_output(raw_text: str) -> dict[str, Any]:
    text = clean_json_block(raw_text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed, _ = json.JSONDecoder().raw_decode(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except Exception as exc:
                raise ValueError(f"Model did not return valid JSON:\n{text}") from exc
        except Exception as exc:
            raise ValueError(f"Model did not return valid JSON:\n{text}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Model returned a non-dict payload:\n{text}")
    return parsed


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def empty_reviewer_feedback() -> dict[str, list[str]]:
    return {key: [] for key in REVIEWER_FEEDBACK_KEYS}


def normalize_reviewer_feedback(feedback: Any) -> dict[str, list[str]]:
    if not isinstance(feedback, dict):
        return empty_reviewer_feedback()
    return {
        key: [str(item) for item in feedback.get(key, []) if str(item).strip()]
        for key in REVIEWER_FEEDBACK_KEYS
    }


def extract_insight_texts(insights: list[dict[str, Any]]) -> list[str]:
    return [
        insight["insight"]
        for insight in insights
        if isinstance(insight, dict) and "insight" in insight
    ]


def normalize_insights_input(insights: Any) -> list[dict[str, Any]]:
    if isinstance(insights, dict):
        return [insights]
    if isinstance(insights, list):
        return [item for item in insights if isinstance(item, dict)]
    raise TypeError("insights must be a dict or a list of dicts")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def build_recall_judge_prompt(expected_insights: list[str], candidate_insights: list[str]) -> str:
    return f"""
You are evaluating recall for generated insights against expected reference insights.

Expected insights:
{json.dumps(expected_insights, indent=2)}

Candidate insights:
{json.dumps(candidate_insights, indent=2)}

Task:
- For each expected insight, decide whether at least one candidate insight captures the same business meaning.
- Match by semantic meaning, not wording.
- Use the single best candidate for each expected insight.
- Mark `matched` as false when the candidate is only vaguely related or misses the key claim.

Return JSON only with this shape:
{{
  "matches": [
    {{
      "expected_insight": "string",
      "matched": true,
      "best_candidate": "string",
      "rationale": "string"
    }}
  ]
}}
""".strip()


def score_recall_with_llm(model: Any, expected_insights: list[str], candidate_insights: list[str]) -> dict[str, Any]:
    prompt = build_recall_judge_prompt(expected_insights, candidate_insights)
    message = model.generate(
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    payload = parse_json_output(str(message.content))
    matches = payload.get("matches", [])
    matched_count = sum(1 for item in matches if item.get("matched"))
    total_expected = len(expected_insights)
    recall = matched_count / total_expected if total_expected else 0.0
    return {
        "matched_expected_count": matched_count,
        "total_expected_count": total_expected,
        "recall": round(recall, 4),
        "matches": matches,
    }


def build_ssot_annotation_prompt(expected_insights: list[str], generated_insights: list[dict[str, Any]]) -> str:
    expected_payload = [
        {"ssot_index": index, "insight": insight}
        for index, insight in enumerate(expected_insights)
    ]
    generated_payload = [
        {
            **insight,
            "insight_for_matching": str(insight.get("evidence", "")).strip(),
        }
        for insight in generated_insights
    ]
    return f"""
You are annotating generated insights against SSOT reference insights.

SSOT insights:
{json.dumps(expected_payload, indent=2)}

Generated insights:
{json.dumps(generated_payload, indent=2)}

Task:
- For each generated insight, decide whether `insight_for_matching` matches one or more SSOT insights by semantic meaning.
- If it matches, return `in_ssot: true` and the matching `ssot_indices`.
- If it does not match any SSOT insight, return `in_ssot: false` and `ssot_indices: []`.
- Match on business meaning, not wording.

Return JSON only with this shape:
{{
  "annotations": [
    {{
      "insight_for_matching": "string",
      "in_ssot": true,
      "ssot_indices": [0]
    }}
  ]
}}
""".strip()


def annotate_insights_with_ssot(
    model: Any,
    expected_insights: list[str],
    generated_insights: list[dict[str, Any]],
    log_path: Path | None = None,
) -> list[dict[str, Any]]:
    prompt = build_ssot_annotation_prompt(expected_insights, generated_insights)
    message = model.generate(
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    raw_output = str(message.content)
    try:
        payload = parse_json_output(raw_output)
    except Exception as exc:
        if log_path is not None:
            write_json(
                log_path,
                {
                    "prompt": prompt,
                    "raw_output": raw_output,
                    "parse_error": str(exc),
                },
            )
        raise
    if log_path is not None:
        write_json(
            log_path,
            {
                "prompt": prompt,
                "raw_output": raw_output,
                "parsed_output": payload,
            },
        )
    annotations = payload.get("annotations", [])
    annotation_by_text = {
        item.get("insight_for_matching", ""): {
            "in_ssot": bool(item.get("in_ssot")),
            "ssot_indices": [
                index
                for index in item.get("ssot_indices", [])
                if isinstance(index, int) and 0 <= index < len(expected_insights)
            ],
        }
        for item in annotations
    }

    def annotate_one(insight: dict[str, Any]) -> dict[str, Any]:
        text = str(insight.get("evidence", "")).strip()
        annotation = annotation_by_text.get(text, {})
        ssot_indices = annotation.get("ssot_indices", [])
        return {
            **insight,
            "in_ssot": annotation.get("in_ssot", False),
            "ssot_indices": ssot_indices,
            "ssot_insights": [expected_insights[index] for index in ssot_indices],
        }

    return [annotate_one(insight) for insight in generated_insights]
