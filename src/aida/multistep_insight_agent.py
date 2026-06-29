from __future__ import annotations

import json
from typing import Any

from .smolagents_utils import require_module


SmolTool = require_module("smolagents").Tool
ToolCallingAgent = require_module("smolagents").ToolCallingAgent
pd = require_module("pandas")


PROMPT = """
You are a curious multi-step insight generation agent.

Goal:
{goal}

Previous round context ({prior_context_label}):
{prior_context_json}

You must follow this process:

1. Call `get_dataset_schema` to inspect the dataset.
2. If text columns exist, inspect/summarize examples from them.
3. Generate {questions_per_round} curious analysis questions.
4. Use `run_python` for all analysis needed to answer the questions.
5. Call `submit_insights` exactly once with final questions and evidence-backed insights.

The `run_python` tool can be used for all analysis tasks, including:
- aggregations
- percentages
- comparisons
- correlations
- trend checks
- segment analysis
- text-field analysis
- keyword searches in descriptions
- validation checks
- outlier detection

Question rules:
- Ask benchmark-style questions, not vague profiling questions.
- Prefer questions that search for imbalance, named outliers, trends, spikes, correlations, hidden text drivers, or disproven causes.
- Bad question: "What is the average resolution time?"
- Good question: "Is one category responsible for the unusually high resolution time?"
- Bad question: "How are incidents distributed?"
- Good question: "Is incident volume concentrated in one category, location, agent, or caller?"
- Include at least one question about free-text fields if text columns exist.

Question Suggestions: {suggested_questions_json}


Insight rules:
- Return only materially new insights.
- Do not return generic observations.
- Do not return data gaps as insights.
- Do not say only that something "varies" or "differs"; identify what stands out.
- Each insight should name the specific segment, metric, and direction.
- Each insight must be supported by quantified evidence.
- Prefer insights that match these forms:
  - "[Segment] is significantly higher/lower than [comparison group]."
  - "[Metric] is increasing/decreasing over time, especially for [segment]."
  - "[Specific entity] is the main outlier for [metric]."
  - "[Text theme] is the dominant hidden driver inside [category]."
  - "[Suspected driver] is unlikely because [metric] is uniform across [groups]."

Evidence rules:
- Evidence must summarize the result, not the method.
- Evidence must include actual numbers, percentages, comparisons, rankings, correlations, or time windows.
- Do not write "SQL query shows", "Python shows", "query result", or "analysis shows".
- Good evidence: "Hardware accounts for 336 of 500 incidents, or 67.2%; the next largest category is Network with 51."
- Bad evidence: "A query counted incidents by category."

Do not return free text before the final tool call.
""".strip()

class SubmitInsightsTool(SmolTool):
    name = "submit_insights"
    description = "Submit final generated questions and evidence-backed insights."
    inputs = {
        "questions": {
            "type": "array",
            "description": "Analysis questions answered during the run.",
            "items": {"type": "string"},
        },
            "insights": {
                "type": "array",
                "description": "Concise Data Driven insights, materially new insights. Do not include data gaps, monitoring gaps, or generic observations.",
                "items": {
                    "type": "object",
                    "properties": {
                        "insight": {
                            "type": "string",
                            "description": "A concise business insight based on dataset evidence."
                        },
                        "evidence": {
                            "type": "string",
                            "description": (
                                "Concise summary of the computed result, including actual numbers, percentages, "
                                "comparisons, rankings, or trends. Do not mention SQL, Python, queries, tools, "
                                "or analysis method. Evidence should describe the finding, not how it was computed."
                            ),},                        "question_answered": {
                            "type": "string",
                            "description": "The analysis question this insight answers."
                        }
                    },
                    "required": ["insight", "evidence", "question_answered"]
                }
}    }
    output_type = "object"

    def __init__(self):
        super().__init__()
        self.last_result: dict[str, Any] | None = None

    def forward(self, questions: list[str], insights: list[dict[str, str]]) -> dict[str, Any]:
        self.last_result = {
            "questions": [{"question": question} for question in questions],
            "insights": insights,
        }
        return self.last_result

class SampleTextFieldsTool(SmolTool):
    name = "sample_text_fields"
    description = """
    Sample free-text fields from the dataset so the agent can discover hidden themes,
    repeated issue types, failure modes, user complaints, or operational patterns.

    Use this before generating questions when the dataset has text columns.
    """

    inputs = {
        "columns": {
            "type": "array",
            "description": (
                "Text columns to sample, such as description, short_description, "
                "comments, notes, summary, or resolution_notes."
            ),
            "items": {"type": "string"},
        },
        "sample_size": {
            "type": "integer",
            "description": "Number of non-empty rows to sample. Usually 10 to 30.",
            "nullable": True,
        },
    }

    output_type = "object"

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df

    def forward(
        self,
        columns: list[str],
        sample_size: int = 20,
    ) -> dict[str, Any]:
        valid_columns = [col for col in columns if col in self.df.columns]

        if not valid_columns:
            return {
                "success": False,
                "message": "No valid text columns found.",
                "available_columns": list(self.df.columns),
                "samples": [],
            }

        text_df = self.df[valid_columns].copy()

        # Keep rows where at least one selected text column is non-empty
        mask = text_df.fillna("").astype(str).apply(
            lambda row: any(value.strip() for value in row),
            axis=1,
        )

        text_df = text_df[mask]

        if text_df.empty:
            return {
                "success": False,
                "message": "Selected text columns are empty.",
                "columns": valid_columns,
                "samples": [],
            }

        sample_size = min(sample_size, len(text_df))

        sampled = text_df.sample(
            n=sample_size,
            random_state=42,
        )

        samples = []

        for idx, row in sampled.iterrows():
            item = {"row_index": int(idx)}
            for col in valid_columns:
                value = row[col]
                if pd.isna(value):
                    value = ""
                item[col] = str(value)[:500]
            samples.append(item)

        return {
            "success": True,
            "columns": valid_columns,
            "sample_size": sample_size,
            "samples": samples,
        }

def build_prompt(
    goal: str,
    prior_context: list[dict[str, Any]],
    questions_per_round: int,
    suggested_questions: list[str] | None = None,
    good_insight_example: str | None = None,
    prior_context_label: str = "generated insights",
) -> str:
    return PROMPT.format(
        goal=goal,
        prior_context_label=prior_context_label,
        prior_context_json=json.dumps(prior_context, indent=2),
        suggested_questions_json=json.dumps(suggested_questions or [], indent=2),
        good_insight_example=json.dumps(good_insight_example or ""),
        questions_per_round=questions_per_round,
    )


def run_multistep_insight_agent(
    model: Any,
    tools: list[Any],
    goal: str,
    prior_context: list[dict[str, Any]],
    questions_per_round: int,
    verbosity_level: int = 0,
    dataframe: Any | None = None,
    suggested_questions: list[str] | None = None,
    good_insight_example: str | None = None,
    prior_context_label: str = "generated insights",
) -> tuple[str, dict[str, Any], str]:
    submit_tool = SubmitInsightsTool()
    agent_tools = [*tools]
    if dataframe is not None:
        agent_tools.append(SampleTextFieldsTool(dataframe))
    agent = ToolCallingAgent(
        model=model,
        tools=[*agent_tools, submit_tool],
        verbosity_level=verbosity_level,
        name="multistep_insight_agent",
        description="Generates analysis questions, uses dataset tools, and submits evidence-backed insights.",
        max_steps=12,
    )
    prompt = build_prompt(
        goal,
        prior_context,
        questions_per_round,
        suggested_questions,
        good_insight_example,
        prior_context_label=prior_context_label,
    )
    raw = agent.run(prompt)
    if submit_tool.last_result is None:
        retry_prompt = (
            prompt
            + "\n\nYou already performed some analysis but did not finish correctly."
            + "\nCall `submit_insights` exactly once now using the evidence you have already gathered."
            + "\nDo not do more exploration unless strictly necessary."
            + "\nDo not return free text."
        )
        retry_raw = agent.run(retry_prompt)
        if submit_tool.last_result is None:
            raw_text = f"{raw}\n\n--- RETRY ---\n\n{retry_raw}"
            prompt_excerpt = retry_prompt[:3000]
            raise ValueError(
                "Insight agent did not call submit_insights after retry.\n\n"
                f"Raw agent output:\n{raw_text}\n\n"
                f"Prompt excerpt:\n{prompt_excerpt}"
            )
        return f"{raw}\n\n--- RETRY ---\n\n{retry_raw}", submit_tool.last_result, retry_prompt
    return str(raw), submit_tool.last_result, prompt
