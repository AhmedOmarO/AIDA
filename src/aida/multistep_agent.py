from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------
# This assumes you already have these helpers in your project,
# as in your original code.
from .smolagents_models import build_model
from .progress import progress
from .smolagents_utils import parse_json_output, require_module


# ---------------------------------------------------------------------
# Smolagents imports through your require_module helper
# ---------------------------------------------------------------------
SmolTool = require_module("smolagents").Tool
ToolCallingAgent = require_module("smolagents").ToolCallingAgent


TOOLS_JSON_PATH = Path(__file__).with_name("multistep_agent_tools.json")
with TOOLS_JSON_PATH.open() as f:
    TOOL_SCHEMAS = {
        tool["name"]: tool for tool in json.load(f)["tools"]
    }


def get_tool_schema(name: str) -> dict[str, Any]:
    try:
        return TOOL_SCHEMAS[name]
    except KeyError as exc:
        raise ValueError(f"Missing tool schema for {name!r}") from exc


# ---------------------------------------------------------------------
# Small local helper
# ---------------------------------------------------------------------
def normalize_insights_input(
    insights: dict[str, Any] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Converts either:
    - one insight dict
    - or a list of insight dicts

    into a list of insight dicts.
    """
    if isinstance(insights, dict):
        return [insights]

    if isinstance(insights, list):
        return insights

    raise TypeError("insights must be a dict or a list of dicts")


# ---------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------
MULTISTEP_REVIEWER_PROMPT = """
You are agent 3, the reviewer.

Goal:
{goal}

Current insight:
{insight_json}

You must review the current insight in multiple steps.

Follow this exact process:

Step 1:
Call `assess_business_impact`.


Step 2:
Call `assess_actionability`.

Step 3:
Call `NoveltyAssessmentTool`.

Step 4:
Call `submit_final_review`.

Rules:
- Do not skip steps.
- Use the previous tool outputs when making later decisions.
- In `assess_business_impact`, score `importance` and provide `importance_rationale`.
- In `assess_actionability`, recommend 2 to 3 concrete actions.
- Include those recommended actions in `submit_final_review`.
- In `submit_final_review`, include `importance_rationale` and `interestingness_rationale`.
- If an insight is not useful for the goal, mark it as not_relevant.
- Novelty should measure what new information, specificity, evidence, or interpretation the insight adds beyond the seen-insight baseline.
- Do not use redundancy as a decision criterion in this review.
- Call `submit_final_review` exactly once.
- Do not return markdown.
- Do not return free text before the final review.
""".strip()


RESEARCH_QUESTION_PROMPT = """
You are a follow-up research question agent.

Goal:
{goal}

Current reviewed insight:
{review_json}

All reviewed insights:
{all_reviews_json}

Previously selected insights:
{selected_insights_json}

Previously prioritized questions:
{prioritized_questions_json}

Your job is to propose follow-up research questions only when the insight is important or interesting enough to justify deeper work.
The purpose of the questions is to turn promising insights into more actionable insights.

Follow this exact process:

Step 1:
Decide whether the insight is worth follow-up based on its importance, interestingness, relevance, and existing recommended actions.

Step 2:
Check whether each candidate research question is already answered by one of the reviewed insights in `All reviewed insights`.
Remove any question that is already answered or materially duplicated there.

Step 3:
If it is worth follow-up, call `submit_research_questions` once with 2 to 4 concrete research questions.
If it is not worth follow-up, call `submit_research_questions` once with an empty `research_questions` list and explain why in `priority_reason`.

Step 4:
Call `prioritize_research_questions` once.
Merge the new batch with `Previously prioritized questions`, rank the combined set by priority, deduplicate overlapping questions, and keep only the top 3.

Rules:
- Prefer insights that are relevant and either important or interesting.
- Use research questions to close gaps that block direct action.
- Do not ask a research question if the answer is already present in another reviewed insight.
- Questions must be specific and answerable from data, analysis, or targeted operational investigation.
- `prioritize_research_questions` must always return exactly 3 questions unless fewer than 3 valid questions exist in the combined pool.
- Call `submit_research_questions` exactly once.
- Call `prioritize_research_questions` exactly once.
- Do not return markdown.
- Do not return free text before the final tool call.
""".strip()


BATCH_RESEARCH_QUESTION_PROMPT = """
You are a follow-up research question agent.

Goal:
{goal}

Reviewed insights:
{reviews_json}

Previously selected insights:
{selected_insights_json}

Previously prioritized questions:
{prioritized_questions_json}

Your job is to propose follow-up research questions only when an insight is important or interesting enough to justify deeper work.
The purpose of the questions is to turn promising insights into more actionable insights.

For each reviewed insight:
- decide whether it is worth follow-up based on importance, interestingness, relevance, and existing recommended actions
- if it is worth follow-up, return 2 to 4 concrete research questions
- if it is not worth follow-up, return an empty `research_questions` list and explain why in `priority_reason`
- do not ask a question that is already answered by another reviewed insight

Then:
- merge the new questions with `Previously prioritized questions`
- deduplicate overlapping questions
- rank the combined set by priority
- return exactly the top 3 questions unless fewer than 3 valid questions exist

Return JSON only with this shape:
{{
  "research_payload": [
    {{
      "insight": "string",
      "importance": 1,
      "interestingness": 1,
      "priority_reason": "string",
      "research_questions": ["string"]
    }}
  ],
  "prioritized_questions_payload": {{
    "prioritized_questions": ["string"],
    "priority_rationale": "string"
  }}
}}
""".strip()


BATCH_REVIEW_PROMPT = """
You are agent 3, the reviewer.

Goal:
{goal}

Insights:
{insights_json}

Review each insight in order and return exactly one final review per insight.

For each insight:
- score `importance` from 1 to 10 and provide `importance_rationale`
- score `actionability` from 1 to 10 and provide 2 to 3 `recommended_actions` when direct action is possible, otherwise return []
- score `interestingness` from 1 to 10 and provide `interestingness_rationale`
- set `relevance_flag` to either "relevant" or "not_relevant"
- provide a short final `rationale`
- use `user_knowledge_baseline` to summarize what is already known from earlier reviewed insights
- if there are no earlier reviewed insights for an item, use an empty list

Rules:
- Do not drop, merge, or reorder insights.
- Return one review object for every input insight.
- Keep the output JSON only.

Return JSON with this shape:
{{
  "reviews": [
    {{
      "insight": "string",
      "evidence": "string",
      "importance": 1,
      "importance_rationale": "string",
      "actionability": 1,
      "recommended_actions": ["string"],
      "interestingness": 1,
      "interestingness_rationale": "string",
      "relevance_flag": "relevant",
      "rationale": "string",
      "user_knowledge_baseline": ["string"]
    }}
  ]
}}
""".strip()


# ---------------------------------------------------------------------
# Step 1 tool: novelty
# ---------------------------------------------------------------------
class NoveltyAssessmentTool(SmolTool):
    schema = get_tool_schema("assess_novelty")
    name = schema["name"]
    description = schema["description"]
    inputs = schema["inputs"]
    output_type = schema["output_type"]

    def __init__(self):
        super().__init__()
        self.last_result: dict[str, Any] | None = None

    def forward(
        self,
        insight: str,
        adds_new_information: bool,
        novelty_score: int,
        new_information_added: str,
        rationale: str,
        user_knowledge_baseline: list[str]
    ) -> dict[str, Any]:
        result = {
            "insight": insight,
            "adds_new_information": adds_new_information,
            "novelty_score": novelty_score,
            "new_information_added": new_information_added,
            "rationale": rationale,
            "user_knowledge_baseline": user_knowledge_baseline
        }

        self.last_result = result
        return result

# ---------------------------------------------------------------------
# Step 2 tool: business impact
# ---------------------------------------------------------------------
class AssessBusinessImpactTool(SmolTool):
    schema = get_tool_schema("assess_business_impact")
    name = schema["name"]
    description = schema["description"]
    inputs = schema["inputs"]
    output_type = schema["output_type"]

    def __init__(self):
        super().__init__()
        self.last_result: dict[str, Any] | None = None

    def forward(
        self,
        insight: str,
        importance: int,
        importance_rationale: str,
    ) -> dict[str, Any]:
        result = {
            "insight": insight,
            "importance": importance,
            "importance_rationale": importance_rationale,
        }

        self.last_result = result
        return result


# ---------------------------------------------------------------------
# Step 3 tool: actionability
# ---------------------------------------------------------------------
class AssessActionabilityTool(SmolTool):
    schema = get_tool_schema("assess_actionability")
    name = schema["name"]
    description = schema["description"]
    inputs = schema["inputs"]
    output_type = schema["output_type"]

    def __init__(self):
        super().__init__()
        self.last_result: dict[str, Any] | None = None

    def forward(
        self,
        insight: str,
        actionability: int,
        recommended_actions: list[str],
        rationale: str,
    ) -> dict[str, Any]:
        result = {
            "insight": insight,
            "actionability": actionability,
            "recommended_actions": recommended_actions,
            "rationale": rationale,
        }

        self.last_result = result
        return result


# ---------------------------------------------------------------------
# Step 4 tool: final structured review
# ---------------------------------------------------------------------
class SubmitFinalReviewTool(SmolTool):
    schema = get_tool_schema("submit_final_review")
    name = schema["name"]
    description = schema["description"]
    inputs = schema["inputs"]
    output_type = schema["output_type"]

    def __init__(self):
        super().__init__()
        self.last_review: dict[str, Any] | None = None

    def forward(
        self,
        insight: str,
        evidence: str,
        importance: int,
        importance_rationale: str,
        actionability: int,
        recommended_actions: list[str],
        interestingness: int,
        interestingness_rationale: str,
        relevance_flag: str,
        rationale: str,
        # redundant_with_seen_insights: bool,
        # redundant_with: str,
        user_knowledge_baseline: list[str]
    ) -> dict[str, Any]:
        review = {
            "insight": insight,
            "evidence": evidence,
            "importance": importance,
            "importance_rationale": importance_rationale,
            "actionability": actionability,
            "recommended_actions": recommended_actions,
            "interestingness": interestingness,
            "interestingness_rationale": interestingness_rationale,
            "relevance_flag": relevance_flag,
            "rationale": rationale,
            # "redundant_with_seen_insights": redundant_with_seen_insights,
            # "redundant_with": redundant_with,
            # "user_knowledge_baseline": user_knowledge_baseline,
        }

        self.last_review = review
        return review


class SubmitResearchQuestionsTool(SmolTool):
    schema = get_tool_schema("submit_research_questions")
    name = schema["name"]
    description = schema["description"]
    inputs = schema["inputs"]
    output_type = schema["output_type"]

    def __init__(self):
        super().__init__()
        self.last_result: dict[str, Any] | None = None

    def forward(
        self,
        insight: str,
        importance: int,
        interestingness: int,
        priority_reason: str,
        research_questions: list[str],
    ) -> dict[str, Any]:
        result = {
            "insight": insight,
            "importance": importance,
            "interestingness": interestingness,
            "priority_reason": priority_reason,
            "research_questions": research_questions,
        }

        self.last_result = result
        return result


class PrioritizeResearchQuestionsTool(SmolTool):
    schema = get_tool_schema("prioritize_research_questions")
    name = schema["name"]
    description = schema["description"]
    inputs = schema["inputs"]
    output_type = schema["output_type"]

    def __init__(self):
        super().__init__()
        self.last_result: dict[str, Any] | None = None

    def forward(
        self,
        prioritized_questions: list[str],
        priority_rationale: str,
    ) -> dict[str, Any]:
        result = {
            "prioritized_questions": prioritized_questions,
            "priority_rationale": priority_rationale,
        }

        self.last_result = result
        return result


# ---------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------
def build_multistep_reviewer_prompt(
    goal: str,
    insight: dict[str, Any],
    seen_insights: list[str] | None = None,
) -> str:
    reviewer_visible_insight = {
        key: value
        for key, value in insight.items()
        if key not in {"in_ssot", "ssot_indices", "ssot_insights"}
    }
    return MULTISTEP_REVIEWER_PROMPT.format(
        goal=goal.strip(),
        insight_json=json.dumps(reviewer_visible_insight, indent=2),
        seen_insights_json=json.dumps(seen_insights or [], indent=2),
    )


def build_batch_reviewer_prompt(
    goal: str,
    insights: list[dict[str, Any]],
    seen_insights: list[str] | None = None,
) -> str:
    memory = list(seen_insights or [])
    payload: list[dict[str, Any]] = []
    for insight in insights:
        reviewer_visible_insight = {
            key: value
            for key, value in insight.items()
            if key not in {"in_ssot", "ssot_indices", "ssot_insights"}
        }
        payload.append(
            {
                "insight": reviewer_visible_insight,
                "user_knowledge_baseline": list(memory),
            }
        )
        if isinstance(insight, dict) and "insight" in insight:
            memory.append(str(insight["insight"]))
    return BATCH_REVIEW_PROMPT.format(
        goal=goal.strip(),
        insights_json=json.dumps(payload, indent=2),
    )


def build_research_question_prompt(
    goal: str,
    review: dict[str, Any],
    all_reviews: list[dict[str, Any]],
    selected_insights: list[str] | None = None,
    prioritized_questions: list[str] | None = None,
) -> str:
    return RESEARCH_QUESTION_PROMPT.format(
        goal=goal.strip(),
        review_json=json.dumps(review, indent=2),
        all_reviews_json=json.dumps(all_reviews, indent=2),
        selected_insights_json=json.dumps(selected_insights or [], indent=2),
        prioritized_questions_json=json.dumps(prioritized_questions or [], indent=2),
    )


def build_batch_research_question_prompt(
    goal: str,
    payload: list[dict[str, Any]],
    selected_insights: list[str] | None = None,
    prioritized_questions: list[str] | None = None,
) -> str:
    return BATCH_RESEARCH_QUESTION_PROMPT.format(
        goal=goal.strip(),
        reviews_json=json.dumps(payload, indent=2),
        selected_insights_json=json.dumps(selected_insights or [], indent=2),
        prioritized_questions_json=json.dumps(prioritized_questions or [], indent=2),
    )


# ---------------------------------------------------------------------
# Main reviewer runner
# ---------------------------------------------------------------------
def run_reviewer(
    model: Any,
    goal: str,
    insights: dict[str, Any] | list[dict[str, Any]],
    seen_insights: list[str] | None = None,
    verbosity_level: int = 2,
    trace_collector: list[dict[str, Any]] | None = None,
    batch_review: bool = False,
) -> list[dict[str, Any]]:
    model = build_model(model)

    normalized_insights = normalize_insights_input(insights)

    if batch_review:
        prompt = build_batch_reviewer_prompt(
            goal=goal,
            insights=normalized_insights,
            seen_insights=seen_insights,
        )
        try:
            raw_output = model.generate(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            payload = parse_json_output(str(raw_output.content))
            raw_reviews = payload.get("reviews", [])
            if isinstance(raw_reviews, dict):
                if "reviews" in raw_reviews:
                    raw_reviews = raw_reviews.get("reviews", [])
                else:
                    raw_reviews = list(raw_reviews.values())
            reviews = normalize_insights_input(raw_reviews)
            if len(reviews) != len(normalized_insights):
                raise ValueError(
                    "Batch reviewer did not return one review per insight. "
                    f"expected={len(normalized_insights)} actual={len(reviews)}"
                )
            if trace_collector is not None:
                trace_collector.append(
                    {
                        "agent_name": "multistep_review_agent",
                        "agent_role": "batched insight review",
                        "logical_input": {
                            "goal": goal,
                            "insights": normalized_insights,
                            "seen_insights": list(seen_insights or []),
                            "batch_review": True,
                        },
                        "prompt": prompt,
                        "raw_output": str(raw_output),
                        "parsed_output": reviews,
                    }
                )
            return reviews
        except Exception as exc:
            if trace_collector is not None:
                trace_collector.append(
                    {
                        "agent_name": "multistep_review_agent",
                        "agent_role": "batched insight review fallback",
                        "logical_input": {
                            "goal": goal,
                            "insights": normalized_insights,
                            "seen_insights": list(seen_insights or []),
                            "batch_review": True,
                        },
                        "prompt": prompt,
                        "raw_output": "" if "raw_output" not in locals() else str(raw_output),
                        "parsed_output": {"fallback_to_single": True, "error": str(exc)},
                    }
                )

    memory = list(seen_insights or [])
    reviews: list[dict[str, Any]] = []

    for insight in progress(normalized_insights, desc="Reviewing insights"):
        redundancy_tool = NoveltyAssessmentTool()
        impact_tool = AssessBusinessImpactTool()
        actionability_tool = AssessActionabilityTool()
        final_review_tool = SubmitFinalReviewTool()

        agent = ToolCallingAgent(
            model=model,
            tools=[
                redundancy_tool,
                impact_tool,
                actionability_tool,
                final_review_tool,
            ],
            verbosity_level=verbosity_level,
            name="multistep_review_agent",
            description=(
                "Reviews insights through novelty, business impact, "
                "actionability, and final scoring."
            ),
            max_steps=8,
        )

        prompt = build_multistep_reviewer_prompt(
            goal=goal,
            insight=insight,
            seen_insights=memory,
        )

        raw_output = agent.run(prompt)

        if final_review_tool.last_review is None:
            raise ValueError("Reviewer did not call submit_final_review.")

        review = final_review_tool.last_review
        reviews.append(review)
        if trace_collector is not None:
            trace_collector.append(
                {
                    "agent_name": "multistep_review_agent",
                    "agent_role": "insight review",
                    "logical_input": {
                        "goal": goal,
                        "insight": insight,
                        "seen_insights": memory,
                    },
                    "prompt": prompt,
                    "raw_output": str(raw_output),
                    "parsed_output": review,
                }
            )

        # This is the simple memory.
        # It lets the next insight be compared against previous reviewed insights.
        memory.append(review["insight"])

    return reviews


def run_research_question_agent(
    model: Any,
    goal: str,
    payload: list[dict[str, Any]],
    verbosity_level: int = 2,
    batch_review: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = build_model(model)
    selected_insights: list[str] = []
    prioritized_questions: list[str] = []
    priority_rationale = ""
    research_payload: list[dict[str, Any]] = []

    if batch_review:
        prompt = build_batch_research_question_prompt(
            goal=goal,
            payload=payload,
            selected_insights=selected_insights,
            prioritized_questions=prioritized_questions,
        )
        try:
            raw_output = model.generate(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            parsed_output = parse_json_output(str(raw_output.content))
            research_payload = normalize_insights_input(parsed_output.get("research_payload", []))
            prioritized_questions_payload = parsed_output.get("prioritized_questions_payload", {})
            prioritized_questions_payload = {
                "prioritized_questions": [
                    str(item)
                    for item in prioritized_questions_payload.get("prioritized_questions", [])
                    if str(item).strip()
                ],
                "priority_rationale": str(prioritized_questions_payload.get("priority_rationale", "")),
            }
            return research_payload, prioritized_questions_payload
        except Exception:
            pass

    for review in progress(payload, desc="Generating research questions"):
        research_tool = SubmitResearchQuestionsTool()
        prioritize_tool = PrioritizeResearchQuestionsTool()

        agent = ToolCallingAgent(
            model=model,
            tools=[research_tool, prioritize_tool],
            verbosity_level=verbosity_level,
            name="research_question_agent",
            description=(
                "Generates follow-up research questions for important or "
                "interesting reviewed insights to make them more actionable, "
                "then ranks the running question pool down to the top 3."
            ),
            max_steps=5,
        )

        prompt = build_research_question_prompt(
            goal=goal,
            review=review,
            all_reviews=payload,
            selected_insights=selected_insights,
            prioritized_questions=prioritized_questions,
        )

        agent.run(prompt)

        if research_tool.last_result is None:
            raise ValueError("Research question agent did not call submit_research_questions.")
        if prioritize_tool.last_result is None:
            raise ValueError("Research question agent did not call prioritize_research_questions.")

        result = research_tool.last_result
        research_payload.append(result)
        prioritized_questions = prioritize_tool.last_result["prioritized_questions"]
        priority_rationale = prioritize_tool.last_result["priority_rationale"]

        if result["research_questions"]:
            selected_insights.append(result["insight"])

    prioritized_questions_payload = {
        "prioritized_questions": prioritized_questions,
        "priority_rationale": priority_rationale,
    }

    return research_payload, prioritized_questions_payload


# ---------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------
# if __name__ == "__main__":
#     goal = "Find insights that help reduce customer churn."

#     insights = [
#         {
#             "insight": "Users with repeated failed payments are more likely to churn.",
#             "evidence": "Users with 2 or more failed payments have 30% higher churn.",
#         },
#         {
#             "insight": "Payment failures are associated with higher churn.",
#             "evidence": "Users who experienced payment failures churn more often.",
#         },
#         {
#             "insight": "Users who contact support within their first week retain better.",
#             "evidence": "New users who contacted support had 12% higher 60-day retention.",
#         },
#     ]

#     payload = run_reviewer(
#         goal=goal,
#         insights=insights[:3],
#         model="gemini-flash-lite-latest",
#         verbosity_level=1,
#     )

#     print(json.dumps(payload, indent=2))
