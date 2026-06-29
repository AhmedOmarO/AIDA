from pathlib import Path
from typing import Any

from .smolagents_context import context_to_dict, load_context
from .smolagents_models import build_model
from .multistep_insight_agent import run_multistep_insight_agent
from .multistep_agent import run_reviewer, run_research_question_agent
from .progress import progress
from .smolagents_tools import build_tools
from .smolagents_trace import append_trace_call, init_trace, write_trace_logs

PACKAGE_ROOT = Path(__file__).resolve().parent
PUBLIC_ROOT = PACKAGE_ROOT.parents[2]
DEFAULT_PROMPTS_PATH = PACKAGE_ROOT / "agent_prompts.json"


OUTPUTS_PATH = PUBLIC_ROOT / "smolagents_ssot_loop_outputs.json"


def build_progress_label(context: Any) -> str:
    dataset_name = Path(str(context.dataset_path)).stem.replace("_", " ").replace("-", " ").strip()
    if dataset_name:
        return f"Analyzing {dataset_name}"
    return "Analyzing CSV"


def run_loop(
    context: Any,
    model: Any,
    rounds: int,
    questions_per_round: int,
    model_name: str | None = None,
    log_dir: Path | str | None = None,
    verbosity_level: int = 1,
    review_agent: bool = False,
    batch_review: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tools = build_tools(context)
    suggested_questions: list[str] = []
    state = {
        "flag_id": context.flag_id,
        "goal": context.goal,
        "difficulty": context.difficulty,
        "dataset_path": str(context.dataset_path),
        "rounds": [],
        "verbosity_level": verbosity_level,
    }
    trace = init_trace(context, model_name)
    call_index = 0
    prior_round_context: list[dict[str, Any]] = []
    last_review_payload: list[dict[str, Any]] = []
    # good_insight_example = random.choice(context.expected_insights) if context.expected_insights else ""

    round_iterator = progress(
        range(1, rounds + 1),
        desc=build_progress_label(context),
        leave=False,
    )
    for round_index in round_iterator:
        multi_step_input = {
            "goal": context.goal,
            "prior_round_context": prior_round_context,
            "questions_per_round": questions_per_round,
        }
        prior_context_label = "review payloads" if review_agent else "generated insights"
        ## Run multistep insight agent
        multi_step_raw, multi_step_payload, multi_step_prompt = run_multistep_insight_agent(
            model=model,
            tools=tools,
            goal=context.goal,
            prior_context=prior_round_context,
            questions_per_round=questions_per_round,
            verbosity_level=verbosity_level,
            dataframe=context.dataframe,
            suggested_questions=suggested_questions,
            prior_context_label=prior_context_label,
        )
        questions = multi_step_payload.get("questions", [])
        generated_insights = multi_step_payload.get("insights", [])
        call_index += 1
        append_trace_call(
            trace,
            call_index=call_index,
            round=round_index,
            agent_name="multistep_insight_agent",
            agent_role="tool-based insight generation",
            logical_input=multi_step_input,
            prompt=multi_step_prompt,
            raw_output=str(multi_step_raw),
            parsed_output=multi_step_payload,
        )
        if review_agent:
            review_trace_calls: list[dict[str, Any]] = []
            review_payload = run_reviewer(
                    goal=context.goal,
                    insights=generated_insights,
                    model=model_name,
                    verbosity_level=verbosity_level,
                    seen_insights=prior_round_context,
                    trace_collector=review_trace_calls,
                    batch_review=batch_review,
            )
            for review_trace_call in review_trace_calls:
                call_index += 1
                append_trace_call(
                    trace,
                    call_index=call_index,
                    round=round_index,
                    **review_trace_call,
                )
            last_review_payload.extend(review_payload)
            if round_index < rounds:
                research_payload, prioritized_questions_payload = run_research_question_agent(
                        goal=context.goal,
                        payload=review_payload,
                        model=model_name,
                        verbosity_level=verbosity_level,
                        batch_review=batch_review,
                )
                suggested_questions = prioritized_questions_payload["prioritized_questions"]
        else:
            pass
        if review_agent:
            prior_round_context.extend(review_payload)
        else:
            prior_round_context.extend(generated_insights)
        state["rounds"].append(
            {
                "round": round_index,
                "questions": questions,
                "generated_insights": generated_insights,
                "review": '',
            }
        )

    state["final_relevant_insights"] = (
    [insight for round_data in state["rounds"] for insight in round_data["generated_insights"]]
    )
    state["expected_ssot_insights"] = context.expected_insights
    if log_dir is not None:
        state["llm_trace_files"] = write_trace_logs(trace, Path(log_dir), context.flag_id)
    state["llm_trace"] = trace
    if review_agent:
        assert len(state["final_relevant_insights"]) == len(last_review_payload)
        print(f"Total insights generated: {len(state['final_relevant_insights'])} Total reviews generated: {len(last_review_payload)}")
    return state, last_review_payload


def run_flag(
    flag_id: str = "flag-1",
    ssot_path: Path | str | None = None,
    model_name: str = "gemini/gemini-flash-lite-latest",
    rounds: int = 2,
    questions_per_round: int = 4,
    log_dir: Path | str | None = "logs",
    prompts_path: Path | str = DEFAULT_PROMPTS_PATH,
    outputs_path: Path | str = OUTPUTS_PATH,
    verbosity_level: int = 1,
    review_agent: bool = False,
    batch_review: bool = True,
 ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if ssot_path is None:
        raise ValueError("ssot_path is required for SSOT-based runs.")
    return run_loop(
        load_context(PUBLIC_ROOT, Path(ssot_path), flag_id),
        build_model(model_name),
        rounds,
        questions_per_round,
        model_name=model_name,
        log_dir=log_dir,
        verbosity_level=verbosity_level,
        review_agent=review_agent,
        batch_review=batch_review,
    )


def validate_flag(flag_id: str = "flag-1", ssot_path: Path | str | None = None) -> dict[str, Any]:
    if ssot_path is None:
        raise ValueError("ssot_path is required for SSOT validation.")
    return context_to_dict(load_context(PUBLIC_ROOT, Path(ssot_path), flag_id))
