from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .smolagents_context import DatasetContext, read_csv_rows_with_fallback


class AIDA:
    @staticmethod
    def build_context(
        csv_path: str | Path,
        goal: str,
        flag_id: str | None = None,
        difficulty: str = "custom",
    ) -> DatasetContext:
        path = Path(csv_path).expanduser().resolve()
        resolved_flag_id = flag_id or path.stem
        try:
            pd = importlib.import_module("pandas")
        except ImportError:
            columns, rows = read_csv_rows_with_fallback(path)
            dataframe: Any = {
                "columns": columns,
                "sample_rows": [[row.get(column, "") for column in columns] for row in rows[:8]],
                "row_count": len(rows),
            }
        else:
            dataframe = pd.read_csv(path)
        return DatasetContext(
            flag_id=resolved_flag_id,
            goal=goal,
            expected_insights=[],
            difficulty=difficulty,
            dataset_path=path,
            dataframe=dataframe,
        )

    @staticmethod
    def run(
        csv_path: str | Path,
        goal: str,
        model_name: str,
        rounds: int,
        questions_per_round: int,
        with_review: bool,
        verbosity_level: int = 0,
        flag_id: str | None = None,
        difficulty: str = "custom",
    ) -> Any:
        from .agent_loop import run_loop
        from .smolagents_models import build_model

        context = AIDA.build_context(
            csv_path=csv_path,
            goal=goal,
            flag_id=flag_id,
            difficulty=difficulty,
        )
        model = build_model(model_name)
        return run_loop(
            context=context,
            model=model,
            rounds=rounds,
            questions_per_round=questions_per_round,
            model_name=model_name,
            verbosity_level=verbosity_level,
            review_agent=with_review,
        )

    @staticmethod
    def review(
        goal: str,
        insights: dict[str, Any] | list[dict[str, Any]],
        model_name: str,
        seen_insights: list[str] | None = None,
        verbosity_level: int = 0,
    ) -> list[dict[str, Any]]:
        from .multistep_agent import run_reviewer

        return run_reviewer(
            model=model_name,
            goal=goal,
            insights=insights,
            seen_insights=seen_insights,
            verbosity_level=verbosity_level,
        )
