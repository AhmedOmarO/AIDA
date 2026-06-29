from __future__ import annotations

import io
import json
import sqlite3
import traceback
from contextlib import redirect_stdout
from typing import Any

from .smolagents_context import DatasetContext
from .smolagents_utils import require_module


def build_tools(context: DatasetContext) -> list[Any]:
    pd = require_module("pandas")
    tool = require_module("smolagents").tool
    dataframe = context.dataframe.copy()
    sqlite_conn = sqlite3.connect(":memory:")
    dataframe.to_sql("dataset", sqlite_conn, index=False, if_exists="replace")

    @tool
    def get_dataset_schema() -> str:
        """Return the dataset schema, dtypes, row count, and a small CSV sample."""
        return json.dumps(
            {
                "flag_id": context.flag_id,
                "goal": context.goal,
                "dataset_path": context.dataset_path_value,
                "row_count": context.row_count,
                "columns": context.schema,
                "sample_rows_csv": context.sample_rows_csv,
            },
            indent=2,
        )

    @tool
    def run_sql(query: str) -> str:
        """
        Run a read-only SQL query against a SQLite table named `dataset`.

        Args:
            query: A single SELECT statement to execute against the dataset table.
        """
        if not query.strip().lower().startswith("select"):
            return "Only SELECT queries are allowed."
        try:
            result = pd.read_sql_query(query, sqlite_conn)
        except Exception as exc:
            return f"SQL error: {exc}"
        return "Query returned 0 rows." if result.empty else result.head(50).to_csv(index=False)

    @tool
    def run_python(code: str) -> str:
        """
        Execute read-only Python analysis against a pandas DataFrame named df.

        Args:
            code: Python code that analyzes the DataFrame `df`. Set a variable named `result` or print output.
        """
        local_vars: dict[str, Any] = {"pd": pd, "df": dataframe.copy()}
        stdout = io.StringIO()
        try:
            with redirect_stdout(stdout):
                exec(code, {"__builtins__": __builtins__}, local_vars)
        except Exception:
            return "Python error:\n" + traceback.format_exc(limit=2)
        result, printed = local_vars.get("result"), stdout.getvalue().strip()
        if isinstance(result, pd.DataFrame):
            return result.head(50).to_csv(index=False)
        if isinstance(result, pd.Series):
            return result.head(50).to_string()
        return str(result) if result is not None else (printed or "Python executed with no output.")

    return [get_dataset_schema, run_sql, run_python]
