from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .smolagents_utils import require_module, resolve_dataset_paths


CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


@dataclass
class DatasetContext:
    flag_id: str
    goal: str
    expected_insights: list[str]
    difficulty: str
    dataset_path: Path | list[Path]
    dataframe: Any

    @property
    def dataset_path_value(self) -> str | list[str]:
        return str(self.dataset_path) if isinstance(self.dataset_path, Path) else [str(path) for path in self.dataset_path]

    @property
    def schema(self) -> list[dict[str, str]]:
        if hasattr(self.dataframe, "dtypes"):
            return [{"column": name, "dtype": str(dtype)} for name, dtype in self.dataframe.dtypes.items()]
        return [{"column": name, "dtype": "unknown"} for name in self.dataframe["columns"]]

    @property
    def sample_rows_csv(self) -> str:
        if hasattr(self.dataframe, "head"):
            return self.dataframe.head(8).to_csv(index=False)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(self.dataframe["columns"])
        writer.writerows(self.dataframe["sample_rows"])
        return output.getvalue()

    @property
    def row_count(self) -> int:
        return int(len(self.dataframe)) if hasattr(self.dataframe, "index") else int(self.dataframe["row_count"])


def read_csv_with_fallback(pd: Any, path: Path) -> Any:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path)


def read_csv_rows_with_fallback(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            with path.open(newline="", encoding=encoding) as handle:
                reader = csv.DictReader(handle)
                return list(reader.fieldnames or []), list(reader)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def load_context(root: Path, ssot_path: Path, flag_id: str) -> DatasetContext:
    ssot = json.loads(ssot_path.read_text())
    if flag_id not in ssot:
        available = ", ".join(sorted(ssot.keys())[:10])
        raise KeyError(f"Unknown flag '{flag_id}'. Sample available flags: {available}")
    entry = ssot[flag_id]
    dataset_paths = resolve_dataset_paths(root, entry["data"])
    dataset_path: Path | list[Path] = dataset_paths[0] if len(dataset_paths) == 1 else dataset_paths
    try:
        pd = require_module("pandas")
        if len(dataset_paths) == 1:
            dataframe: Any = read_csv_with_fallback(pd, dataset_paths[0])
        else:
            frames = []
            for path in dataset_paths:
                frame = read_csv_with_fallback(pd, path)
                frame["source_dataset"] = path.stem
                frames.append(frame)
            dataframe = pd.concat(frames, ignore_index=True, sort=False)
    except SystemExit:
        rows: list[dict[str, str]] = []
        column_order: list[str] = []
        for path in dataset_paths:
            file_columns, file_rows = read_csv_rows_with_fallback(path)
            combined_columns = file_columns + (["source_dataset"] if len(dataset_paths) > 1 else [])
            for column in combined_columns:
                if column not in column_order:
                    column_order.append(column)
            for row in file_rows:
                normalized_row = {column: row.get(column, "") for column in file_columns}
                if len(dataset_paths) > 1:
                    normalized_row["source_dataset"] = path.stem
                rows.append(normalized_row)
        dataframe = {
            "columns": column_order,
            "sample_rows": [[row.get(column, "") for column in column_order] for row in rows[:8]],
            "row_count": len(rows),
        }
    return DatasetContext(flag_id, entry["goal"], entry.get("insights", []), entry.get("difficulty", "unknown"), dataset_path, dataframe)


def context_to_dict(context: DatasetContext) -> dict[str, Any]:
    return {
        "flag_id": context.flag_id,
        "goal": context.goal,
        "difficulty": context.difficulty,
        "dataset_path": context.dataset_path_value,
        "row_count": context.row_count,
        "schema": context.schema,
        "sample_rows_csv": context.sample_rows_csv,
        "expected_insights": context.expected_insights,
    }
