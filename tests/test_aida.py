from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

from aida import AIDA


class AIDATestCase(unittest.TestCase):
    def _write_csv(self) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "sample.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["category", "value"])
            writer.writeheader()
            writer.writerow({"category": "A", "value": 10})
            writer.writerow({"category": "B", "value": 20})
        return path

    def test_build_context_loads_csv(self) -> None:
        csv_path = self._write_csv()
        context = AIDA.build_context(
            csv_path=csv_path,
            goal="Find useful patterns.",
            flag_id="sample",
            difficulty="easy",
        )

        self.assertEqual(context.flag_id, "sample")
        self.assertEqual(context.goal, "Find useful patterns.")
        self.assertEqual(context.row_count, 2)
        self.assertEqual(context.schema[0]["column"], "category")

    def test_review_delegates_to_run_reviewer(self) -> None:
        captured: dict[str, object] = {}

        def fake_run_reviewer(**kwargs):
            captured.update(kwargs)
            return [{"relevance_flag": "relevant"}]

        fake_module = ModuleType("aida.multistep_agent")
        fake_module.run_reviewer = fake_run_reviewer
        original = sys.modules.get("aida.multistep_agent")
        sys.modules["aida.multistep_agent"] = fake_module

        def restore_module() -> None:
            if original is None:
                sys.modules.pop("aida.multistep_agent", None)
            else:
                sys.modules["aida.multistep_agent"] = original

        self.addCleanup(restore_module)

        result = AIDA.review(
            goal="Find useful patterns.",
            insights=[{"insight": "A", "evidence": "B"}],
            model_name="openai/gpt-4.1-mini",
        )

        self.assertEqual(result, [{"relevance_flag": "relevant"}])
        self.assertEqual(captured["goal"], "Find useful patterns.")
        self.assertEqual(captured["model"], "openai/gpt-4.1-mini")


if __name__ == "__main__":
    unittest.main()
