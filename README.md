# AIDA — AI-Assisted Insight Discovery for Tabular Data

AIDA is a Python package that turns a CSV and an analytical goal into a ranked set of decision-useful insights. It uses an iterative LLM-driven analyst loop to explore the data, generate evidence-backed findings, critique coverage, and refine the next round of questions.

The project explores a practical question: **how can an AI analyst surface the most useful findings before a human has to inspect every possible insight?**

## Highlights

- Goal-aware analysis of arbitrary CSV files
- Iterative question generation and data exploration
- Optional reviewer agent that critiques findings and identifies missing areas
- Structured outputs for downstream ranking, reporting, or evaluation
- Provider-flexible model configuration
- Installable Python package with tests and a minimal notebook

## How it works

1. A user supplies a CSV file and an analytical goal.
2. The analyst agent inspects the data and asks targeted questions.
3. Candidate insights are generated with supporting evidence.
4. When review is enabled, a second agent critiques coverage and suggests follow-up questions.
5. Feedback is carried into the next round, improving relevance and breadth.

## Quick start

### Install

```bash
git clone https://github.com/AhmedOmarO/AIDA.git
cd AIDA
pip install -e .
```

AIDA requires Python 3.10 or newer.

### Configure a model

Set the API key required by your chosen model provider. For Gemini:

```bash
export GEMINI_API_KEY=your_key_here
```

### Run an analysis

```python
from aida import AIDA

result = AIDA.run(
    csv_path="airline-safety.csv",
    goal="Find the most decision-useful patterns in airline safety incidents.",
    model_name="gemini/gemini-flash-lite-latest",
    rounds=2,
    questions_per_round=3,
    with_review=True,
)

for insight in result["final_relevant_insights"][:3]:
    print(insight)
```

Download the example dataset:

```bash
curl -L https://raw.githubusercontent.com/fivethirtyeight/data/master/airline-safety/airline-safety.csv \
  -o airline-safety.csv
```

## Reviewer agent

With `with_review=True`, AIDA runs a reviewer after every generation round. The reviewer:

- critiques the current candidate insights;
- identifies weak or underexplored areas;
- proposes follow-up questions; and
- guides the analyst's next iteration.

Use `with_review=False` for the raw generation loop, or call `AIDA.review(...)` to review an existing set of candidate insights.

## Input and output

AIDA accepts any local CSV readable by `pandas.read_csv`. Mixed numeric and categorical columns are supported. The result is a structured dictionary containing generated findings, review feedback, and the final relevant insights.

## Repository structure

```text
src/aida/                         Python package
tests/                            Lightweight tests
notebooks/minimal_aida_usage.ipynb  Runnable walkthrough
pyproject.toml                    Package metadata and dependencies
```

## Tests

```bash
python -m unittest discover -s tests
```

## Project status

AIDA is an experimental research prototype for proactive, context-aware data analysis. Current work focuses on insight quality, ranking, novelty, and evaluation against reference findings.

## License

MIT — see [LICENSE](LICENSE).
