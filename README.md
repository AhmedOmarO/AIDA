# Public AIDA

Public AIDA is the cleaned user-facing package in this monorepo. It exposes the `AIDA` interface for running insight generation on a CSV with an LLM-backed analyst loop.

## Layout

- `src/aida/`: package code
- `notebooks/minimal_aida_usage.ipynb`: minimal usage walkthrough

## Install

From the repo root:

```bash
pip install -e ./public
```

If you are running the notebook before installing the package, the notebook adds `public/src` to `sys.path` so `from aida import AIDA` still works.

## Environment variables

Set the API key for the provider you want to use:

```bash
export OPENAI_API_KEY=your_key
```

or

```bash
export GEMINI_API_KEY=your_key
```

or

```bash
export DEEPSEEK_API_KEY=your_key
```

## Minimal example

```python
from aida import AIDA

result = AIDA.run(
    csv_path="airline-safety.csv",
    goal="Find the most decision-useful patterns in airline safety incidents.",
    model_name="openai/gpt-4.1-mini",
    rounds=2,
    questions_per_round=3,
    with_review=False,
)

print(result["final_relevant_insights"][:2])
```

Example dataset download:

```bash
curl -L https://raw.githubusercontent.com/fivethirtyeight/data/master/airline-safety/airline-safety.csv -o airline-safety.csv
```

## Review mode

`with_review=False` returns the generated insights directly from the multi-round loop.

`with_review=True` runs the reviewer agent each round and uses those reviews as the carry-forward context between rounds.

If you already have candidate insights and only want relevance scoring, use `AIDA.review(...)`.

## User CSVs

Pass any local CSV path to `csv_path`. The file should be readable by `pandas.read_csv`. Mixed text and numeric columns work best because the agent can inspect both structure and free-text fields.

For FiveThirtyEight examples, download the CSV you want directly from the FiveThirtyEight GitHub repository rather than using vendored data from this repo.

## Notebook

Open [minimal_aida_usage.ipynb](/Users/ahmed/Aida/public/notebooks/minimal_aida_usage.ipynb) for the smallest runnable example in this repo.

## Tests

Run the lightweight package tests from the repo root:

```bash
PYTHONPATH=public/src python -m unittest discover -s public/tests
```
