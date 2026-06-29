# AIDA

AIDA is a Python package for finding decision-useful insights in a CSV with an LLM-driven analyst loop.

## What You Need To Do

### 1. Install AIDA

If you are in this monorepo:

```bash
pip install -e ./public
```

If you are in the standalone `AIDA` repo:

```bash
pip install -e .
```

This installs the required runtime dependencies:

- `litellm`
- `pandas`
- `smolagents`
- `tqdm`

### 2. Set Your Gemini API Key

```bash
export GEMINI_API_KEY=your_key_here
```

The simplest default model for this package is:

```text
gemini/gemini-flash-lite-latest
```

### 3. Run AIDA On Your CSV

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

print(result["final_relevant_insights"][:2])
```

Example CSV:

```bash
curl -L https://raw.githubusercontent.com/fivethirtyeight/data/master/airline-safety/airline-safety.csv -o airline-safety.csv
```

## What The Review Agent Does

When `with_review=True`, AIDA runs a second agent after each generation round.

The review agent does not create the main insights itself. It critiques the current candidate insights, identifies weak spots, lists missing areas to cover, and suggests follow-up questions for the next round. That feedback is then carried into the next iteration so the main analyst agent can improve coverage and relevance.

Use `with_review=False` if you want the raw generation loop only.

If you already have candidate insights and only want the review step, use:

```python
AIDA.review(...)
```

## Input Expectations

- Pass any local CSV path to `csv_path`.
- The file should be readable by `pandas.read_csv`.
- Mixed numeric and text columns work well.

## Repo Layout

- `src/aida/`: package source
- `tests/`: lightweight package tests
- `notebooks/minimal_aida_usage.ipynb`: minimal walkthrough

## Notebook

Open [minimal_aida_usage.ipynb](notebooks/minimal_aida_usage.ipynb) for the smallest runnable example.

## Tests

From the monorepo root:

```bash
PYTHONPATH=public/src python -m unittest discover -s public/tests
```

From the standalone repo root:

```bash
python -m unittest discover -s tests
```

## License

MIT. See `LICENSE`.
