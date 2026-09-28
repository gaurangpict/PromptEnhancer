# PromptEnhancer

LLM answers are long because prompts are vague: the model has to guess the
audience, length, tone and format, so it pads the answer to cover every guess.

PromptEnhancer asks a few **one-tap questions** about what your prompt leaves
open, then (next milestone) turns it into a precise prompt. It never answers the
prompt itself. It runs on free, open-source models via
[NVIDIA NIM](https://build.nvidia.com).

**No hidden assumptions:** every gap in your prompt is either asked about or
shown to you as an assumption you can flip.

## Status

| Part | State |
|---|---|
| Setup, NIM client, `/api/health` | ✅ done |
| Question engine, `/api/questions` | ✅ done |
| Evals for the question engine | ✅ done |
| Prompt enhancer, `/api/enhance` | next |
| Web page | planned |
| Reference-file upload + tool-calling agent | planned |

## How the questions work

One LLM call finds the gaps in the prompt and scores each one for **impact**
(how much the answer would change) and **confidence** (how sure the model is of
the likely answer). Then plain code decides:

```
impact <= 1                        → ignore
impact >= 3 and confidence <= 3    → ask the user (max 5, most valuable first)
everything else                    → assume, and show it so the user can flip it
```

Every option carries its **effect**: the exact instruction it adds to the
final prompt. Full rules and reasoning: [docs/question-design.md](docs/question-design.md).

Example response for `write an email to my boss about leave`:

```json
{
  "task_type": "writing",
  "known": ["format: email", "recipient: the user's boss", "topic: a leave request"],
  "questions": [
    {
      "id": "leave_type",
      "question": "Planned leave (vacation) or sudden (sick or emergency)?",
      "kind": "either",
      "options": [
        {"label": "Planned", "effect": "Write it as a request for planned leave, asking for approval in advance."},
        {"label": "Sudden", "effect": "Write it as a short-notice message about unplanned leave due to illness or an emergency."}
      ],
      "default": 0, "impact": 5, "confidence": 2, "dimension": "purpose"
    }
  ],
  "assumptions": [
    {
      "id": "length",
      "question": "Keep it under 120 words?",
      "kind": "yesno",
      "options": [
        {"label": "Yes", "effect": "Keep the email under 120 words."},
        {"label": "No", "effect": "Allow up to 250 words."}
      ],
      "default": 0, "impact": 3, "confidence": 5, "dimension": "length"
    }
  ]
}
```

## Setup

Needs Python 3.11+.

1. Get a free NVIDIA API key (no credit card): sign in at
   [build.nvidia.com](https://build.nvidia.com) and click **Get API Key**.
2. Install and configure:

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   cp .env.example .env               # then put your key in .env
   ```

3. Run the server:

   ```bash
   uvicorn app.main:app --reload
   ```

4. Try it at <http://localhost:8000/docs> (interactive API page), or:

   ```bash
   curl -X POST localhost:8000/api/questions \
     -H "content-type: application/json" \
     -d '{"prompt": "write an email to my boss about leave"}'
   ```

## Configuration

All settings can be set in `.env` (see `.env.example`).

| Variable | Default | Meaning |
|---|---|---|
| `NVIDIA_API_KEY` | — | Your NIM key (required) |
| `QUESTION_MODEL` | `meta/llama-3.3-70b-instruct` | Model that finds and scores gaps |
| `JUDGE_MODEL` | `meta/llama-3.1-405b-instruct` | Evals only: grades the output |
| `MAX_QUESTIONS` | `5` | Most questions asked at once |
| `ASK_MIN_IMPACT` / `ASK_MAX_CONFIDENCE` | `3` / `3` | When a gap is asked instead of assumed |
| `IGNORE_MAX_IMPACT` | `1` | Gaps at or below this impact are ignored |

NVIDIA retires model names now and then. If you get "model not found", pick a
current one on build.nvidia.com and set it in `.env`.

## Tests and evals

**Tests** use a fake LLM: offline, free, and the same result every time.

```bash
pytest
ruff check . && ruff format --check .
```

**Evals** use the real models and your free quota (~30 calls per full run).
Run them whenever you change the instructions in `app/prompts.py`, a
threshold, or a model, and compare the summaries:

```bash
python -m evals.run_questions --tag baseline
# change something...
python -m evals.run_questions --tag my-change
```

They measure question-count accuracy, words per question, coverage of the
expected gaps, questions about things the prompt already says, and the phrasing
rules (one idea, neutral, specific, same language), graded by a separate judge
model. Test cases live in `evals/questions.jsonl`; add your own.

## Project layout

```
app/
  config.py     settings from .env
  llm.py        NIM client, error mapping, JSON parsing with one retry
  prompts.py    the instructions sent to the model (the core of the product)
  analyzer.py   question engine: parse → drop useless → ask / assume / ignore
  schemas.py    data shapes for model output and the API
  main.py       FastAPI app
docs/
  question-design.md   the rules for questions
evals/
  questions.jsonl      test prompts with expected gaps
  run_questions.py     eval runner + LLM judge
tests/                 unit tests (fake LLM)
```
