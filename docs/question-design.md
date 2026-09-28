# Question design

This is the spec for how PromptEnhancer turns a rough prompt into clarifying
questions. The model instructions (`app/prompts.py`) and the code checks
(`app/analyzer.py`) both follow it. When you change a rule here, change it in
both places and re-run the evals (`evals/`).

## Goal

The final prompt should contain **no hidden assumptions**. Every gap in the
user's prompt ends up in exactly one of three places:

| Gap | Where it goes |
|---|---|
| Changes the answer a lot, and we can't guess it well | **Asked** as a question |
| Changes the answer a lot, but the likely value is obvious | **Assumed**, and shown to the user so they can flip it |
| Barely changes the answer | **Ignored** |

We never guess silently.

## Pipeline

```
user prompt
   │
   ▼
1. UNDERSTAND   task type + facts the prompt already states ("known")
2. FIND GAPS    per-task checklist + prompt-specific gaps,
                each scored: impact (1-5), confidence (1-5)
3. DECISIONS    each gap becomes a question with options,
                each option carries the exact effect on the final prompt
   ── steps 1-3 are one LLM call ──
4. CHECK        code rejects badly-formed decisions
5. SELECT       code decides ask / assume / ignore, and ranks the questions
```

The **LLM gives the scores; code makes the decisions.** That keeps behaviour
predictable: if users see too many questions, we change a threshold, not a
prompt.

## Scores

- **impact** (1-5): how different the final answer would be depending on the
  option chosen. 5 = a completely different answer; 1 = cosmetic.
- **confidence** (1-5): how sure the model is that its default option is what
  the user wants. 5 = almost certain; 1 = a pure guess.

## Selection rule (code, `app/analyzer.py`)

```
impact <= 1                         → ignore
impact >= 3 and confidence <= 3     → ask     (ranked by impact × (6 − confidence))
everything else                     → assume  (default option, shown to user)
```

At most `MAX_QUESTIONS` (default 5) are asked. Extra "ask" gaps become
assumptions. All numbers live in `app/config.py` and can be set from `.env`.

## Checklists per task type

These make coverage consistent. The model also adds gaps specific to the
prompt.

- **code**: language/version, runtime or framework, input/output shape, error
  handling, tests, explanation vs. code only
- **writing**: audience, purpose, tone, length, structure, call to action
- **analysis**: data/source available, depth, form of conclusion, audience
- **learning**: current level, format (steps, analogy, example), depth
- **all tasks**: output format, length, scope (what to include or leave out)

## Question kinds

| Kind | Use for | Options |
|---|---|---|
| `yesno` | include/skip decisions ("Include code comments?") | exactly `Yes`, `No` |
| `either` | two clear alternatives ("Beginners or experienced developers?") | exactly 2 |
| `choice` | several distinct values (length, format) | 3 or 4 |

The UI always adds a **"Doesn't matter"** answer. It means "leave this open":
the enhancer adds no instruction for it, instead of inventing one.

Why `either` exists: a yes/no question built around one guess ("Is it for
beginners?") leads the user towards that guess, and a "No" tells us nothing
about what they do want. When there are two clear sides, naming both costs the
user the same one tap.

## Phrasing rules

| Rule | Bad | Good |
|---|---|---|
| Use the user's own topic and words | "What is the purpose?" | "Is the leave email for a sick day?" |
| One idea per question | "Should it be formal and short?" | Two decisions |
| Concrete, no jargon | "What verbosity level?" | "Under 150 words?" |
| Neutral, not leading | "Formal, as usual for a boss?" | "Is your relationship with your boss casual?" |
| Name both sides when there are two | "Is it for beginners?" | "Beginners or experienced developers?" |
| Answerable in under 3 seconds | "What are your success criteria?" | "Should it end with a clear next step?" |
| Same language as the user's prompt | | |
| Never ask what the prompt already says | "Who is the email for?" (prompt says "my boss") | |

## Effects

Every option carries an **effect**: the exact instruction that goes into the
final prompt when that option is chosen, written as an imperative sentence.

```json
{
  "question": "Mention who covers your work while you're away?",
  "kind": "yesno",
  "options": [
    {"label": "Yes", "effect": "Add one sentence naming the colleague who covers urgent work."},
    {"label": "No",  "effect": "Do not mention work coverage."}
  ],
  "default": 0
}
```

Effects make the pipeline checkable:

1. If two options have the same effect, the question doesn't matter → dropped.
2. The enhancer combines precise instructions instead of interpreting "Yes".
3. Flipping an assumption just swaps one effect for another; no new LLM call
   to re-ask.
4. Evals can check that each chosen effect really shows up in the final prompt.

## Code checks (`app/analyzer.py`)

A decision is **dropped** when:
- its options all have the same effect (the answer changes nothing)
- it repeats another decision (same id or same question text)

A decision that would be **asked** is **demoted to an assumption** when:
- the question is longer than `MAX_QUESTION_WORDS` (default 15) words, or
  `MAX_QUESTION_CHARS` characters (for languages without spaces)
- it is a `yesno` question containing "or" (it is really an `either`)

Some rules can't be checked reliably by code. "One idea per question" is the
main one: a regex can't tell "Include pros and cons?" (one idea) from "Formal
and short?" (two ideas). Those rules are judged in the evals by an LLM judge.

## Worked example

Prompt: *"write an email to my boss about leave"*

| Gap | Default | Impact | Confidence | Result |
|---|---|---|---|---|
| Type of leave | planned | 5 | 2 | **Ask**: "Planned leave (vacation) or sudden (sick/emergency)?" |
| Dates | placeholders | 4 | 2 | **Ask**: "Leave [date] blanks for you to fill in?" |
| Tone | formal | 4 | 3 | **Ask**: "Is your relationship with your boss casual?" |
| Handover | mention it | 3 | 2 | **Ask**: "Mention who covers your work?" |
| Length | under 120 words | 3 | 5 | **Assume** |
| Language | English | 5 | 5 | **Assume** |
| Font | — | 1 | — | **Ignore** |
