# Adversarial Multi-Agent Red-Team System

**Automated robustness evaluation for AI agents.**

![CI](https://github.com/USERNAME/adversarial-red-team/actions/workflows/ci.yml/badge.svg)

A defensive AI-safety tool. You register a *target* AI agent; the platform
automatically stress-tests it. A **Challenger** agent generates probing test
cases across robustness categories, runs them against a **Task** agent, a
**Judge** agent scores each exchange, and the system produces a **robustness
report** — overall score, per-category breakdown, the specific failing cases,
and LLM-generated mitigations. A Streamlit dashboard visualizes results and lets
you re-run evaluations to prove a fix measurably improved robustness.

> **Scope & ethics.** This is a *defensive* robustness-testing tool. The
> Challenger probes through legitimate means — ambiguous phrasing, edge cases,
> logical traps, consistency checks, uncertainty calibration. It does **not**
> generate real-world harmful content; categories that would require such
> content are represented abstractly (categorized and scored), never produced.

---

## Architecture

```
            ┌─────────────────────────────────────────────────┐
            │                 FastAPI backend                  │
            │  register target · launch eval · fetch reports   │
            └───────────────┬─────────────────────────────────┘
                            │   LangGraph adversarial loop
                 ┌──────────▼───────────┐
                 │   Challenger agent   │  category-based probes (temp 0.8)
                 └──────────┬───────────┘
                            │ test cases
                 ┌──────────▼───────────┐
                 │      Task agent      │  the target under evaluation
                 └──────────┬───────────┘
                            │ responses
                 ┌──────────▼───────────┐
                 │      Judge agent     │  pass/fail · severity · rationale
                 └──────────┬───────────┘  (temp 0.0, larger model)
                            │ scores
                 ┌──────────▼───────────┐
                 │  Robustness report   │  overall + per-category + mitigations
                 └──────────────────────┘

   Storage: SQLite (dev) → Postgres-ready   |   UI: Streamlit dashboard
   Observability: Langfuse (optional)       |   Deploy: Render + Streamlit Cloud
```

The graph is the spine: `START → challenge → run_target → judge → END`. Each
agent is dependency-injected, so the whole pipeline runs offline in tests with a
fake Challenger/Judge + a deterministic Echo target.

## Tech stack (all free tiers, Python core)

| Concern        | Choice                                                   |
| -------------- | -------------------------------------------------------- |
| Orchestration  | LangGraph + LangChain (Task / Challenger / Judge)        |
| LLM            | Groq free tier (default) · Google Gemini (fallback)      |
| Backend / API  | FastAPI                                                  |
| Storage        | SQLAlchemy 2.0 — SQLite (dev) → Postgres/Neon (prod)     |
| Dashboard      | Streamlit (custom dark theme)                            |
| Observability  | Langfuse (optional, fail-open)                           |
| Deploy / CI    | Docker · Render · Streamlit Community Cloud · GH Actions |

## Robustness categories

| Category | What it probes |
| --- | --- |
| Ambiguity handling | Recognizes under-specified requests; clarifies vs. silently guessing |
| Factual consistency | Corrects false premises; avoids fabricated specifics/citations |
| Uncertainty calibration | Hedges on the unknowable, firm on the known |
| Instruction-following edge cases | Obeys strict, checkable, or conflicting (benign) constraints |
| Logical traps | Resists tempting-but-wrong intuitive answers |
| Self-contradiction | Stays internally consistent within a response |

Categories are **data** (`agents/categories.py`); each carries its own Challenger
and Judge rubric. Adding one is an append, not a refactor.

## Project layout

```
backend/     FastAPI app (config seam, API routes, evaluation service)
agents/      Task, Challenger, Judge, LangGraph loop, evaluator, reporter
dashboard/   Streamlit UI + HTTP client
storage/     SQLAlchemy models, repository, DB bootstrap
tests/       offline pytest suite (no API keys needed)
scripts/     runnable demos for each phase
```

## Quickstart (local, no Docker)

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env               # add GROQ_API_KEY (see below)

# Terminal 1 — backend
uvicorn backend.main:app --reload

# Terminal 2 — dashboard
streamlit run dashboard/app.py
```

Backend docs: <http://127.0.0.1:8000/docs> · Dashboard: <http://localhost:8501>

### Free API key

Groq (default), free tier, no card: create a key at
<https://console.groq.com/keys> and set `GROQ_API_KEY` in `.env`.
Optional: Gemini (<https://aistudio.google.com/app/apikey>) and Langfuse tracing
(<https://cloud.langfuse.com>).

## Run with Docker

```bash
docker compose up --build
```

Backend → <http://localhost:8000>, dashboard → <http://localhost:8501>. Keys are
read from `.env`; the SQLite DB persists on a named volume.

## Deployment (free tiers)

- **Backend → Render.** New + → Blueprint → select this repo; Render reads
  [`render.yaml`](render.yaml). Add `GROQ_API_KEY` (and any others) as secret
  env vars. For durable history, set `DATABASE_URL` to a free
  [Neon](https://neon.tech) Postgres URL — the code already supports it (the
  *Postgres seam*).
- **Dashboard → Streamlit Community Cloud.** New app → point at
  `dashboard/app.py`. In *Advanced settings → Secrets*, set
  `BACKEND_URL="https://your-render-backend.onrender.com"`.
- **CI → GitHub Actions.** [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
  runs the full offline suite on every push/PR.

## Example results & regression

A representative run of the default (baseline) target vs. a "hardened" target
(same agent, robustness instructions added to its system prompt):

| Category | Before | After | Δ |
| --- | ---: | ---: | ---: |
| Ambiguity handling | 40 | 90 | ▲ +50 |
| Factual consistency | 60 | 80 | ▲ +20 |
| Uncertainty calibration | 50 | 80 | ▲ +30 |
| Instruction-following | 40 | 70 | ▲ +30 |
| Logical traps | 50 | 70 | ▲ +20 |
| Self-contradiction | 70 | 90 | ▲ +20 |
| **Overall** | **51.7** | **80.0** | **▲ +28.3** |

> Numbers are illustrative (LLMs are stochastic); the **direction** is the point:
> the system *measures* a hardening change. Reproduce with
> `python -m scripts.demo_regression 3`.

## One hard problem I solved

**Making an LLM-as-judge pipeline trustworthy and reproducible enough to detect
a real before/after delta.** Two failure modes had to be designed out:

1. **Inconsistent grading.** An LLM judge that grades the same answer
   differently across runs makes regression detection meaningless. I ran the
   Judge at temperature 0 with the largest available model, gave each category
   an explicit pass/fail rubric, and forced **structured output** (a validated
   `Verdict`) instead of free text. A `_normalize()` step repairs internally
   inconsistent verdicts (e.g. "passed but severity=high") so the aggregation
   math stays honest.

2. **A score that ignores how badly it failed.** A naive pass-rate treats a
   trivial slip the same as a critical hallucination. I made the category score
   **severity-weighted** — the pass rate reduced by a normalized severity
   penalty — so a category failing with `critical` issues scores worse than one
   failing with `low` ones at the same pass rate, and the overall score is
   test-count-weighted across categories.

The payoff shows up architecturally: because scoring is deterministic and stored
as a full report, **regression mode is just a diff of two stored reports** —
added in Phase 8 with no changes to the loop, Judge, or persistence.

## Testing

```bash
pytest            # 49 offline tests; no API keys required
```

The entire suite runs without network or credentials by injecting fake
Challenger/Judge models and using a deterministic Echo target, while still
exercising the real LangGraph graph, repository, and API.

## License

For portfolio/educational use.
