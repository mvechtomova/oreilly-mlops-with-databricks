---
name: evaluate-repo
description: Evaluate a student MLOps/LLMOps repository against the course best-practices checklist and produce a scored report. Trigger whenever the user asks to evaluate, grade, review, or score a student repo (or any external repo) against the best practices, the checklist, or the course rubric, e.g. "evaluate ~/repos/student-x", "grade this repo", "check whether this repo follows the best practices".
---

# evaluate-repo

Evaluate a repository against the rubric in [best_practices/](../../../best_practices/):
`checklist.md` is the instrument; `foundations.md`, `mlops.md`, and `llmops.md`
explain each practice and point at reference implementations in this repo.

The argument is the path to the repository to evaluate. Optionally followed by a
track: `mlops`, `llmops`, or `both`. If no path is given, ask for one. Never modify
the target repository; this is a read-only review.

## Steps

### 1. Load the rubric

Read `best_practices/checklist.md` from THIS repo (the course repo, where this skill
lives). Skim `foundations.md`, `mlops.md`, and `llmops.md` as needed to interpret
items. Grade practices, not tools: poetry instead of uv, GitLab CI instead of GitHub
Actions, plain `logging` instead of loguru all satisfy the corresponding items.

### 2. Scope the target repo

Survey the target: top-level layout, `pyproject.toml`, lockfile, `databricks.yml`,
`resources/`, `src/`, `tests/`, notebooks, CI config (`.github/workflows/` or
equivalent). If the repo has multiple projects (monorepo), evaluate each or ask which.

Detect the track if not given:
- MLOps signals: sklearn/lightgbm/xgboost deps, `mlflow.models.evaluate`,
  lakehouse monitoring, feature engineering.
- LLMOps signals: `ResponsesAgent`/`ChatAgent`, `databricks-agents`, vector search,
  `mlflow.genai`, MCP, prompts in config.
- Both signal sets present: evaluate both tracks. Foundations always apply.

### 3. Verify checklist items with evidence

Work through every applicable area (F1-F8, plus M1-M7 and/or L1-L8). For each item,
gather concrete evidence: a file path and line, a grep hit, or a command output.
Useful probes (adapt paths):

- Layout: `find <repo> -maxdepth 3` filtered of caches; check `src/` layout in
  `pyproject.toml`.
- Hardcoding: grep `src/`, scripts, and resource YAML for literal catalog names,
  workspace URLs, user emails/home paths.
- Secrets: grep for `dapi`, `client_secret`, `password`, token-like strings; check
  `.gitignore` covers `.env`; spot-check `git log -p` for leaked values if suspicious.
- Mock discipline: grep tests for `autospec=True` and for patches without it.
- Cost: every file under `resources/` must show `budget_policy_id`, `tags:`, or
  `custom_tags`.
- Schedules: grep resources for `pause_status` and check the prd target override.
- CI path filters: `paths: ['folder']` does NOT match folder contents; it must be
  `folder/**`. Flag this, it is a common silent-CI failure.
- Aliases: grep for `set_registered_model_alias` / `@` alias resolution vs hardcoded
  version numbers or legacy stages.

Where safe and cheap, run things instead of only reading: the linter, and the unit
test suite in an isolated environment (`uv run pytest` or equivalent) with network
access assumed absent. Do not run anything that deploys, calls a workspace, or needs
credentials.

If the repo is large, fan out Explore subagents per area group (foundations /
mlops-track / llmops-track) and have each return per-item evidence; verify surprising
claims yourself before scoring.

### 4. Score

Per area: 2 (all core items evidenced), 1 (partial), 0 (absent). Advanced items never
reduce a score; list them as distinctions. When evidence is ambiguous, score down and
say why, quoting what was found.

### 5. Report

Produce the report in this order:

1. **Summary**: track(s), total score as `X / max` for applicable areas, one-paragraph
   verdict of overall maturity.
2. **Scorecard table**: area | score | one-line verdict.
3. **Top findings**: the 5-10 most impactful gaps, each with evidence (file:line), why
   it matters (one sentence, may reference the best-practices doc), and the concrete
   fix, pointing at the reference implementation in this repo (e.g. "see
   mlops/src/hotel_booking/models/serving.py for the idempotent create-or-update
   pattern").
4. **Distinctions**: advanced items achieved.
5. **Per-area detail**: for each area, the failed or partial core items with evidence.
   Passed items need no commentary beyond the evidence path.

Keep the tone constructive: this is feedback to a student, not a gate. Lead with what
is done well before the gaps.
