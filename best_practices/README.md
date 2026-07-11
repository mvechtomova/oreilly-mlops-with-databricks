# MLOps / LLMOps best practices and evaluation kit

This folder distills the engineering practices demonstrated in [mlops/](../mlops/) and
[llmops/](../llmops/) into a reusable rubric. It is meant for evaluating repositories
that students build for their own use cases: check whether they follow the same
practices, not whether they copied the same code.

## Contents

| File | What it is |
|------|------------|
| [foundations.md](foundations.md) | Practices shared by both tracks: repo layout, packaging, config, code quality, testing, CI/CD, Declarative Automation Bundles, cost attribution |
| [mlops.md](mlops.md) | Classical ML track: data reproducibility, training and tracking, registry and promotion, serving, monitoring, alerting |
| [llmops.md](llmops.md) | GenAI/agent track: agent architecture, tracing, evaluation, deployment and auth, vector search, production monitoring |
| [checklist.md](checklist.md) | The evaluation checklist: every practice as a checkable item with a "how to verify" hint and a scoring scheme |

There is also a Claude Code skill, [.claude/skills/evaluate-repo](../.claude/skills/evaluate-repo/SKILL.md),
that automates the evaluation: point it at a student repository and it produces a
scored report against `checklist.md`.

## How to use for grading

1. Decide the track: MLOps, LLMOps, or both. Foundations always apply.
2. Go through `checklist.md` area by area. For each item record a score with evidence
   (a file path or a quoted line), not just a checkmark.
3. Items are tagged **[core]** or **[advanced]**. A solid repo has all core items;
   advanced items distinguish excellent work.
4. Or automate it: run the `evaluate-repo` skill from a Claude Code session, passing
   the path to the student repo.

## What this rubric is not

- It is not a style guide for the exact tool choices. `uv`, `loguru`, or `pydantic`
  can be swapped for equivalents; the practice being graded is "locked, reproducible
  environments", "structured logging", "validated typed config".
- It is not a completeness check against the book chapters. A student repo with a
  smaller scope but sound engineering should score well on the areas it covers.
