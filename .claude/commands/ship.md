Commit all staged and unstaged changes, then push to the remote branch.

## Steps

### 1. Check the current branch

Run:
```bash
git branch --show-current
```

If the current branch is `main` or `master`, **stop immediately** and tell the user:
> "Refusing to commit directly to `main`. Please switch to a feature branch first."

### 2. Inspect what changed

Run these in parallel:
```bash
git status
git diff HEAD
git log --oneline -5
```

Use the output to understand:
- Which files changed (staged + unstaged)
- The nature of each change (new feature, fix, refactor, docs, chore, etc.)
- The recent commit style in this repo (prefix conventions, tone, length)

### 3. Stage all changes

```bash
git add -A
```

### 4. Craft the commit message

Write a commit message with this structure:

```
<type>(<scope>): <short imperative summary> (50 chars max)

<blank line>

<body — what changed and why, not how. Wrap at 72 chars.
  Use bullet points for multiple logical changes.
  Reference filenames or modules when helpful.>

<blank line>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

**Type** — pick the most accurate:
- `feat`: new capability added
- `fix`: bug corrected
- `refactor`: restructured without behaviour change
- `chore`: tooling, config, dependencies, CI
- `docs`: documentation only
- `style`: formatting, linting (no logic change)
- `test`: tests added or updated

**Scope** — the affected area in parentheses, e.g. `(examples)`, `(executor)`, `(models)`, `(cli)`, `(deps)`. Omit if the change is truly cross-cutting.

**Body** rules:
- Explain *why* the change was made, not just *what* was changed
- Group related file changes under the same bullet
- Be specific: name the module, class, or function when it clarifies intent
- Do not exceed 72 characters per line

**Examples of good messages:**

```
feat(models): add init_environment() for widget-based env resolution

- Reads env from dbutils.widgets.get("env") first, falls back to
  DATABRICKS_ENV env var, then defaults to "dev"
- Replaces manual set_current_environment() calls in all examples
- Exported from brickkit.__init__ so callers need only one import

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

```
chore(examples): replace print() with loguru logger across all notebooks

All 17 example files updated; loguru==0.7.3 added to project deps.
Loguru provides structured output that works in both local and
Databricks notebook contexts.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

### 5. Commit

```bash
git commit -m "$(cat <<'EOF'
<your message here>
EOF
)"
```

If the commit is rejected by a pre-commit hook, fix the reported issue, re-stage, and create a **new** commit (never amend).

### 6. Push

```bash
git push -u origin HEAD
```

Report the output, including the remote URL if printed by git.
