# Figure 5.17: Asynchronous integration testing with a merge gate

The orchestrated ML pipeline can run as part of CI, on a small data sample, to
confirm the whole thing wires together. A run that samples data, assembles
features, trains, evaluates, registers, and deploys a model exercises every
wiring decision in a way no isolated test can.

The obstacle is duration. Even on sampled data, a run takes long enough that you
do not want a GitHub runner burning minutes to poll it. The pattern that avoids
this is asynchronous. The CI workflow validates the bundle, deploys it to the
test target, triggers the job run, and exits green. Total runner time is a
couple of minutes. The commit SHA is passed in as a job parameter so the result
can be linked back to the right commit.

The run reports its own result through two tasks at the end of the job. One runs
with run_if: ALL_SUCCESS and posts success. The other runs with run_if:
AT_LEAST_ONE_FAILED and posts failure. Either way the job calls the GitHub
Commit Status API to mark the commit with one of error, failure, pending, or
success. That state surfaces on any pull request involving the commit. Include
target_url and
description: the URL points to the Databricks run output, the description
summarizes what happened, and both make the status useful in the GitHub UI. The
context field names the service reporting the status, for example
databricks/pipeline-integration, so this check stays distinct from other checks
on the same commit. For the token, the repo:status OAuth scope grants access to
statuses without granting access to repository code.

Branch protection turns that status into a merge gate. Mark the status context
as a required status check. The pull request stays pending until the run reports
back, and merging is blocked until the check is green, long after the CI
workflow finished.

This decouples "CI is fast" from "merge is safe." The workflow always succeeds
quickly. The required status check gates the merge on the actual pipeline
result.

## Diagram layout

Left side, the integration-testing pipeline as a Lakeflow job on Databricks. The
linear steps run top to bottom:

- Preprocessing (sample data)
- Model training & validation
- Deploy model

The job then ends in two report tasks that are NOT wired into the linear chain
(they run conditionally on the overall job state):

- Report success, run_if: ALL_SUCCESS
- Report failure, run_if: AT_LEAST_ONE_FAILED

Right side, GitHub:

- GitHub Actions CI, a 3-step flow:
  - Unit tests
  - Bundle validate & integration test run (this step triggers the Lakeflow job)
  - Status check: pending (sets the commit status to pending, then CI exits
    green)
- Branch protection (merge gate), a required-checks block:
  - Successful CI
  - Approvals
  - Status check (databricks/pipeline-integration)

Cross-links:

- Bundle validate & integration test run -> triggers the Lakeflow job run (commit
  SHA passed as a job parameter). Drawn as a horizontal arrow into Preprocessing.
- The two report tasks -> both call the Commit Status API and report
  success/failure into the Status check inside branch protection (a converging
  fan-in).
