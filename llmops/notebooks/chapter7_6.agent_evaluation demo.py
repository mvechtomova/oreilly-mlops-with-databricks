# Databricks notebook source
import mlflow
from mlflow.genai.scorers import Guidelines

from arxiv_curator.utils.common import set_mlflow_tracking_uri

set_mlflow_tracking_uri()

# COMMAND ----------

# Example using Guidelines for escalation handling
mlflow.set_experiment("/Shared/guidelines-example")
escalation_guidelines = Guidelines(
    name="escalation_handling",
    guidelines=[
        "When a user indicates previous attempts failed, the response must "
        "acknowledge their efforts and either escalate or offer a new approach"
    ],
    model="databricks:/databricks-gpt-oss-120b"
)

data = [
    {
        "inputs": {"message": "Tried everything you suggested"},
        "outputs": "Have you tried restarting?",
    },
    {
        "inputs": {"message": "Tried everything you suggested"},
        "outputs": (
            "I understand you've already tried the previous suggestions "
            "without success. Let me escalate this to our senior support "
            "team who can look into this more deeply. In the meantime, "
            "could you share any error logs you've encountered?"
        ),
    },
]

mlflow.genai.evaluate(data=data, scorers=[escalation_guidelines])

# COMMAND ----------
import mlflow
from mlflow.genai.judges import make_judge

# Example using make_judge for escalation quality evaluation (1-5 scale)
mlflow.set_experiment("/Shared/make-judge-example")
escalation_judge = make_judge(
    name="escalation_quality",
    instructions=(
        "Evaluate how well the response in {{ outputs }} handles the user's "
        "message in {{ inputs }} when they indicate previous attempts failed. "
        "Score from 1 to 5:\n"
        "1 - Ignores context, repeats already-tried suggestions\n"
        "2 - Acknowledges frustration but offers no new solutions\n"
        "3 - Offers a new approach but lacks empathy or clarity\n"
        "4 - Acknowledges efforts and offers a reasonable new approach\n"
        "5 - Empathetic, escalates appropriately or provides creative solution"
    ),
    model="databricks:/databricks-gpt-oss-120b",
    feedback_value_type=int,
)

mlflow.genai.evaluate(data=data, scorers=[escalation_judge])

# COMMAND ----------
experiment_id = mlflow.search_experiments(
    filter_string="name='/Shared/make-judge-example'")[0].experiment_id

from mlflow.genai.judges.optimizers import SIMBAAlignmentOptimizer

# Retrieve traces with both judge and human assessments
traces_for_alignment = mlflow.search_traces(
    experiment_ids=[experiment_id],
    return_type="list"
)

# Filter for traces with both judge and human feedback
valid_traces = []
for trace in traces_for_alignment:
    feedbacks = trace.search_assessments(name="escalation_quality")
    has_judge = any(f.source.source_type == "LLM_JUDGE" for f in feedbacks)
    has_human = any(f.source.source_type == "HUMAN" for f in feedbacks)
    if has_judge and has_human:
        valid_traces.append(trace)

optimizer = SIMBAAlignmentOptimizer(
    model="databricks:/databricks-gpt-oss-120b"
)

# Align the judge based on human feedback
aligned_judge = escalation_judge.align(optimizer, valid_traces)

# COMMAND ----------

from typing import Literal

performance_judge = make_judge(
    name="performance_analyzer",
    instructions=(
        "Analyze the {{ trace }} for performance issues.\n\n"
        "Check for:\n"
        "- Operations taking longer than 2 seconds\n"
        "- Redundant API calls or database queries\n"
        "- Inefficient data processing patterns\n"
        "- Proper use of caching mechanisms\n\n"
        "Rate as: 'optimal', 'acceptable', or 'needs_improvement'"
    ),
    feedback_value_type=Literal["optimal", "acceptable", "needs_improvement"],
    model="databricks:/databricks-gpt-oss-120b",
)
