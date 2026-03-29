import mlflow
from mlflow.genai.scorers import Guidelines

from arxiv_curator.agent import ArxivAgent
from arxiv_curator.config import ProjectConfig

polite_tone_guideline = Guidelines(
    name="polite_tone",
    guidelines=[
        "The response must use a polite and professional tone throughout",
        "The response should be friendly and helpful without being condescending",
        "The response must avoid any dismissive or rude language"
    ],
    model="databricks:/databricks-gpt-oss-120b"
)

hook_in_post_guideline = Guidelines(
    name="hook_in_post",
    guidelines=[
        "The response must start with an engaging hook that captures attention",
        "The opening should make the reader want to continue reading",
        "The response should have a compelling introduction before diving into details"
    ],
    model="databricks:/databricks-gpt-oss-120b"
)


@mlflow.genai.scorer
def word_count_check(outputs: str) -> bool:
    """Check that the output is under 350 words."""
    return len(outputs.split()) < 350


def evaluate_agent(
    cfg: ProjectConfig, eval_inputs_path: str
) -> mlflow.models.EvaluationResult:
    """Run evaluation on the agent.

    Args:
        cfg: Project configuration.
        eval_inputs_path: Path to evaluation inputs file.

    Returns:
        MLflow EvaluationResult with metrics.
    """
    agent = ArxivAgent(
        llm_endpoint=cfg.llm_endpoint,
        system_prompt=cfg.system_prompt,
        catalog=cfg.catalog,
        schema=cfg.schema,
        genie_space_id=cfg.genie_space_id,
        lakebase_project_id=cfg.lakebase_project_id,
    )

    with open(eval_inputs_path) as f:
        eval_data = [
            {"inputs": {"question": line.strip()}}
            for line in f if line.strip()
        ]

    def predict_fn(question: str) -> str:
        request = {"input": [{"role": "user", "content": question}]}
        result = agent.predict(request)
        return result.output[-1].content

    return mlflow.genai.evaluate(
        predict_fn=predict_fn,
        data=eval_data,
        scorers=[word_count_check, polite_tone_guideline, hook_in_post_guideline],
    )
