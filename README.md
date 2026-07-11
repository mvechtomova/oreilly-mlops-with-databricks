# MLOps with Databricks

<a href="https://learning.oreilly.com/library/view/mlops-with-databricks/9798341608245/">
  <img src="mlops_with_databricks_cover.jpeg" alt="MLOps with Databricks — Machine Learning End-to-End" width="300"/>
</a>

Code repository accompanying the O'Reilly [*MLOps with Databricks: Machine Learning End-to-End*](https://learning.oreilly.com/library/view/mlops-with-databricks/9798341608245/) book by Maria Vechtomova.

## Repository Structure

This repository is organised into two self-contained projects, one per part of the book:

---

### [mlops/](mlops/)

Covers **Chapters 2-6** of the book. Demonstrates a complete ML lifecycle for a hotel booking price prediction use case, built on LightGBM, MLflow, Unity Catalog, and Declarative Automation Bundles.

| Chapter | Topic |
|---------|-------|
| 2 | Developing on Databricks — data preprocessing |
| 3 | Experiment tracking in MLflow, model training, logging, and registration in Unity Catalog |
| 4 | Model serving, feature serving, and endpoint authentication |
| 5 | CI/CD with Declarative Automation Bundles |
| 6 | Data Quality monitoring |

**Bundle resources** ([mlops/resources/](mlops/resources/)) — Declarative Automation Bundle definitions:

```text
resources/
├── ml_pipeline.yml     # Lakeflow job: preprocess, train + register, deploy
├── ml_monitoring.yml   # Lakeflow job: refresh the monitoring table
├── alert.yml           # SQL alert on a monitoring metric
└── testing.yml         # notebook test jobs for the chapter demos
```

---

### [llmops/](llmops/)

Covers **Chapters 7-9** of the book. Demonstrates LLMOps patterns on Databricks, including building, evaluating, and deploying an LLM-powered agent.

| Chapter | Topic |
|---------|-------|
| 7 | Foundation models and context engineering — vector search, Genie |
| 8 | MLflow for GenAI — tracing and agent evaluation |
| 9 | Monitoring and deployment — agent deployment and production monitoring |

**Bundle resources** ([llmops/resources/](llmops/resources/)) — Declarative Automation Bundle definitions:

```text
resources/
├── process_data.yml            # Lakeflow job: process arxiv documents
├── register_deploy_agent.yml   # Lakeflow job: register & deploy the agent
├── production_monitoring.yml   # Lakeflow job: agent production monitoring
├── integration_test.yml        # Lakeflow job: end-to-end test chaining the jobs above
├── dashboard/                  # Lakeview monitoring dashboard
│   ├── agent_monitoring_dashboard.yml
│   └── agent_monitoring_dashboard.lvdash.json
└── deployment_notebooks/       # notebooks executed by the jobs above
    ├── process_data.py
    ├── log_register_agent.py
    ├── deploy_agent.py
    └── production_monitoring.py
```

---

## Getting Started

Each project uses **Python 3.12** with **uv** for dependency management. See the `README.md` inside each folder for setup instructions and environment-specific details.

```bash
# MLOps project (Chapters 2–6)
cd mlops
uv sync --extra dev

# LLMOps project (Chapters 7–9)
cd llmops
uv sync --extra dev
```
