# Databricks notebook source
import random
import time
from datetime import datetime

from databricks.sdk import WorkspaceClient
from openai import OpenAI

workspace = WorkspaceClient()
host = workspace.config.host
token = workspace.tokens.create(lifetime_seconds=2000).token_value

endpoint_name = "arxiv-agent-endpoint"

client = OpenAI(
    api_key=token,
    base_url=f"{host}/serving-endpoints",
)

# COMMAND ----------
queries = [
    "What are the most common topics in recent LLM research?",
    "Summarize trending themes in NLP papers.",
    "What topics dominate reinforcement learning research?",
    "List popular research areas in computer vision.",
    "What are the main themes in AI safety papers?",
    "Summarize key topics in transformer architecture research.",
    "What subjects appear most in recent RAG papers?",
    "List common themes in fine-tuning research.",
    "What are popular topics in multi-agent systems?",
    "Summarize trending areas in code generation research.",
    "What are the main research directions in RLHF?",
    "List frequent topics in knowledge distillation papers.",
    "What themes dominate diffusion model research?",
    "Summarize common topics in prompt engineering papers.",
    "What are trending subjects in embedding model research?",
    "List popular areas in federated learning papers.",
    "What topics are most discussed in AI alignment?",
    "Summarize key themes in synthetic data research.",
    "What are common research directions in sparse models?",
    "List trending topics in continual learning papers.",
    "What subjects appear most in text-to-SQL research?",
    "Summarize popular themes in vision-language models.",
    "What are the main topics in reward modeling papers?",
    "List common areas in neural architecture search.",
    "What themes are trending in multilingual NLP?",
    "Summarize key topics in graph neural network papers.",
    "What are popular research directions in LLM efficiency?",
    "List frequent themes in long-context model research.",
    "What topics dominate agentic AI papers?",
    "Summarize trending areas in inference optimization research.",
]

# COMMAND ----------
for i, query in enumerate(queries):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
    request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

    print(f"[{i + 1}/30] {query[:60]}...")
    response = client.responses.create(
        model=endpoint_name,
        input=[
            {"role": "user", "content": query}
        ],
        extra_body={"custom_inputs": {
            "session_id": session_id,
            "request_id": request_id,
        }},
    )
    time.sleep(2)

# COMMAND ----------
