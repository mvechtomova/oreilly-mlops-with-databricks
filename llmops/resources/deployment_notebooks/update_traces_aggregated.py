# Databricks notebook source
import mlflow
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp

from arxiv_curator.config import ProjectConfig
from arxiv_curator.evaluation import (
    hook_in_post_guideline,
    polite_tone_guideline,
    word_count_check,
)
from arxiv_curator.utils.common import get_widget, set_mlflow_tracking_uri

set_mlflow_tracking_uri()

env = get_widget("env", "dev")
cfg = ProjectConfig.from_yaml("../../project_config.yml", env=env)
mlflow.set_experiment(cfg.experiment_path)

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

catalog = cfg.catalog
schema = cfg.schema

traces_table = f"{catalog}.{schema}.arxiv_traces"
aggregated_table = f"{catalog}.{schema}.arxiv_traces_aggregated"

# COMMAND ----------

# Create the aggregated table if it does not exist
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {aggregated_table} (
        trace_id STRING,
        request_time TIMESTAMP,
        latency_seconds DOUBLE,
        call_llm_exec_count LONG,
        tool_call_count LONG,
        total_tokens_used LONG,
        processed_ts TIMESTAMP,
        word_count_check STRING,
        polite_tone STRING,
        hook_in_post STRING
    )
""")

# COMMAND ----------
# Get new traces not yet in the aggregated table
new_traces_df = spark.sql(f"""
    SELECT
        trace_id,
        request_time,
        request_preview,
        response_preview,
        execution_duration_ms / 1000.0 AS latency_seconds,
        COUNT(IF(s.name = 'call_llm', 1, NULL)) AS call_llm_exec_count,
        COUNT(IF(s.name = 'execute_tool', 1, NULL)) AS tool_call_count,
        SUM(
            IF(
                s.name = 'call_llm',
                CAST(
                    get_json_object(
                        get_json_object(
                            s.attributes['mlflow.spanOutputs'],
                            '$.usage'
                        ),
                        '$.total_tokens'
                    ) AS INT
                ),
                0
            )
        ) AS total_tokens_used
    FROM {traces_table}
    LATERAL VIEW explode(spans) AS s
    WHERE tags['model_serving_endpoint_name'] = 'arxiv-agent-endpoint'
      AND request_time > (
        SELECT COALESCE(MAX(request_time), CAST('1970-01-01' AS TIMESTAMP))
        FROM {aggregated_table}
    )
    GROUP BY trace_id, request_time, execution_duration_ms, request_preview, response_preview
""")

# COMMAND ----------

traces_pdf = new_traces_df.toPandas()
logger.info(f"New traces to evaluate: {len(traces_pdf)}")

eval_pdf = traces_pdf.assign(
    inputs=traces_pdf["request_preview"].apply(lambda x: {"query": x}),
    outputs=traces_pdf["response_preview"],
)

# COMMAND ----------
# Run word_count_check on all traces

wc_result = mlflow.genai.evaluate(
    data=eval_pdf[["inputs", "outputs"]],
    scorers=[word_count_check],
)
traces_pdf["word_count_check"] = wc_result.result_df["assessments"].apply(
    lambda a: a[0]["feedback"]["value"]
)

# COMMAND ----------
# Run LLM-judge scorers on a 10% sample

sample_size = max(1, int(len(traces_pdf) * 0.1))
sampled_idx = traces_pdf.sample(n=sample_size).index
logger.info(f"Sampled {len(sampled_idx)} traces for LLM-judge evaluation")

sampled_pdf = eval_pdf.loc[sampled_idx, ["inputs", "outputs"]]

llm_result = mlflow.genai.evaluate(
    data=sampled_pdf,
    scorers=[polite_tone_guideline, hook_in_post_guideline],
)


def extract_score(assessments: list, name: str) -> object:
    """Extract feedback value for a given scorer name."""
    for a in assessments:
        if a["assessment_name"] == name:
            return a["feedback"]["value"]
    return None


traces_pdf.loc[sampled_idx, "polite_tone"] = (
    llm_result.result_df["assessments"]
    .apply(lambda a: extract_score(a, "polite_tone"))
    .values
)
traces_pdf.loc[sampled_idx, "hook_in_post"] = (
    llm_result.result_df["assessments"]
    .apply(lambda a: extract_score(a, "hook_in_post"))
    .values
)

# COMMAND ----------

final_df = (
    spark.createDataFrame(traces_pdf)
    .withColumn("processed_ts", current_timestamp())
)

final_df.write.mode("append").saveAsTable(aggregated_table)
logger.info(f"Written {len(traces_pdf)} rows to {aggregated_table}.")

# COMMAND ----------
