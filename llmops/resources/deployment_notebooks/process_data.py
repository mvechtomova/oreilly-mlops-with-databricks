# Databricks notebook source
import yaml
from loguru import logger
from pyspark.sql import SparkSession

from arxiv_curator.config import ProjectConfig
from arxiv_curator.data_processor import DataProcessor
from arxiv_curator.utils.common import get_widget
from arxiv_curator.vector_search import VectorSearchManager

env = get_widget("env", "dev")

project_config = ProjectConfig.from_yaml(
    "../../project_config.yml", env=env)
logger.info("Configuration loaded:")
logger.info(yaml.dump(project_config, default_flow_style=False))

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

# Process data
data_processor = DataProcessor(
    config=project_config,
    spark=spark)
records = data_processor.download_and_store_papers()

if records is None:
    logger.info("No new papers to process. Exiting.")
    pass
else:
    data_processor.parse_pdfs_with_ai()
    logger.info("Parseed documents.")

    data_processor.process_chunks()
    logger.info("Processing complete!")

# Sync vector search index
vector_search_manager = VectorSearchManager(config=project_config)
vector_search_manager.sync_index()
