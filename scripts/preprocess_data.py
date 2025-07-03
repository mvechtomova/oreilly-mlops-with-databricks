import argparse

import pandas as pd
import yaml
from loguru import logger
from pyspark.sql import SparkSession

from hotel_booking.config import ProjectConfig
from hotel_booking.data.data_processor import DataProcessor

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default="dev",
    type=str,
    required=True,
)

args = parser.parse_args()

project_config = ProjectConfig.from_yaml(config_path=f"{args.root_path}/files/project_config.yml", env=args.env)

logger.info("Configuration loaded:")
logger.info(yaml.dump(project_config, default_flow_style=False))

spark = SparkSession.builder.getOrCreate()

# Load the dataset
df = pd.read_csv(f"{args.root_path}/files/data/booking.csv")

# Initialize DataProcessor
data_processor = DataProcessor(df=df, config=project_config, spark=spark)
data_processor.preprocess()
data_processor.save_to_catalog()
