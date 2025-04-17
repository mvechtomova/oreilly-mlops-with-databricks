import argparse

import pandas as pd
import yaml
from loguru import logger
from pyspark.sql import SparkSession

from hotel_booking.config import ProjectConfig
from hotel_booking.data_processor import DataProcessor

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

config = ProjectConfig.from_yaml(config_path=f"{args.root_path}/files/project_config.yml", env=args.env)

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

spark = SparkSession.builder.getOrCreate()

# Load the dataset
df = pd.read_csv(f"{args.root_path}/files/data/booking.csv")

# Initialize DataProcessor
data_processor = DataProcessor(df, config, spark)
data_processor.preprocess()

# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)
