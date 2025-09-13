import pandas as pd
import yaml
from loguru import logger
from pyspark.sql import SparkSession

from hotel_booking.config import ProjectConfig
from hotel_booking.data.data_processor import DataProcessor
from hotel_booking.utils.common import create_parser

args = create_parser()

project_config = ProjectConfig.from_yaml(config_path=f"{args.root_path}/files/project_config.yml", env=args.env)
logger.info("Configuration loaded:")
logger.info(yaml.dump(project_config, default_flow_style=False))

spark = SparkSession.builder.getOrCreate()

# Load the dataset
df = pd.read_csv(f"{args.root_path}/files/data/booking.csv")

# Initialize DataProcessor
data_processor = DataProcessor(df=df, config=project_config, spark=spark)
data_processor.preprocess()
data_processor.generate_synthetic_df(n=1000, max_date=None)
data_processor.save_to_catalog()
