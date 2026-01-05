from loguru import logger
from pyspark.sql import SparkSession

from hotel_booking.config import ProjectConfig
from hotel_booking.monitoring import MonitoringManager
from hotel_booking.utils.common import create_parser

args = create_parser()

cfg = ProjectConfig.from_yaml(
    config_path=f"{args.root_path}/files/project_config.yml", env=args.env
)

spark = SparkSession.builder.getOrCreate()

logger.info("Initializing MonitoringManager")
manager = MonitoringManager(spark=spark, config=cfg)

logger.info("Starting monitoring table update")
num_records = manager.update_monitoring_table()

if num_records > 0:
    logger.info(f"Successfully processed {num_records} records")
    logger.info("Triggering monitor refresh")
    manager.refresh_monitor()
    logger.info("Monitor refresh completed")
else:
    logger.info("No new records to process")
