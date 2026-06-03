"""Shared fixtures for the hotel-booking test suite.

These fixtures deliberately avoid any Spark or Databricks dependency so the unit
layer runs on a plain GitHub runner. Anything that needs a real runtime belongs
in tests/integration and is marked with @pytest.mark.integration.
"""

import pandas as pd
import pytest

from hotel_booking.config import ProjectConfig


@pytest.fixture
def cfg() -> ProjectConfig:
    """The real project config, resolved for the dev environment.

    pytest's rootdir is the mlops/ project (set via pyproject.toml), so the
    config path is relative to it.
    """
    return ProjectConfig.from_yaml(config_path="project_config.yml", env="dev")


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """A tiny DataFrame mirroring the raw booking.csv schema (pre-preprocess).

    Column names use the original spaces/hyphens so tests exercise the rename in
    DataProcessor.preprocess. Includes the 2018-2-29 leap-year row that the
    preprocessing code special-cases.
    """
    return pd.DataFrame(
        {
            "Booking_ID": ["INN00001", "INN00002", "INN00003"],
            "number of adults": [1, 2, 1],
            "number of children": [0, 1, 0],
            "number of weekend nights": [1, 2, 0],
            "number of week nights": [2, 3, 1],
            "type of meal": ["Meal Plan 1", "Not Selected", "Meal Plan 1"],
            "car parking space": [0, 1, 0],
            "room type": ["Room_Type 1", "Room_Type 1", "Room_Type 2"],
            "lead time": [10, 5, 0],
            "market segment type": ["Offline", "Online", "Online"],
            "repeated": [0, 0, 1],
            "P-C": [0, 0, 0],
            "P-not-C": [0, 0, 0],
            "average price": [88.0, 106.68, 50.0],
            "special requests": [0, 1, 0],
            # row 2 carries the malformed leap-year date the code rewrites to 3/1/2018
            "date of reservation": ["10/2/2015", "2018-2-29", "11/6/2018"],
            "booking status": ["Not_Canceled", "Not_Canceled", "Canceled"],
        }
    )


@pytest.fixture
def training_df() -> pd.DataFrame:
    """A small, fully-numeric/categorical frame ready to train the pipeline on.

    Columns match cfg.num_features + cfg.cat_features + cfg.target so it can be
    sliced the same way DataLoader.split does.
    """
    n = 24
    return pd.DataFrame(
        {
            "number_of_adults": [1, 2] * (n // 2),
            "number_of_children": [0, 1] * (n // 2),
            "number_of_weekend_nights": [1, 2] * (n // 2),
            "number_of_week_nights": [2, 3] * (n // 2),
            "car_parking_space": [0, 1] * (n // 2),
            "special_requests": [0, 1] * (n // 2),
            "lead_time": list(range(n)),
            "type_of_meal": ["Meal Plan 1", "Not Selected"] * (n // 2),
            "room_type": ["Room_Type 1", "Room_Type 2"] * (n // 2),
            "arrival_month": [1, 6] * (n // 2),
            "market_segment_type": ["Offline", "Online"] * (n // 2),
            "average_price": [80.0 + i for i in range(n)],
        }
    )
