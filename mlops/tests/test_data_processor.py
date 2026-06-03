"""Unit tests for the pure pandas preprocessing logic (no Spark)."""
# test fixtures (mocker, cfg, ...) are conventionally un-annotated
# ruff: noqa: ANN001, ANN201, ANN202

from datetime import datetime

import pandas as pd
import pytest

from hotel_booking.data.data_processor import DataProcessor


def _processor(raw_df: pd.DataFrame, cfg) -> DataProcessor:
    # spark=None is fine: preprocess() never touches the Spark session.
    return DataProcessor(df=raw_df, config=cfg, spark=None)


def test_preprocess_renames_spaced_and_hyphenated_columns(raw_df, cfg) -> None:
    dp = _processor(raw_df, cfg)
    dp.preprocess()
    assert "lead_time" in dp.df.columns
    assert "number_of_adults" in dp.df.columns
    assert "P_C" in dp.df.columns and "P_not_C" in dp.df.columns
    # no original spaced/hyphenated names survive
    assert not any((" " in c or "-" in c) for c in dp.df.columns)


def test_preprocess_rewrites_invalid_leap_year_date(raw_df, cfg) -> None:
    # The "2018-2-29" row must be rewritten to 3/1/2018 instead of raising.
    dp = _processor(raw_df, cfg)
    dp.preprocess()
    assert dp.df.loc[1, "date_of_reservation"] == datetime(2018, 3, 1)


def test_preprocess_computes_arrival_date_and_month(raw_df, cfg) -> None:
    dp = _processor(raw_df, cfg)
    dp.preprocess()
    row0 = dp.df.iloc[0]
    # 10/2/2015 + 10 days lead_time = 2015-10-12
    assert row0["arrival_date"] == datetime(2015, 10, 12)
    assert row0["arrival_month"] == 10


def test_preprocess_raises_on_unparseable_date(raw_df, cfg) -> None:
    raw_df.loc[0, "date of reservation"] = "not-a-date"
    dp = _processor(raw_df, cfg)
    with pytest.raises(ValueError):
        dp.preprocess()


def test_target_table_uses_catalog_and_schema(raw_df, cfg) -> None:
    dp = _processor(raw_df, cfg)
    assert dp.target_table == "mlops_dev.hotel_booking.hotel_booking"
