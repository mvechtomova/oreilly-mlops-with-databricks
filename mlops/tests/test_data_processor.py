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


def _patch_sdv(mocker, sampled: pd.DataFrame):
    """autospec the SDV surface so a renamed class/method/kwarg fails the test.

    Returns the autospec'd synthesizer instance with .sample stubbed to echo
    ``sampled``. autospec validates Metadata.detect_from_dataframe, the
    GaussianCopulaSynthesizer constructor, and the .fit/.sample signatures.
    """
    mocker.patch(
        "hotel_booking.data.data_processor.Metadata.detect_from_dataframe",
        autospec=True,
    )
    synth_cls = mocker.patch(
        "hotel_booking.data.data_processor.GaussianCopulaSynthesizer", autospec=True
    )
    synthesizer = synth_cls.return_value
    synthesizer.sample.return_value = sampled
    return synthesizer


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


def test_save_to_catalog_appends_and_enables_cdf(mocker, raw_df, cfg) -> None:
    spark = mocker.Mock()
    dp = DataProcessor(df=raw_df, config=cfg, spark=spark)
    dp.save_to_catalog()

    # The frame is written in append mode to the fully-qualified target table.
    spark.createDataFrame.assert_called_once_with(raw_df)
    writer = spark.createDataFrame.return_value.write
    writer.mode.assert_called_once_with("append")
    writer.mode.return_value.saveAsTable.assert_called_once_with(dp.target_table)

    # A follow-up ALTER TABLE turns on Change Data Feed.
    alter_sql = spark.sql.call_args.args[0]
    assert dp.target_table in alter_sql
    assert "delta.enableChangeDataFeed = true" in alter_sql


def test_generate_synthetic_df_builds_month_after_max_date(mocker, raw_df, cfg) -> None:
    # SDV is heavy and non-deterministic, so the synthesizer is mocked. It just
    # echoes a frame the size of the requested sample; the date/arrival logic is
    # what we actually assert on.
    dp = _processor(raw_df, cfg)
    dp.preprocess()  # gives self.df the lead_time column the method needs

    sampled = dp.df.copy()
    synthesizer = _patch_sdv(mocker, sampled)

    n = len(sampled)
    # max_date in November -> the populated month must be the following December.
    dp.generate_synthetic_df(n=n, max_date=datetime(2024, 11, 15))

    synthesizer.fit.assert_called_once()
    synthesizer.sample.assert_called_once_with(num_rows=n)
    assert (dp.df["date_of_reservation"].dt.month == 12).all()
    assert (dp.df["date_of_reservation"].dt.year == 2024).all()
    # arrival_month is derived from date_of_reservation + lead_time.
    assert "arrival_month" in dp.df.columns


def test_generate_synthetic_df_queries_max_date_when_not_given(
    mocker, raw_df, cfg
) -> None:
    # With max_date omitted, the latest date is read from the target table.
    spark = mocker.Mock()
    spark.sql.return_value.collect.return_value = [{"max_date": datetime(2024, 11, 15)}]
    dp = DataProcessor(df=raw_df, config=cfg, spark=spark)
    dp.preprocess()

    sampled = dp.df.copy()
    _patch_sdv(mocker, sampled)

    dp.generate_synthetic_df(n=len(sampled))

    assert dp.target_table in spark.sql.call_args.args[0]
    assert (dp.df["date_of_reservation"].dt.month == 12).all()
