"""Unit tests for split/query logic. Spark is mocked via pytest-mock's `mocker`."""
# test fixtures (mocker, cfg, ...) are conventionally un-annotated
# ruff: noqa: ANN001, ANN201, ANN202

from datetime import datetime

from hotel_booking.data.data_loader import DataLoader


def test_get_start_end_dates_window_math(cfg) -> None:
    loader = DataLoader(config=cfg, spark=None)  # pure helper, no Spark needed
    train_start, train_end, test_start, test_end = loader._get_start_end_dates(
        max_date=datetime(2024, 12, 15), test_months=1, train_months=12
    )
    # month_start = 2024-12-01; 12 months back = 2023-12-01; train_end = day before.
    assert train_start == "2023-12-01"
    assert train_end == "2024-11-30"
    # test_months=1 -> test_start = month_start; test_end = max_date.
    assert test_start == "2024-12-01"
    assert test_end == "2024-12-15"


def test_generate_queries_embeds_version_and_window(mocker, cfg) -> None:
    spark = mocker.Mock()
    spark.sql.return_value.collect.return_value = [{"max_date": datetime(2024, 12, 15)}]
    mocker.patch(
        "hotel_booking.data.data_loader.get_delta_table_version",
        return_value="7",
        autospec=True,
    )

    loader = DataLoader(config=cfg, spark=spark)
    train_q, test_q = loader.generate_queries(test_months=1, train_months=12)

    assert "VERSION AS OF 7" in train_q
    assert "VERSION AS OF 7" in test_q
    assert "BETWEEN DATE('2023-12-01') AND DATE('2024-11-30')" in train_q
    assert "BETWEEN DATE('2024-12-01') AND DATE('2024-12-15')" in test_q
    # the table reference is fully qualified from config
    assert "mlops_dev.hotel_booking.hotel_booking" in train_q


def test_generate_queries_honors_explicit_max_date_and_version(mocker, cfg) -> None:
    # When max_date and version are passed, Spark must not be queried for them.
    spark = mocker.Mock()
    version_spy = mocker.patch(
        "hotel_booking.data.data_loader.get_delta_table_version", autospec=True
    )

    loader = DataLoader(config=cfg, spark=spark)
    train_q, _ = loader.generate_queries(max_date=datetime(2024, 6, 30), version=3)

    spark.sql.assert_not_called()
    version_spy.assert_not_called()
    assert "VERSION AS OF 3" in train_q


def test_split_slices_features_and_target_from_spark_results(
    mocker, cfg, training_df
) -> None:
    # split() runs the generated queries through Spark and slices the resulting
    # pandas frames into (X_train, y_train, X_test, y_test). Spark is mocked: the
    # two spark.sql calls return frames whose toPandas() yields training_df.
    full = training_df.copy()
    full[cfg.target] = [80.0 + i for i in range(len(full))]

    spark = mocker.Mock()
    spark.sql.return_value.toPandas.return_value = full
    mocker.patch.object(
        DataLoader,
        "generate_queries",
        return_value=("TRAIN_QUERY", "TEST_QUERY"),
        autospec=True,
    )

    loader = DataLoader(config=cfg, spark=spark)
    X_train, y_train, X_test, y_test = loader.split(max_date=datetime(2024, 12, 15))

    spark.sql.assert_any_call("TRAIN_QUERY")
    spark.sql.assert_any_call("TEST_QUERY")
    # X frames carry only the model features; y is the target column.
    assert list(X_train.columns) == cfg.num_features + cfg.cat_features
    assert y_train.name == cfg.target
    assert list(X_test.columns) == cfg.num_features + cfg.cat_features
    assert y_test.name == cfg.target
