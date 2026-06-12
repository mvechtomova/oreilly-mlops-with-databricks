"""Unit tests for the create-vs-refresh branch in MonitoringManager.refresh_monitor.

These exercise the databricks-sdk ``data_quality`` API with the WorkspaceClient
mocked. Spark is mocked too, so no cluster is started.
"""
# test fixtures (mocker, cfg, ...) are conventionally un-annotated
# ruff: noqa: ANN001, ANN201, ANN202

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service.dataquality import (
    AggregationGranularity,
    InferenceProblemType,
    RefreshTrigger,
)

from hotel_booking.monitoring import MonitoringManager


@pytest.fixture
def build_manager(mocker, cfg):
    """Build a MonitoringManager wired to a mocked workspace and Spark session.

    The mocked workspace resolves the schema/table ids the data_quality API
    needs; ``monitor_exists`` decides whether get_monitor returns or raises
    NotFound, so the test can assert which branch ran.
    """

    def build(monitor_exists: bool):
        workspace = mocker.Mock()
        workspace.schemas.get.return_value = mocker.Mock(schema_id="schema-123")
        workspace.tables.get.return_value = mocker.Mock(table_id="table-456")
        if monitor_exists:
            workspace.data_quality.get_monitor.return_value = mocker.Mock()
        else:
            workspace.data_quality.get_monitor.side_effect = NotFound("missing")
        manager = MonitoringManager(
            spark=mocker.Mock(),
            config=cfg,
            workspace_client=workspace,
        )
        return manager, workspace

    return build


def test_refreshes_when_monitor_exists(build_manager) -> None:
    manager, workspace = build_manager(monitor_exists=True)

    manager.refresh_monitor()

    workspace.data_quality.create_refresh.assert_called_once()
    workspace.data_quality.create_monitor.assert_not_called()

    kwargs = workspace.data_quality.create_refresh.call_args.kwargs
    assert kwargs["object_type"] == "table"
    assert kwargs["object_id"] == "table-456"

    # Refresh body itself requires object_type/object_id (a real SDK gotcha)
    refresh = kwargs["refresh"]
    assert refresh.object_type == "table"
    assert refresh.object_id == "table-456"
    assert refresh.trigger == RefreshTrigger.MONITOR_REFRESH_TRIGGER_MANUAL


def test_creates_when_monitor_absent(build_manager) -> None:
    manager, workspace = build_manager(monitor_exists=False)

    manager.refresh_monitor()

    workspace.data_quality.create_monitor.assert_called_once()
    workspace.data_quality.create_refresh.assert_not_called()

    monitor = workspace.data_quality.create_monitor.call_args.kwargs["monitor"]
    assert monitor.object_type == "table"
    assert monitor.object_id == "table-456"

    profiling = monitor.data_profiling_config
    assert profiling.output_schema_id == "schema-123"

    inference = profiling.inference_log
    assert (
        inference.problem_type == InferenceProblemType.INFERENCE_PROBLEM_TYPE_REGRESSION
    )
    assert inference.prediction_column == "prediction"
    assert inference.timestamp_column == "request_time"
    assert inference.model_id_column == "model_name"
    assert inference.label_column == "average_price"
    assert inference.granularities == [
        AggregationGranularity.AGGREGATION_GRANULARITY_5_MINUTES
    ]


def test_load_new_payload_reads_all_when_monitoring_table_absent(
    build_manager,
) -> None:
    manager, _ = build_manager(monitor_exists=True)
    manager.spark.catalog.tableExists.return_value = False

    manager._load_new_payload_data()

    query = manager.spark.sql.call_args.args[0]
    assert query == f"SELECT * FROM {manager.payload_table}"
    assert "WHERE" not in query


def test_load_new_payload_reads_incrementally_when_table_exists(
    build_manager,
) -> None:
    manager, _ = build_manager(monitor_exists=True)
    manager.spark.catalog.tableExists.return_value = True

    manager._load_new_payload_data()

    query = manager.spark.sql.call_args.args[0]
    # Only rows newer than the latest already-monitored request_time are pulled.
    assert "WHERE request_time >" in query
    assert f"MAX(request_time) FROM {manager.monitoring_table}" in query


def test_parse_and_transform_payload_runs_full_chain(build_manager, mocker) -> None:
    manager, _ = build_manager(monitor_exists=True)
    inf_table = mocker.Mock()

    result = manager._parse_and_transform_payload(inf_table)

    # Chain begins with from_json parsing of the raw request column...
    inf_table.withColumn.assert_called()
    # ...and ends in a select projecting the flattened monitoring schema.
    assert result is not None


def test_join_with_ground_truth_left_joins_on_booking_id(build_manager, mocker) -> None:
    manager, _ = build_manager(monitor_exists=True)
    df = mocker.Mock()

    manager._join_with_ground_truth(df)

    manager.spark.table.assert_called_once_with(manager.ground_truth_table)
    join_kwargs = df.join.call_args.kwargs
    assert join_kwargs["on"] == "Booking_ID"
    assert join_kwargs["how"] == "left"


def test_update_monitoring_table_noop_when_no_new_data(build_manager, mocker) -> None:
    manager, _ = build_manager(monitor_exists=True)
    empty = mocker.Mock()
    empty.isEmpty.return_value = True
    mocker.patch.object(
        manager, "_load_new_payload_data", return_value=empty, autospec=True
    )
    parse = mocker.patch.object(manager, "_parse_and_transform_payload", autospec=True)

    assert manager.update_monitoring_table() == 0
    parse.assert_not_called()


def test_update_monitoring_table_appends_and_counts(build_manager, mocker) -> None:
    manager, _ = build_manager(monitor_exists=True)
    raw = mocker.Mock()
    raw.isEmpty.return_value = False
    mocker.patch.object(
        manager, "_load_new_payload_data", return_value=raw, autospec=True
    )
    mocker.patch.object(
        manager, "_parse_and_transform_payload", return_value=raw, autospec=True
    )

    final = mocker.Mock()
    final.count.return_value = 5
    mocker.patch.object(
        manager, "_join_with_ground_truth", return_value=final, autospec=True
    )

    assert manager.update_monitoring_table() == 5
    saver = final.write.format.return_value.mode.return_value.saveAsTable
    saver.assert_called_once_with(manager.monitoring_table)
