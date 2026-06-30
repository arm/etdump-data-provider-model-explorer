# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import cast

from etrecord_adapter_model_explorer.builder.builder import (
    ETRecordGraphBuilder,
)
from model_explorer import config_editor, node_data_builder

from etdump_data_provider_model_explorer import (
    provider as provider_module,
)
from etdump_data_provider_model_explorer.profiling import (
    DebugHandleRuntimeOverlay,
    RuntimeEventSummary,
)
from etdump_data_provider_model_explorer.provider import (
    ETDumpProfilingOverlayProvider,
)

TESTS_ROOT = Path(__file__).resolve().parent
FIXTURES_ROOT = TESTS_ROOT / "fixtures" / "simple_convnet"


def test_provider_exposes_required_etdump_upload():
    provider = ETDumpProfilingOverlayProvider()
    metadata = provider.metadata
    assert not isinstance(metadata, list)
    assert metadata.filter is not None
    assert metadata.filter.adapterIds == ["etrecord_adapter"]

    result = provider.get_config_editors(metadata.id)

    assert result.error == ""

    config_editors = cast(
        list[config_editor.FileConfigEditor], result.configEditors
    )
    first_editor = config_editors[0]
    assert first_editor.id == "etdump_path"
    assert first_editor.required is True
    assert first_editor.fileExts == ["etdp", "etdump"]


def test_provider_builds_runtime_overlay_for_graph():
    provider = ETDumpProfilingOverlayProvider()
    builder = ETRecordGraphBuilder(
        FIXTURES_ROOT / "input.etrecord", settings={}
    )
    graph_collections = builder.build()
    assert graph_collections

    graphs = graph_collections[0].graphs
    assert graphs

    graph = graphs[0]

    metadata = provider.metadata
    assert not isinstance(metadata, list)

    result = provider.run(
        provider_id=metadata.id,
        model_path=str(FIXTURES_ROOT / "input.etrecord"),
        graph_id=graph.id,
        config_values={"etdump_path": str(FIXTURES_ROOT / "input.etdp")},
        graph=graph,
    )

    assert result.error == ""

    overlays_result = result.result
    assert overlays_result is not None
    overlays_result = (
        overlays_result
        if isinstance(overlays_result, list)
        else [overlays_result]
    )
    overlays = cast(list[node_data_builder.GraphNodeData], overlays_result)

    overlay_names = [overlay.name for overlay in overlays]
    for expected_name in [
        "Runtime Total Avg (ms)",
        "Runtime Total Max (ms)",
        "Runtime Total Min (ms)",
        "Runtime Total P50 (ms)",
        "Runtime Event Count",
    ]:
        assert expected_name in overlay_names

    for overlay in overlays:
        assert overlay.results
        for node_result in overlay.results.values():
            assert isinstance(node_result.value, (int, float))
            assert node_result.value > 0
        assert overlay.gradient is not None
        assert len(overlay.gradient) == 2
        assert overlay.hideAggregatedStats == ["sum"]


def test_provider_requires_explicit_etdump_upload():
    provider = ETDumpProfilingOverlayProvider()
    builder = ETRecordGraphBuilder(
        FIXTURES_ROOT / "input.etrecord", settings={}
    )
    graph_collections = builder.build()
    assert graph_collections

    graphs = graph_collections[0].graphs
    assert graphs

    graph = graphs[0]

    metadata = provider.metadata
    assert not isinstance(metadata, list)

    result = provider.run(
        provider_id=metadata.id,
        model_path=str(FIXTURES_ROOT / "input.etrecord"),
        graph_id=graph.id,
        config_values={},
        graph=graph,
    )

    assert result.result is None
    assert "ETDump file upload is required" in result.error


def test_provider_sums_concrete_events_and_ignores_operator_call(
    monkeypatch,
):
    provider = ETDumpProfilingOverlayProvider()
    builder = ETRecordGraphBuilder(
        FIXTURES_ROOT / "input.etrecord", settings={}
    )
    graph = builder.build()[0].graphs[0]

    monkeypatch.setattr(
        provider_module,
        "load_runtime_overlay",
        lambda **_: {
            1: DebugHandleRuntimeOverlay(
                debug_handle=1,
                summaries=[
                    RuntimeEventSummary(
                        event_block_name="Execute",
                        name="OPERATOR_CALL",
                        avg_ms=3.0,
                        p50_ms=3.0,
                        min_ms=3.0,
                        max_ms=3.0,
                        raw_ms=[3.0],
                        delegate_backend_name=None,
                    ),
                    RuntimeEventSummary(
                        event_block_name="Execute",
                        name="OTHER_EVENT",
                        avg_ms=1.0,
                        p50_ms=1.0,
                        min_ms=1.0,
                        max_ms=1.0,
                        raw_ms=[1.0],
                        delegate_backend_name=None,
                    ),
                    RuntimeEventSummary(
                        event_block_name="Execute",
                        name="ANOTHER_EVENT",
                        avg_ms=2.0,
                        p50_ms=2.0,
                        min_ms=2.0,
                        max_ms=2.0,
                        raw_ms=[2.0],
                        delegate_backend_name=None,
                    ),
                ],
            )
        },
    )

    metadata = provider.metadata
    assert not isinstance(metadata, list)

    result = provider.run(
        provider_id=metadata.id,
        model_path=str(FIXTURES_ROOT / "input.etrecord"),
        graph_id=graph.id,
        config_values={"etdump_path": str(FIXTURES_ROOT / "input.etdp")},
        graph=graph,
    )

    assert result.error == ""
    overlays = cast(list[node_data_builder.GraphNodeData], result.result)
    avg_overlay = next(
        overlay
        for overlay in overlays
        if overlay.name == "Runtime Total Avg (ms)"
    )
    assert avg_overlay.results["aten_convolution_default_"].value == 3.0

    count_overlay = next(
        overlay
        for overlay in overlays
        if overlay.name == "Runtime Event Count"
    )
    assert count_overlay.results["aten_convolution_default_"].value == 2
    assert all(
        not overlay.name.startswith("Runtime OPERATOR_CALL")
        for overlay in overlays
    )


def test_provider_ignores_delegate_call_when_no_concrete_events(
    monkeypatch,
):
    provider = ETDumpProfilingOverlayProvider()
    builder = ETRecordGraphBuilder(
        FIXTURES_ROOT / "input.etrecord", settings={}
    )
    graph = builder.build()[0].graphs[0]

    monkeypatch.setattr(
        provider_module,
        "load_runtime_overlay",
        lambda **_: {
            1: DebugHandleRuntimeOverlay(
                debug_handle=1,
                summaries=[
                    RuntimeEventSummary(
                        event_block_name="Execute",
                        name="DELEGATE_CALL",
                        avg_ms=3.0,
                        p50_ms=3.0,
                        min_ms=3.0,
                        max_ms=3.0,
                        raw_ms=[3.0],
                        delegate_backend_name="MockBackend",
                    )
                ],
            )
        },
    )

    metadata = provider.metadata
    assert not isinstance(metadata, list)

    result = provider.run(
        provider_id=metadata.id,
        model_path=str(FIXTURES_ROOT / "input.etrecord"),
        graph_id=graph.id,
        config_values={"etdump_path": str(FIXTURES_ROOT / "input.etdp")},
        graph=graph,
    )

    assert result.result is None
    assert "No profiling events" in result.error


def test_provider_uses_delegate_event_as_concrete_event(
    monkeypatch,
):
    provider = ETDumpProfilingOverlayProvider()
    builder = ETRecordGraphBuilder(
        FIXTURES_ROOT / "input.etrecord", settings={}
    )
    graph = builder.build()[0].graphs[0]

    monkeypatch.setattr(
        provider_module,
        "load_runtime_overlay",
        lambda **_: {
            1: DebugHandleRuntimeOverlay(
                debug_handle=1,
                summaries=[
                    RuntimeEventSummary(
                        event_block_name="Execute",
                        name="aten_hardtanh_default_cast_fp16:clip",
                        avg_ms=4.0,
                        p50_ms=4.0,
                        min_ms=4.0,
                        max_ms=4.0,
                        raw_ms=[4.0],
                        delegate_backend_name="CoreMLBackend",
                    )
                ],
            )
        },
    )

    metadata = provider.metadata
    assert not isinstance(metadata, list)

    result = provider.run(
        provider_id=metadata.id,
        model_path=str(FIXTURES_ROOT / "input.etrecord"),
        graph_id=graph.id,
        config_values={"etdump_path": str(FIXTURES_ROOT / "input.etdp")},
        graph=graph,
    )

    assert result.error == ""
    overlays = cast(list[node_data_builder.GraphNodeData], result.result)

    avg_overlay = next(
        overlay
        for overlay in overlays
        if overlay.name == "Runtime Total Avg (ms)"
    )
    assert avg_overlay.results["aten_convolution_default_"].value == 4.0
    assert all(
        overlay.name
        != (
            "Runtime CoreMLBackend/"
            "aten_hardtanh_default_cast_fp16:clip Avg (ms)"
        )
        for overlay in overlays
    )


def test_provider_ignores_duplicate_operator_calls(monkeypatch):
    provider = ETDumpProfilingOverlayProvider()
    builder = ETRecordGraphBuilder(
        FIXTURES_ROOT / "input.etrecord", settings={}
    )
    graph = builder.build()[0].graphs[0]

    monkeypatch.setattr(
        provider_module,
        "load_runtime_overlay",
        lambda **_: {
            1: DebugHandleRuntimeOverlay(
                debug_handle=1,
                summaries=[
                    RuntimeEventSummary(
                        event_block_name="Execute",
                        name="OPERATOR_CALL",
                        avg_ms=1.0,
                        p50_ms=1.0,
                        min_ms=1.0,
                        max_ms=1.0,
                        raw_ms=[1.0],
                        delegate_backend_name=None,
                    ),
                    RuntimeEventSummary(
                        event_block_name="Execute",
                        name="OPERATOR_CALL",
                        avg_ms=2.0,
                        p50_ms=2.0,
                        min_ms=2.0,
                        max_ms=2.0,
                        raw_ms=[2.0],
                        delegate_backend_name=None,
                    ),
                ],
            )
        },
    )

    metadata = provider.metadata
    assert not isinstance(metadata, list)

    result = provider.run(
        provider_id=metadata.id,
        model_path=str(FIXTURES_ROOT / "input.etrecord"),
        graph_id=graph.id,
        config_values={"etdump_path": str(FIXTURES_ROOT / "input.etdp")},
        graph=graph,
    )

    assert result.result is None
    assert "No profiling events" in result.error


def test_provider_sums_duplicate_concrete_events(monkeypatch):
    provider = ETDumpProfilingOverlayProvider()
    builder = ETRecordGraphBuilder(
        FIXTURES_ROOT / "input.etrecord", settings={}
    )
    graph = builder.build()[0].graphs[0]

    monkeypatch.setattr(
        provider_module,
        "load_runtime_overlay",
        lambda **_: {
            1: DebugHandleRuntimeOverlay(
                debug_handle=1,
                summaries=[
                    RuntimeEventSummary(
                        event_block_name="Execute",
                        name="OPERATOR_CALL",
                        avg_ms=3.0,
                        p50_ms=3.0,
                        min_ms=3.0,
                        max_ms=3.0,
                        raw_ms=[3.0],
                        delegate_backend_name=None,
                    ),
                    RuntimeEventSummary(
                        event_block_name="Execute",
                        name="OTHER_EVENT",
                        avg_ms=1.0,
                        p50_ms=1.0,
                        min_ms=1.0,
                        max_ms=1.0,
                        raw_ms=[1.0],
                        delegate_backend_name=None,
                    ),
                    RuntimeEventSummary(
                        event_block_name="Execute",
                        name="OTHER_EVENT",
                        avg_ms=2.0,
                        p50_ms=2.0,
                        min_ms=2.0,
                        max_ms=2.0,
                        raw_ms=[2.0],
                        delegate_backend_name=None,
                    ),
                ],
            )
        },
    )

    metadata = provider.metadata
    assert not isinstance(metadata, list)

    result = provider.run(
        provider_id=metadata.id,
        model_path=str(FIXTURES_ROOT / "input.etrecord"),
        graph_id=graph.id,
        config_values={"etdump_path": str(FIXTURES_ROOT / "input.etdp")},
        graph=graph,
    )

    assert result.error == ""
    overlays = cast(list[node_data_builder.GraphNodeData], result.result)

    avg_overlay = next(
        overlay
        for overlay in overlays
        if overlay.name == "Runtime Total Avg (ms)"
    )
    assert avg_overlay.results["aten_convolution_default_"].value == 3.0

    count_overlay = next(
        overlay
        for overlay in overlays
        if overlay.name == "Runtime Event Count"
    )
    assert count_overlay.results["aten_convolution_default_"].value == 2
