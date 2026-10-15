# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

import unittest
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

_ASSERT = unittest.TestCase()

TESTS_ROOT = Path(__file__).resolve().parent
FIXTURES_ROOT = TESTS_ROOT / "fixtures" / "simple_convnet"


def test_provider_exposes_required_etdump_upload():
    provider = ETDumpProfilingOverlayProvider()
    metadata = provider.metadata
    if isinstance(metadata, list):
        raise AssertionError("not isinstance(metadata, list)")
    if metadata.filter is None:
        raise AssertionError("Expected an adapter filter")
    _ASSERT.assertEqual(metadata.filter.adapterIds, ["etrecord_adapter"])

    result = provider.get_config_editors(metadata.id)

    _ASSERT.assertEqual(result.error, "")

    config_editors = cast(
        list[config_editor.FileConfigEditor], result.configEditors
    )
    first_editor = config_editors[0]
    _ASSERT.assertEqual(first_editor.id, "etdump_path")
    _ASSERT.assertIs(first_editor.required, True)
    _ASSERT.assertEqual(first_editor.fileExts, ["etdp", "etdump"])


def test_provider_builds_runtime_overlay_for_graph():
    provider = ETDumpProfilingOverlayProvider()
    builder = ETRecordGraphBuilder(
        FIXTURES_ROOT / "input.etrecord", settings={}
    )
    graph_collections = builder.build()
    _ASSERT.assertTrue(graph_collections)

    graphs = graph_collections[0].graphs
    _ASSERT.assertTrue(graphs)

    graph = graphs[0]

    metadata = provider.metadata
    if isinstance(metadata, list):
        raise AssertionError("not isinstance(metadata, list)")

    result = provider.run(
        provider_id=metadata.id,
        model_path=str(FIXTURES_ROOT / "input.etrecord"),
        graph_id=graph.id,
        config_values={"etdump_path": str(FIXTURES_ROOT / "input.etdp")},
        graph=graph,
    )

    _ASSERT.assertEqual(result.error, "")

    overlays_result = result.result
    if overlays_result is None:
        raise AssertionError("Expected profiling overlays")
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
        _ASSERT.assertIn(expected_name, overlay_names)

    for overlay in overlays:
        _ASSERT.assertTrue(overlay.results)
        for node_result in overlay.results.values():
            if not isinstance(node_result.value, (int, float)):
                raise AssertionError("Expected a numeric node metric")
            _ASSERT.assertGreater(node_result.value, 0)
        if overlay.gradient is None:
            raise AssertionError("Expected an overlay gradient")
        _ASSERT.assertEqual(len(overlay.gradient), 2)
        _ASSERT.assertEqual(overlay.hideAggregatedStats, ["sum"])


def test_provider_requires_explicit_etdump_upload():
    provider = ETDumpProfilingOverlayProvider()
    builder = ETRecordGraphBuilder(
        FIXTURES_ROOT / "input.etrecord", settings={}
    )
    graph_collections = builder.build()
    _ASSERT.assertTrue(graph_collections)

    graphs = graph_collections[0].graphs
    _ASSERT.assertTrue(graphs)

    graph = graphs[0]

    metadata = provider.metadata
    if isinstance(metadata, list):
        raise AssertionError("not isinstance(metadata, list)")

    result = provider.run(
        provider_id=metadata.id,
        model_path=str(FIXTURES_ROOT / "input.etrecord"),
        graph_id=graph.id,
        config_values={},
        graph=graph,
    )

    _ASSERT.assertIsNone(result.result)
    _ASSERT.assertIn("ETDump file upload is required", result.error)


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
    if isinstance(metadata, list):
        raise AssertionError("not isinstance(metadata, list)")

    result = provider.run(
        provider_id=metadata.id,
        model_path=str(FIXTURES_ROOT / "input.etrecord"),
        graph_id=graph.id,
        config_values={"etdump_path": str(FIXTURES_ROOT / "input.etdp")},
        graph=graph,
    )

    _ASSERT.assertEqual(result.error, "")
    overlays = cast(list[node_data_builder.GraphNodeData], result.result)
    avg_overlay = next(
        overlay
        for overlay in overlays
        if overlay.name == "Runtime Total Avg (ms)"
    )
    _ASSERT.assertEqual(
        avg_overlay.results["aten_convolution_default_"].value, 3.0
    )

    count_overlay = next(
        overlay
        for overlay in overlays
        if overlay.name == "Runtime Event Count"
    )
    _ASSERT.assertEqual(
        count_overlay.results["aten_convolution_default_"].value, 2
    )
    _ASSERT.assertTrue(
        all(
            (
                not overlay.name.startswith("Runtime OPERATOR_CALL")
                for overlay in overlays
            )
        )
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
    if isinstance(metadata, list):
        raise AssertionError("not isinstance(metadata, list)")

    result = provider.run(
        provider_id=metadata.id,
        model_path=str(FIXTURES_ROOT / "input.etrecord"),
        graph_id=graph.id,
        config_values={"etdump_path": str(FIXTURES_ROOT / "input.etdp")},
        graph=graph,
    )

    _ASSERT.assertIsNone(result.result)
    _ASSERT.assertIn("No profiling events", result.error)


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
    if isinstance(metadata, list):
        raise AssertionError("not isinstance(metadata, list)")

    result = provider.run(
        provider_id=metadata.id,
        model_path=str(FIXTURES_ROOT / "input.etrecord"),
        graph_id=graph.id,
        config_values={"etdump_path": str(FIXTURES_ROOT / "input.etdp")},
        graph=graph,
    )

    _ASSERT.assertEqual(result.error, "")
    overlays = cast(list[node_data_builder.GraphNodeData], result.result)

    avg_overlay = next(
        overlay
        for overlay in overlays
        if overlay.name == "Runtime Total Avg (ms)"
    )
    _ASSERT.assertEqual(
        avg_overlay.results["aten_convolution_default_"].value, 4.0
    )
    _ASSERT.assertTrue(
        all(
            (
                overlay.name
                != "Runtime CoreMLBackend/aten_hardtanh_default_cast_fp16:clip Avg (ms)"
                for overlay in overlays
            )
        )
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
    if isinstance(metadata, list):
        raise AssertionError("not isinstance(metadata, list)")

    result = provider.run(
        provider_id=metadata.id,
        model_path=str(FIXTURES_ROOT / "input.etrecord"),
        graph_id=graph.id,
        config_values={"etdump_path": str(FIXTURES_ROOT / "input.etdp")},
        graph=graph,
    )

    _ASSERT.assertIsNone(result.result)
    _ASSERT.assertIn("No profiling events", result.error)


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
    if isinstance(metadata, list):
        raise AssertionError("not isinstance(metadata, list)")

    result = provider.run(
        provider_id=metadata.id,
        model_path=str(FIXTURES_ROOT / "input.etrecord"),
        graph_id=graph.id,
        config_values={"etdump_path": str(FIXTURES_ROOT / "input.etdp")},
        graph=graph,
    )

    _ASSERT.assertEqual(result.error, "")
    overlays = cast(list[node_data_builder.GraphNodeData], result.result)

    avg_overlay = next(
        overlay
        for overlay in overlays
        if overlay.name == "Runtime Total Avg (ms)"
    )
    _ASSERT.assertEqual(
        avg_overlay.results["aten_convolution_default_"].value, 3.0
    )

    count_overlay = next(
        overlay
        for overlay in overlays
        if overlay.name == "Runtime Event Count"
    )
    _ASSERT.assertEqual(
        count_overlay.results["aten_convolution_default_"].value, 2
    )
