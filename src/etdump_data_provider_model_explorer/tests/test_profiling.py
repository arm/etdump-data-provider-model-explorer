# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from etdump_data_provider_model_explorer.profiling import build_runtime_overlay


def test_build_runtime_overlay_filters_events_and_normalizes_handles():
    perf_data = SimpleNamespace(
        avg=1.5,
        p50=1.5,
        min=1.0,
        max=2.0,
        raw=[1.0, 2.0],
    )
    event_block = SimpleNamespace(
        name="Execute",
        events=[
            SimpleNamespace(
                name="missing_perf_data",
                perf_data=None,
                debug_handles=[10],
                delegate_backend_name=None,
            ),
            SimpleNamespace(
                name="missing_debug_handles",
                perf_data=perf_data,
                debug_handles=None,
                delegate_backend_name=None,
            ),
            SimpleNamespace(
                name="Method::execute",
                perf_data=perf_data,
                debug_handles=[44],
                delegate_backend_name=None,
            ),
            SimpleNamespace(
                name="aten_relu_default",
                perf_data=perf_data,
                debug_handles=43,
                delegate_backend_name=None,
            ),
            SimpleNamespace(
                name="aten_convolution_default",
                perf_data=perf_data,
                debug_handles=[42],
                delegate_backend_name=None,
            ),
        ],
    )

    # Only aten_relu_default and aten_convolution_default will be kept
    runtime_overlay = build_runtime_overlay([event_block])

    assert set(runtime_overlay) == {42, 43}
    overlay = runtime_overlay[42]
    assert overlay.debug_handle == 42
    assert len(overlay.summaries) == 1

    summary = overlay.summaries[0]
    assert summary.event_block_name == "Execute"
    assert summary.name == "aten_convolution_default"
    assert summary.avg_ms == 1.5
    assert summary.p50_ms == 1.5
    assert summary.min_ms == 1.0
    assert summary.max_ms == 2.0
    assert summary.raw_ms == [1.0, 2.0]

    single_handle_overlay = runtime_overlay[43]
    assert single_handle_overlay.debug_handle == 43
    assert [summary.name for summary in single_handle_overlay.summaries] == [
        "aten_relu_default"
    ]
