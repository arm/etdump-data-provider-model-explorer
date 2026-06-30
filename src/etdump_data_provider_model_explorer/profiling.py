# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from executorch.devtools import Inspector
from executorch.devtools.inspector import Event
from executorch.devtools.inspector._inspector_utils import (
    RESERVED_FRAMEWORK_EVENT_NAMES,
)

from .constants import PREFERRED_RUNTIME_EVENT
from .util import ensure_flatc_executable


@dataclass(frozen=True)
class RuntimeEventSummary:
    event_block_name: str
    name: str
    avg_ms: float
    p50_ms: float
    min_ms: float
    max_ms: float
    raw_ms: list[float]
    delegate_backend_name: str | None


@dataclass
class DebugHandleRuntimeOverlay:
    debug_handle: int
    summaries: list[RuntimeEventSummary]

    @property
    def preferred_summary(self) -> RuntimeEventSummary:
        for summary in self.summaries:
            if (
                summary.delegate_backend_name is not None
                and summary.name != "DELEGATE_CALL"
            ):
                return summary

        for summary in self.summaries:
            if summary.name == PREFERRED_RUNTIME_EVENT:
                return summary

        for summary in self.summaries:
            if summary.name != "DELEGATE_CALL":
                return summary

        return self.summaries[0]


def load_runtime_overlay(
    etrecord_path: Path,
    etdump_path: Path,
    debug_buffer_path: Path | None = None,
) -> dict[int, DebugHandleRuntimeOverlay]:
    ensure_flatc_executable()

    inspector = Inspector(
        etdump_path=str(etdump_path),
        etrecord=str(etrecord_path),
        debug_buffer_path=(
            str(debug_buffer_path) if debug_buffer_path is not None else None
        ),
    )
    return build_runtime_overlay(inspector.event_blocks)


def build_runtime_overlay(
    event_blocks: Iterable,
) -> dict[int, DebugHandleRuntimeOverlay]:
    overlay_by_handle: dict[int, list[RuntimeEventSummary]] = {}

    for event_block in event_blocks:
        for event in event_block.events:
            summary = summarize_event(event_block.name, event)
            if summary is None:
                continue

            for handle in normalize_debug_handles(event.debug_handles):
                overlay_by_handle.setdefault(handle, []).append(summary)

    return {
        handle: DebugHandleRuntimeOverlay(
            debug_handle=handle,
            summaries=summaries,
        )
        for handle, summaries in overlay_by_handle.items()
    }


def normalize_debug_handles(debug_handles: int | Sequence[int]) -> list[int]:
    if isinstance(debug_handles, int):
        return [debug_handles]
    return [int(handle) for handle in debug_handles]


def summarize_event(
    event_block_name: str,
    event: Event,
) -> RuntimeEventSummary | None:
    if event.perf_data is None or event.debug_handles is None:
        return None
    if event.name in RESERVED_FRAMEWORK_EVENT_NAMES:
        return None

    return RuntimeEventSummary(
        event_block_name=event_block_name,
        name=event.name,
        avg_ms=float(event.perf_data.avg),
        p50_ms=float(event.perf_data.p50),
        min_ms=float(event.perf_data.min),
        max_ms=float(event.perf_data.max),
        raw_ms=[float(sample) for sample in event.perf_data.raw],
        delegate_backend_name=event.delegate_backend_name,
    )
