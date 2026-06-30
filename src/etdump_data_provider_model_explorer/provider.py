# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from etrecord_adapter_model_explorer.constants import ADAPTER_ID
from model_explorer import (
    GetConfigEditorsResult,
    NodeDataProvider,
    NodeDataProviderFilter,
    NodeDataProviderMetadata,
    NodeDataProviderResult,
    config_editor,
    graph_builder,
    node_data_builder,
)

from .constants import (
    HIGH_LATENCY_COLOR,
    LOW_LATENCY_COLOR,
    PREFERRED_RUNTIME_EVENT,
)
from .profiling import RuntimeEventSummary, load_runtime_overlay

_RUNTIME_METRICS = (
    ("Runtime Total Avg (ms)", "avg_ms"),
    ("Runtime Total P50 (ms)", "p50_ms"),
    ("Runtime Total Min (ms)", "min_ms"),
    ("Runtime Total Max (ms)", "max_ms"),
)

_WRAPPER_EVENT_NAMES = {
    PREFERRED_RUNTIME_EVENT,
    "DELEGATE_CALL",
    "Method::execute",
    "Method::init",
    "Program::load_method",
}


class ETDumpProfilingOverlayProvider(NodeDataProvider):
    metadata = NodeDataProviderMetadata(
        id="etdump_data_provider",
        name="ETDump Profiling Overlay",
        description="Upload an ETDump file and overlay runtime latency on ETRecord nodes.",
        icon="speed",
        filter=NodeDataProviderFilter(
            modelFileExts=["etrecord"],
            adapterIds=[ADAPTER_ID],
        ),
    )

    def get_config_editors(self, provider_id: str) -> GetConfigEditorsResult:
        del provider_id
        return GetConfigEditorsResult(
            configEditors=[
                config_editor.FileConfigEditor(
                    id="etdump_path",
                    label="ETDump file",
                    description="Upload the .etdp file to overlay runtime latency data onto the current ETRecord graph.",
                    help="Only the file uploaded here is used. The provider does not search for sibling ETDump files on disk.",
                    fileExts=["etdp", "etdump"],
                    required=True,
                ),
            ]
        )

    def run(
        self,
        provider_id: str,
        model_path: str,
        graph_id: str,
        config_values: dict,
        graph: graph_builder.Graph | None,
    ) -> NodeDataProviderResult:
        del provider_id
        del graph_id

        etrecord_path = Path(model_path)
        if etrecord_path.suffix.lower() != ".etrecord":
            return NodeDataProviderResult(
                error="The ETDump profiling overlay only supports .etrecord models."
            )
        try:
            etrecord_path = _validated_path(
                model_path,
                error="An ETRecord file upload is required.",
            )
            etdump_path = _validated_path(
                str(config_values.get("etdump_path", "")),
                error="An ETDump file upload is required.",
            )
        except ValueError as err:
            return NodeDataProviderResult(error=str(err))

        if graph is None:
            return NodeDataProviderResult(
                error="Model Explorer did not provide the current graph, so runtime data cannot be mapped onto nodes."
            )

        try:
            runtime_overlay = load_runtime_overlay(
                etrecord_path=etrecord_path,
                etdump_path=etdump_path,
            )
        except Exception as err:
            return NodeDataProviderResult(
                error=f"Failed to load profiling data from ETDump: {err}"
            )

        overlay_results: dict[
            str, dict[str, node_data_builder.NodeDataResult]
        ] = {}
        for node in graph.nodes:
            debug_handle = _node_debug_handle(node)
            if debug_handle is None:
                continue
            overlay = runtime_overlay.get(debug_handle)
            if overlay is None:
                continue

            concrete_summaries = _concrete_summaries(overlay.summaries)
            if not concrete_summaries:
                continue

            for overlay_name, summary_attr in _RUNTIME_METRICS:
                overlay_results.setdefault(overlay_name, {})[node.id] = (
                    node_data_builder.NodeDataResult(
                        value=sum(
                            getattr(summary, summary_attr)
                            for summary in concrete_summaries
                        )
                    )
                )

            overlay_results.setdefault("Runtime Event Count", {})[node.id] = (
                node_data_builder.NodeDataResult(value=len(concrete_summaries))
            )

        overlays_by_name = {
            name: results
            for name, results in overlay_results.items()
            if results
        }

        if not overlays_by_name:
            return NodeDataProviderResult(
                error=(
                    "No profiling events from the uploaded ETDump matched any "
                    "debug handles in the current ETRecord graph."
                )
            )

        return NodeDataProviderResult(
            result=[
                node_data_builder.GraphNodeData(
                    name=name,
                    results=results,
                    gradient=[
                        node_data_builder.GradientItem(
                            stop=0,
                            bgColor=LOW_LATENCY_COLOR,
                        ),
                        node_data_builder.GradientItem(
                            stop=1,
                            bgColor=HIGH_LATENCY_COLOR,
                        ),
                    ],
                    hideAggregatedStats=["sum"],
                )
                for name, results in sorted(overlays_by_name.items())
            ]
        )


def _node_debug_handle(node: graph_builder.GraphNode) -> int | None:
    for attr in node.attrs:
        if attr.key != "debug_handle":
            continue
        try:
            return int(str(attr.value))
        except (TypeError, ValueError):
            return None
    return None


def _concrete_summaries(
    summaries: list[RuntimeEventSummary],
) -> list[RuntimeEventSummary]:
    return [
        summary
        for summary in summaries
        if summary.event_block_name == "Execute"
        and summary.name not in _WRAPPER_EVENT_NAMES
    ]


def _validated_path(
    path_str: str,
    error: str,
) -> Path:
    path_str = path_str.strip()
    if not path_str:
        raise ValueError(error)

    path = Path(path_str)
    if not path.exists():
        raise ValueError(error)

    return path
