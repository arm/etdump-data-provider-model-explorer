#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Create a synthetic delegated ETDump and a patched ETRecord for Inspector testing.

This script starts from a real ETRecord file, keeps its graph structure, and writes:
- a patched ETRecord copy with a synthetic delegate map
- a synthetic ETDump binary that references that delegate map

The generated pair is intended for UI / Inspector prototyping when a delegated
backend does not yet emit per-op profiling events with debug handles.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from executorch.devtools.etdump.schema_flatcc import (
    ETDumpFlatCC,
    Event,
    ProfileEvent,
    RunData,
)
from executorch.devtools.etdump.serialize import serialize_to_etdump_flatcc
from executorch.devtools.etrecord import ETRecord, parse_etrecord
from executorch.devtools.inspector._inspector_utils import (
    EDGE_DIALECT_GRAPH_KEY,
    gen_graphs_from_etrecord,
)

FORWARD = "forward"
DEFAULT_DELEGATE_INSTRUCTION_ID = 9000
DEFAULT_DELEGATE_CALL_TIME_MS = 2.08
DEFAULT_NODE_TIME_MS = 0.12
DEFAULT_BACKEND_NAME = "MockDelegateBackend"


def _iter_operator_nodes(graph: Any) -> Iterable[Any]:
    elements = list(getattr(graph, "elements", []) or [])
    for element in elements:
        if hasattr(element, "graph_name") and hasattr(element, "elements"):
            yield from _iter_operator_nodes(element)
        elif hasattr(element, "op"):
            yield element


def _safe_name(value: Any) -> str:
    return str(value) if value is not None else ""


def _node_label(node: Any) -> str:
    metadata = getattr(node, "metadata", None) or {}
    module_stack = metadata.get("nn_module_stack") or {}
    if module_stack:
        entries = list(module_stack.values())
        if entries:
            _path, module_type = entries[-1]
            if module_type:
                return str(module_type).rsplit(".", 1)[-1]
    op = getattr(node, "op", None)
    if op:
        op_str = str(op)
        pieces = op_str.split(".")
        if len(pieces) >= 2:
            return pieces[1]
        return op_str
    return _safe_name(getattr(node, "name", "node"))


def _extract_nodes_by_handle(etrecord: ETRecord) -> list[dict[str, Any]]:
    graphs = gen_graphs_from_etrecord(
        etrecord=etrecord, enable_module_hierarchy=True
    )
    edge_graph = graphs[EDGE_DIALECT_GRAPH_KEY]
    records: list[dict[str, Any]] = []
    for node in _iter_operator_nodes(edge_graph):
        metadata = getattr(node, "metadata", None) or {}
        handle = metadata.get("debug_handle")
        if handle is None:
            continue
        try:
            handle_int = int(handle)
        except (TypeError, ValueError):
            continue
        records.append(
            {
                "id": _safe_name(getattr(node, "name", f"node_{handle_int}")),
                "label": _node_label(node),
                "op": _safe_name(getattr(node, "op", "")),
                "debug_handle": handle_int,
            }
        )
    return records


def _ensure_maps(
    etrecord: ETRecord,
) -> tuple[
    dict[str, dict[int, int | list[int]]],
    dict[str, dict[int, dict[str, Any]]],
    dict[str, dict[int, int | list[int]]],
]:
    debug_handle_map = cast(
        dict[str, dict[int, int | list[int]]],
        etrecord._debug_handle_map or {},
    )
    delegate_map = cast(
        dict[str, dict[int, dict[str, Any]]],
        etrecord._delegate_map or {},
    )
    instruction_id_to_num_outs_map = cast(
        dict[str, dict[int, int | list[int]]],
        etrecord._instruction_id_to_num_outs_map or {},
    )

    debug_handle_map.setdefault(FORWARD, {})
    delegate_map.setdefault(FORWARD, {})
    instruction_id_to_num_outs_map.setdefault(FORWARD, {})

    etrecord._debug_handle_map = cast(Any, debug_handle_map)
    etrecord._delegate_map = cast(Any, delegate_map)
    etrecord._instruction_id_to_num_outs_map = cast(
        Any, instruction_id_to_num_outs_map
    )

    return debug_handle_map, delegate_map, instruction_id_to_num_outs_map


def _make_delegate_debug_ids(nodes: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    debug_ids: list[str] = []
    for node in nodes:
        base = (
            re.sub(r"[^a-z0-9]+", "_", node["label"].lower()).strip("_")
            or "op"
        )
        counts[base] = counts.get(base, 0) + 1
        debug_ids.append(f"{base}_{counts[base]}")
    return debug_ids


def _patch_etrecord_for_mock_delegate(
    etrecord: ETRecord,
    nodes: list[dict[str, Any]],
    delegate_instruction_id: int,
    backend_name: str,
) -> tuple[ETRecord, list[str]]:
    debug_handle_map, delegate_map, instruction_id_to_num_outs_map = (
        _ensure_maps(etrecord)
    )

    all_handles = [node["debug_handle"] for node in nodes]
    debug_handle_map[FORWARD][delegate_instruction_id] = all_handles
    instruction_id_to_num_outs_map[FORWARD][delegate_instruction_id] = 1

    delegate_debug_ids = _make_delegate_debug_ids(nodes)
    delegate_mapping = {
        delegate_debug_id: [node["debug_handle"]]
        for delegate_debug_id, node in zip(
            delegate_debug_ids, nodes, strict=True
        )
    }
    delegate_map[FORWARD][delegate_instruction_id] = {
        "name": backend_name,
        "delegate_map": delegate_mapping,
    }
    return etrecord, delegate_debug_ids


def _ms_to_ns(value_ms: float) -> int:
    return int(value_ms * 1_000_000)


def _profile_event(
    *,
    name: str,
    instruction_id: int,
    start_ms: float,
    end_ms: float,
    delegate_debug_id: str | int | None = None,
) -> Event:
    return Event(
        profile_event=ProfileEvent(
            name=name,
            chain_index=0,
            instruction_id=instruction_id,
            delegate_debug_id_int=delegate_debug_id
            if isinstance(delegate_debug_id, int)
            else -1,
            delegate_debug_id_str=delegate_debug_id
            if isinstance(delegate_debug_id, str)
            else "",
            delegate_debug_metadata=bytes(),
            start_time=_ms_to_ns(start_ms),
            end_time=_ms_to_ns(end_ms),
        ),
        allocation_event=None,
        debug_event=None,
    )


def _build_mock_delegate_etdump(
    nodes: list[dict[str, Any]],
    delegate_instruction_id: int,
    backend_name: str,
    delegate_debug_ids: list[str],
    delegate_call_time_ms: float,
    node_time_ms: float,
) -> ETDumpFlatCC:
    default_events = [
        _profile_event(
            name="Method::init",
            instruction_id=0,
            start_ms=0.0,
            end_ms=4.25,
        ),
        _profile_event(
            name="Program::load_method",
            instruction_id=0,
            start_ms=4.25,
            end_ms=6.05,
        ),
    ]

    execute_events: list[Event] = [
        _profile_event(
            name="DELEGATE_CALL",
            instruction_id=delegate_instruction_id,
            start_ms=0.0,
            end_ms=delegate_call_time_ms,
        )
    ]

    cursor_ms = 0.1
    for node, delegate_debug_id in zip(nodes, delegate_debug_ids, strict=True):
        # Use a stable delegate debug id that the patched ETRecord maps back to the node handle.
        start_ms = cursor_ms
        duration_ms = _mock_node_duration_ms(
            node=node,
            delegate_debug_id=delegate_debug_id,
            base_ms=node_time_ms,
        )
        end_ms = start_ms + duration_ms
        execute_events.append(
            _profile_event(
                name=f"{backend_name}::{node['label']}",
                instruction_id=delegate_instruction_id,
                start_ms=start_ms,
                end_ms=end_ms,
                delegate_debug_id=delegate_debug_id,
            )
        )
        cursor_ms = end_ms + 0.02

    execute_events.append(
        _profile_event(
            name="Method::execute",
            instruction_id=9999,
            start_ms=0.0,
            end_ms=max(delegate_call_time_ms, cursor_ms),
        )
    )

    return ETDumpFlatCC(
        version=0,
        run_data=[
            RunData(
                name="Default",
                bundled_input_index=-1,
                allocators=[],
                events=default_events,
            ),
            RunData(
                name="Execute",
                bundled_input_index=-1,
                allocators=[],
                events=execute_events,
            ),
        ],
    )


def _mock_node_duration_ms(
    *,
    node: dict[str, Any],
    delegate_debug_id: str,
    base_ms: float,
) -> float:
    digest = hashlib.sha256(
        f"{delegate_debug_id}:{node['debug_handle']}:{node['label']}".encode(
            "utf-8"
        )
    ).digest()
    # Stable jitter in the range [0.75, 1.45).
    ratio = int.from_bytes(digest[:4], byteorder="little") / 2**32
    multiplier = 0.75 + (ratio * 0.70)
    return round(base_ms * multiplier, 4)


def _write_etdump(path: Path, etdump: ETDumpFlatCC) -> None:
    raw = serialize_to_etdump_flatcc(etdump)
    size_prefixed = len(raw).to_bytes(4, byteorder="little") + raw
    path.write_bytes(size_prefixed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a synthetic delegated ETDump and a patched ETRecord copy that "
            "Inspector can load for prototyping per-op delegate overlays."
        )
    )
    parser.add_argument("etrecord", type=Path, help="Input real ETRecord file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write mock artifacts into (defaults next to input ETRecord)",
    )
    parser.add_argument(
        "--delegate-instruction-id",
        type=int,
        default=DEFAULT_DELEGATE_INSTRUCTION_ID,
        help="Synthetic delegate instruction id used in the patched ETRecord and ETDump",
    )
    parser.add_argument(
        "--backend-name",
        default=DEFAULT_BACKEND_NAME,
        help="Synthetic delegate backend name to store in the patched ETRecord",
    )
    parser.add_argument(
        "--delegate-call-time-ms",
        type=float,
        default=DEFAULT_DELEGATE_CALL_TIME_MS,
        help="Total synthetic delegate call duration in milliseconds",
    )
    parser.add_argument(
        "--node-time-ms",
        type=float,
        default=DEFAULT_NODE_TIME_MS,
        help="Base synthetic per-node duration in milliseconds",
    )
    args = parser.parse_args()

    input_etrecord = args.etrecord.resolve()
    if not input_etrecord.is_file():
        raise FileNotFoundError(f"ETRecord file not found: {input_etrecord}")

    output_dir = (
        args.output_dir or input_etrecord.parent / "mock_delegate_artifacts"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    etrecord = parse_etrecord(str(input_etrecord))
    nodes = _extract_nodes_by_handle(etrecord)
    if not nodes:
        raise RuntimeError(
            "No operator nodes with debug handles were found in the ETRecord."
        )

    patched_etrecord, delegate_debug_ids = _patch_etrecord_for_mock_delegate(
        etrecord,
        nodes,
        delegate_instruction_id=args.delegate_instruction_id,
        backend_name=args.backend_name,
    )

    etdump = _build_mock_delegate_etdump(
        nodes,
        delegate_instruction_id=args.delegate_instruction_id,
        backend_name=args.backend_name,
        delegate_debug_ids=delegate_debug_ids,
        delegate_call_time_ms=args.delegate_call_time_ms,
        node_time_ms=args.node_time_ms,
    )

    patched_etrecord_path = output_dir / f"{input_etrecord.stem}.mock.etrecord"
    etdump_path = output_dir / f"{input_etrecord.stem}.mock.etdump"

    patched_etrecord.save(patched_etrecord_path)
    _write_etdump(etdump_path, etdump)

    print(f"Wrote patched ETRecord: {patched_etrecord_path}")
    print(f"Wrote synthetic ETDump: {etdump_path}")
    print(
        f"Synthesized {len(nodes)} delegate-op events for backend {args.backend_name}."
    )


if __name__ == "__main__":
    main()
