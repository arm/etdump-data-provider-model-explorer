# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from model_explorer.extension_class_processor import ExtensionClassProcessor

_ETRECORD_EXTENSION_ID = "etrecord_adapter_model_explorer.main.ETRecordAdapter"

if not ExtensionClassProcessor.has_extension(_ETRECORD_EXTENSION_ID):
    raise ImportError(
        "ETDump requires `etrecord_adapter_model_explorer` to be loaded first."
    )

from .provider import ETDumpProfilingOverlayProvider  # noqa: F401
