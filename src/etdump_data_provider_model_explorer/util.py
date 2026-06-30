# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from pathlib import Path


def ensure_flatc_executable() -> None:
    if os.getenv("FLATC_EXECUTABLE"):
        return

    if shutil.which("flatc"):
        return

    from executorch.exir._serialize import _flatbuffer

    flatbuffer_path = Path(_flatbuffer.__file__).resolve()
    candidates = [
        flatbuffer_path.parent / "flatbuffers-flatc",
        flatbuffer_path.parents[2] / "data/bin/flatc",
        flatbuffer_path.parents[3] / "data/bin/flatc",
    ]

    for candidate in candidates:
        if candidate.is_file():
            os.environ["FLATC_EXECUTABLE"] = str(candidate)
            return
