# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from unittest.mock import Mock

import pytest
from executorch.exir._serialize import _flatbuffer

from etdump_data_provider_model_explorer.util import ensure_flatc_executable


def test_flatc_executable_from_os_env_passed_through(monkeypatch):
    monkeypatch.setenv("FLATC_EXECUTABLE", "provided_flatc_executable")

    ensure_flatc_executable()

    assert os.getenv("FLATC_EXECUTABLE") == "provided_flatc_executable"


def test_flatc_executable_on_path_does_not_set_env(monkeypatch):
    monkeypatch.delenv("FLATC_EXECUTABLE", raising=False)
    which = Mock(return_value="provided_flatc_executable")
    monkeypatch.setattr(shutil, "which", which)

    ensure_flatc_executable()

    which.assert_called_once()
    assert os.getenv("FLATC_EXECUTABLE") is None


@pytest.mark.parametrize("candidate_index", [0, 1, 2])
def test_flatc_executable_path_resolves(
    candidate_index, tmp_path, monkeypatch
):
    monkeypatch.setenv("FLATC_EXECUTABLE", "")
    monkeypatch.setattr(shutil, "which", lambda _: None)

    fake_module = tmp_path / "pkg" / "subpkg" / "inner" / "_flatbuffer.py"
    fake_module.parent.mkdir(parents=True)
    fake_module.touch()
    monkeypatch.setattr(_flatbuffer, "__file__", str(fake_module))

    candidates = [
        fake_module.parent / "flatbuffers-flatc",
        fake_module.parents[2] / "data" / "bin" / "flatc",
        fake_module.parents[3] / "data" / "bin" / "flatc",
    ]

    expected = candidates[candidate_index]
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.touch()

    ensure_flatc_executable()

    assert os.environ["FLATC_EXECUTABLE"] == str(expected)


def test_continues_when_flatc_cannot_be_found(tmp_path, monkeypatch):
    monkeypatch.delenv("FLATC_EXECUTABLE", raising=False)
    which = Mock(return_value=None)
    monkeypatch.setattr(shutil, "which", which)

    fake_module = tmp_path / "pkg" / "subpkg" / "inner" / "_flatbuffer.py"
    fake_module.parent.mkdir(parents=True)
    fake_module.touch()
    monkeypatch.setattr(_flatbuffer, "__file__", str(fake_module))

    ensure_flatc_executable()

    which.assert_called_once_with("flatc")
    assert os.getenv("FLATC_EXECUTABLE") is None
