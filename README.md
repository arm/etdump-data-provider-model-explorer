<!---
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
  --->

# ETDump Data Provider

The ETDump data provider overlays runtime profiling data from `.etdp` / ETDump files onto graphs produced by the ETRecord adapter.

## Usage

1. Launch Model Explorer with the ETRecord adapter and ETDump data provider enabled.
2. Open an `.etrecord` file.
3. Click `Add per-node data`.
4. Select `ETDump Profiling Overlay`.
5. Upload the `.etdp` file and run the provider.

```bash
model-explorer --extensions=etrecord_adapter_model_explorer,etdump_data_provider_model_explorer
```
