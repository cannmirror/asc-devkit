# Custom Operator Tiling Sink Graph Mode Invocation Sample

## Overview

This sample demonstrates how to invoke custom operators in PyTorch graph mode based on a sample custom operator project, and optimize scheduling performance by enabling Tiling sink to the device side.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── tiling_sink_programming
│   └── test_add_custom_tiling_sink.py
```

## Code Implementation

The sample script `test_add_custom_tiling_sink.py` consists of three key parts:
1. Register a custom operator in PyTorch and provide placeholder implementations for Meta/CPU/PrivateUse1 to ensure it can be incorporated into the graph.
2. Register an FX to GE converter that maps `add_custom_tiling_sink` to the GE-side custom operator `AddCustomTilingSink`.
3. Enable graph mode execution via `torch.compile` and enable the `tiling_schedule_optimize` configuration.

## Build and Run

- Install PyTorch and Ascend Extension for PyTorch Plugin

  Refer to the installation instructions in the [pytorch: Ascend Extension for PyTorch](https://gitcode.com/Ascend/pytorch) open-source repository or [Ascend Extension for PyTorch Ascend Community](https://hiascend.com/document/redirect/Pytorch-index) to select a supported `Python` version and complete the installation of `torch` and `torch-npu`.

- Build, Package, and Deploy Custom Operator Project

  Before running this sample, navigate to the [custom operator project sample](../../00_compilation/custom_op/) directory to complete the build, packaging, and deployment.

- Install Prerequisites

  ```bash
  pip3 install expecttest
  ```

- Configure Environment Variables

  Select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development toolkit on your current environment.
  - Default path, CANN package installed by root user

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, CANN package installed by non-root user

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, CANN package installed

    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Run the Sample

  Execute the following steps in the root directory of this sample to run the sample.

  ```bash
  python3 test_add_custom_tiling_sink.py
  ```

  The following output indicates successful accuracy verification:

  ```bash
  Ran 1 test in **s
  OK
  ```