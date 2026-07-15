# VectorAdd Operator Direct-Call Sample (Tensor API)

## Overview

This sample describes how to directly call an Add operator kernel function based on the Tensor API. The operator supports single-core execution. Unlike the traditional Ascend C Vector API, this sample uses Tensor API interfaces such as `MakeTensor`, `MakeCopy`, and `Transform` to complete data movement and computation, demonstrating a higher-level tensor programming model.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```text
├── vector_add_tensor_api
│   ├── CMakeLists.txt           // Build project file
│   ├── run.sh                   // One-click build and run script
│   └── vector_add.asc           // Ascend C operator implementation and invocation sample, including data generation and result verification
```

## Operator Description

- Operator function:

  The Add operator adds two input tensors and returns the addition result. The corresponding mathematical expression is:

  ```text
  z = x + y
  ```

- Operator specification:

  <table>
  <tr><td rowspan="1" align="center">Operator Type (OpType)</td><td colspan="4" align="center">Add</td></tr>
  <tr><td rowspan="3" align="center">Inputs</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">1 * 256</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">1 * 256</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Output</td><td align="center">z</td><td align="center">1 * 256</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">add_custom</td></tr>
  </table>

- Operator implementation:

  This sample uses the Tensor API programming model. The main process is as follows:

  1. Create tensor objects on the GM and UB sides by using `MakeTensor` and `MakeFrameLayout<NDExtLayoutPtn>`.
  2. Create copy atom operations by using `MakeCopy(CopyGM2UB{})` and `MakeCopy(CopyUB2GM{})`, and use the `Copy` interface to move data between GM and UB.
  3. Use `Transform<Add>` to perform element-wise addition on the two input tensors.
  4. Use `asc_sync_notify` and `asc_sync_wait` for pipeline synchronization to ensure dependencies between data movement and computation.

  The host program generates random input data and computes golden results in the `main` function. After the kernel function is executed, the result on the Device side is copied back to the Host side and compared element by element to verify correctness.

  - Invocation implementation:

    Use the kernel call syntax `<<<>>>` to invoke the kernel function, and pass in the Device-side memory addresses and data length.

## Build and Run

Run the following steps in the sample root directory to build and execute the operator.

- Configure environment variables.

  Select the environment variable configuration command according to the [installation mode](../../../../../docs/quick_start.md#prepare&install) of the CANN development toolkit package in the current environment.

  - Default path, CANN package installed by the root user:

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, CANN package installed by a non-root user:

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - CANN package installed in a specified `install_path`:

    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Method 1: Use `run.sh` for one-click execution.

  ```bash
  bash run.sh
  ```

- Method 2: Build and run manually.

  ```bash
  mkdir -p build && cd build   # Create and enter the build directory
  cmake .. && make -j4         # Build the project
  ./demo                       # Run the sample, including random data generation and result verification
  ```

  If the output is as follows, the precision comparison is successful.

  ```text
  CompareResult passed, all 256 elements are correct.
  ```
