# Mmad GEMV Example

## Overview

This example introduces matrix multiplication in GEMV (M=1) mode.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── mmad_gemv
│   ├── img                         // Illustration files in this document
│   ├── scripts
│   │   ├── gen_data.py             // Input data and golden data generation script
│   │   └── verify_result.py        // Verification script for output data and golden data
│   ├── CMakeLists.txt              // Build project file
│   ├── data_utils.h                // Data read/write functions
│   └── mmad_gemv.asc               // Ascend C example implementation & calling example
```

## Example Description

GEMV mode refers to the scenario where M=1 in Mmad computation, where a left matrix A with shape (1, K) performs matrix multiplication with a right matrix B with shape (K, N). When M=1, GEMV mode is automatically enabled. Only on Ascend 950PR/Ascend 950DT, it can be disabled by setting `mmadParams.disableGemv = true`. In this example, the compilation parameter `DISABLE_GEMV` is used to select whether to disable GEMV mode. 0 means GEMV is enabled, 1 means GEMV is disabled.

Using M=1, K=256, N=32, and half data type for both left and right matrices as a concrete example, the Mmad computation process in GEMV mode and non-GEMV mode is explained.

- GEMV Mode

  When moving matrix A from A1 to A2, the 1 * 256 vector is processed as a 16 * 16 matrix. The LoadData interface is called once to complete the matrix movement of 16 * 16 fractal size. The movement of matrix B and the matrix multiplication computation are the same as the basic scenario, as shown in Figure 1 below.

  <p align="center">
  <img src="img/开启gemv.png" width="1100">
  </p>
  <p align="center">
  Figure 1: GEMV mode, Mmad computation diagram
  </p>

- Non-GEMV Mode

  When moving matrix A from A1 to A2, the 1 * 256 vector is processed as non-aligned matrix data. The M direction is aligned to 16 before movement. The LoadData interface is called to move a 16 * 16 fractal size matrix each time, with a total of CeilDiv(K, 16)=16 movements, resulting in increased data movement volume and poorer performance compared to GEMV mode, as shown in Figure 2 below.

  <p align="center">
  <img src="img/关闭gemv.png" width="1100">
  </p>
  <p align="center">
  Figure 2: Non-GEMV mode, Mmad computation diagram
  </p>

## Constraint Description

- In Mmad computation, to enable GEMV mode, the parameter `mmadParams.m` must equal 1.
- In GEMV scenarios, transposition is not supported when moving left matrix A from L1 to L0A.

## Build and Run

Execute the following steps in the root directory of this example to build and run the operator.

- Configure environment variables

  Please select the corresponding environment variable configuration command based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on the current environment.

  - Default path, root user installed CANN software package
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, non-root user installed CANN software package
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, installed CANN software package
    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Example execution
  ```bash
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DDISABLE_GEMV=0 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                           # Execute the compiled executable program, run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output result correctness, confirm algorithm logic is correct
  ```

  When using CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DDISABLE_GEMV=0 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DDISABLE_GEMV=0 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clean cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then re-run cmake.

- Build option description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products/Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  | `DISABLE_GEMV` | `0` (default), `1` | Whether to disable GEMV mode. `Only supported when CMAKE_ASC_ARCHITECTURES==dav-3510` |

- Execution result

  The execution result is shown below, indicating the precision comparison passed.
  ```bash
  test pass!
  ```