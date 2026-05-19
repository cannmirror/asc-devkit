# Mmad unitFlag Feature Example

## Overview

This example introduces how to use the unitFlag feature when calling the Mmad instruction.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── mmad_unitflag
│   ├── scripts
│   │   ├── gen_data.py             // Input data and golden data generation script
│   │   └── verify_result.py        // Verification script for output data and golden data
│   ├── CMakeLists.txt              // Build project file
│   ├── data_utils.h                // Data read/write functions
│   └── mmad_unitflag.asc           // Ascend C example implementation & calling example
```

## Operator Description

unitFlag is a fine-grained parallelism feature of Mmad and Fixpipe instructions. When this feature is enabled, hardware moves out computation results immediately after completing each fractal computation. This feature is not applicable to scenarios where accumulation is performed in the L0C Buffer.

In this example, matrix A has shape [128, 512] and matrix B has shape [512, 256]. When executing the Mmad instruction, 8 iteration loops are performed along the K axis, with each iteration having a K length of 64. The unitFlag values for each iteration of Mmad computation are described in Table 1:

<table border="2">
<caption>Table 1: unitFlag Value Description</caption>
  <tr>
    <td >unitFlag Value</td>
    <td>Description</td>
    <td>Example Implementation</td>
  </tr>
  <tr>
    <td>0</td>
    <td>Reserved value</td>
    <td>-</td>
  </tr>
  <tr>
    <td>2</td>
    <td>Enable unitFlag. After hardware executes the instruction, the unitFlag feature will not be disabled</td>
    <td>Set unitFlag to 2 for the first 7 Mmad operations to ensure subsequent Mmad can write to L0C</td>
  </tr>
  <tr>
    <td>3</td>
    <td>Enable unitFlag. After hardware executes the instruction, the unitFlag feature will be disabled</td>
    <td>Set to 3 for the last Mmad and Fixpipe to ensure Fixpipe can read from L0C</td>
  </tr>
</table>

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
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DENABLE_UNITFLAG=1 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                           # Execute the compiled executable program, run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output result correctness, confirm algorithm logic is correct
  ```

  When using CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DENABLE_UNITFLAG=1 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DENABLE_UNITFLAG=1 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clean cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then re-run cmake.

- Build option description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products/Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  | `ENABLE_UNITFLAG` | `0`, `1` (default) | Whether to enable unitFlag |

- Execution result

  The execution result is shown below, indicating the precision comparison passed.
  ```bash
  test pass!
  ```