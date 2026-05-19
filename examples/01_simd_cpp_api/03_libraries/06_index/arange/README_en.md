# Arange Example

## Overview

This example implements arithmetic sequence generation using the Arange high-level API. Given a start value, difference value, and length, it returns an arithmetic sequence. The example supports using the input start value and difference value as scalar parameters, generating an arithmetic sequence of the specified length in on-chip memory through the Arange API, and then moving the results to external memory.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── arange
│   ├── scripts
│   │   └── gen_data.py         // Input data and ground truth data generation script
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read and write functions
│   └── arange.asc              // Ascend C example implementation & invocation example
```

## Example Description

- Example Function:
  This example implements arithmetic sequence generation. Given a start value, difference value, and length, it returns an arithmetic sequence.
- Example Specifications:
  <table>
  <caption>Table 1: Example Input/Output Specifications</caption>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="4" align="center"> arange </td></tr>

  <tr><td rowspan="4" align="center">Example Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">firstGm</td><td align="center">[1, 1]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">diffGm</td><td align="center">[1, 1]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="2" align="center">Example Output</td></tr>
  <tr><td align="center">outputGm</td><td align="center">[128, 1]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">arange_custom</td></tr>
  </table>

- Example Implementation:
  This example implements arithmetic sequence generation with fixed shapes: input firstGm[1, 1], diffGm[1, 1], output outputGm[128, 1].

  - Kernel Implementation:
    The computation logic uses the Arange high-level API to complete the arithmetic sequence generation, and then moves the results out.

  - Invocation Implementation:
    Use the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this example to build and run the example.
- Configure Environment Variables
  Select the appropriate environment variable configuration command based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on your current environment.
  - Default path, CANN software package installed by root user
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, CANN software package installed by non-root user
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, CANN software package installed
    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Example Execution (NPU Mode)
  ```bash
  mkdir -p build && cd build;      # Create and enter build directory
  cmake .. -DCMAKE_ASC_ARCHITECTURES=dav-2201;make -j;             # Build project
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                           # Execute the compiled executable program to run the example
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  For example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clear the cmake cache by running `rm CMakeCache.txt` in the build directory and then re-run cmake.

- Build Options Description

  | Option | Available Values | Description |
  |--------|------------------|-------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2/A3 series, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  The execution result is as follows, indicating successful accuracy comparison.
  ```bash
  test pass!
  ```