# CumSum Sample

## Overview

This sample implements the functionality of computing cumulative sums along rows or columns of a tensor using the CumSum high-level API.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```plain
├── cumsum
│   ├── scripts
│   │   └── gen_data.py         // Input data and golden data generation script
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read and write functions
│   └── cumsum.asc              // Ascend C sample implementation & call sample
```

## Sample Description

- Sample Function:
  Computes cumulative sum along rows or columns of an input tensor. Each element in the output result is the cumulative sum of all elements in the corresponding position and all previous rows or columns in the input tensor.
- Sample Specifications:
  <table>
  <caption>Table 1: Sample Specifications</caption>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="4" align="center"> cumsum </td></tr>

  <tr><td rowspan="3" align="center">Sample Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src</td><td align="center">[32, 160]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="3" align="center">Sample Output</td></tr>
  <tr><td align="center">dst</td><td align="center">[32, 160]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">lastRow</td><td align="center">[1, 160]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">cumsum_custom</td></tr>
  </table>

- Sample Implementation:
  This sample implements cumsum_custom with a shape of input src[32, 160] and outputs dst[32, 160], lastRow[1, 160].

  - Kernel Implementation

    Uses the CumSum high-level API interface to complete the cumsum calculation.

  - Tiling Implementation

    The host side uses GetCumSumMaxMinTmpSize to obtain the maximum and minimum temporary space required for the CumSum interface calculation.

  - Call Implementation
    Uses the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this sample to build and run the operator.

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

- Sample Execution

  ```bash
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                           # Execute the generated executable program to run the sample
  ```

  For CPU debugging or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  Example:

  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debugging mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clear the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build Option Description

  | Option | Available Values | Description |
  |--------|------------------|-------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debugging, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2/A3 series, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  The execution result is as follows, indicating that the accuracy comparison passed.

  ```bash
  test pass!
  ```