# PhiloxRandom Sample

## Overview

This sample implements random number generation using the PhiloxRandom high-level API. It supports generating a specified number of random numbers based on the Philox algorithm and is suitable for scenarios that require high-quality random numbers.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```plain
├── philoxrandom
│   ├── scripts
│   │   └── gen_data.py         // Input data and ground truth data generation script
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read and write functions
│   └── philoxrandom.asc        // Ascend C sample implementation and call sample
```

## Sample Description

- Sample Function:  
  Based on the Philox random number generation algorithm, generate a number of random numbers given a random seed.
- Sample Specification:  
  <table>
  <caption>Table 1: Sample Input/Output Specification</caption>
  <tr><td rowspan="1" align="center">Sample Type(OpType)</td><td colspan="4" align="center"> philoxrandom </td></tr>

  <tr><td rowspan="2" align="center">Sample Output</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">dst</td><td align="center">[1, 1280]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">philoxrandom_custom</td></tr>
  </table>

- Sample Implementation:  
  This sample implements a philoxrandom_custom sample with fixed shape output dst[1, 1280].

  - Kernel Implementation

    The computation logic includes the following steps:
    1. Allocate output space in Local Memory (Unified Buffer)
    2. Initialize the output space to zero
    3. Generate random numbers using the PhiloxRandom high-level API
    4. Move the generated random numbers from Local Memory to Global Memory

  - Call Implementation  
    Use the kernel call operator <<<>>> to call the kernel function.

## Build and Run

Execute the following steps in the root directory of this sample to build and run the sample.

- Configure Environment Variables  
  Select the appropriate environment variable configuration command based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on your current environment.
  - Default path, root user installed CANN software package

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, non-root user installed CANN software package

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
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                           # Execute the compiled executable program to run the sample
  ```

  When using CPU debugging or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  Examples:

  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debugging mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching compilation modes, you need to clear the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build Options Description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debugging, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` (default) | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  The execution result is as follows, indicating successful precision comparison.

  ```bash
  test pass!
  ```