# SyncAll Inter-core Synchronization Sample

## Overview

This sample demonstrates the usage of the SyncAll inter-core synchronization interface, applicable to scenarios where different cores operate on the same global memory with data dependencies such as read-after-write, write-after-read, and write-after-write. The sample uses 8 cores for data processing, with each core processing 32 float data elements (total data size of 256). Each core multiplies data by 2 and then accumulates with results from other cores, using SyncAll to achieve inter-core synchronization and ensure all cores complete computation before final accumulation.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── sync_all
│   ├── scripts
│   │   ├── gen_data.py         // Input data and golden data generation script
│   │   └── verify_result.py    // Verification script for comparing output data with golden data
│   ├── CMakeLists.txt          // Build configuration file
│   ├── data_utils.h            // Data read/write functions
│   └── sync_all.asc            // Ascend C sample implementation & invocation sample
```

## Sample Description

- Sample Function:
  Uses 8 cores for data processing, with each core processing 32 float data elements. Each core multiplies its data by 2 and then adds it to data from other cores that have also been multiplied by 2, saving intermediate results to workGm. SyncAll is called to achieve synchronization among the 8 cores, ensuring all cores complete computation before final accumulation.

- Sample Specifications:
  <table border="2">
  <caption>Table 1: Sample Specifications</caption>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="4" align="center">SyncAll</td></tr>
  <tr><td rowspan="3" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[256]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">work</td><td align="center">[256]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">z</td><td align="center">[256]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">sync_all_custom</td></tr>
  </table>

## Build and Run

Execute the following steps in the sample root directory to build and run the sample.

- Configure Environment Variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on your current environment.
  - Default path, root user installed CANN package
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, non-root user installed CANN package
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, installed CANN package
    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Sample Execution
  ```bash
  mkdir -p build && cd build;
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;
  python3 ../scripts/gen_data.py
  ./demo
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin
  ```

  When using CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;
  ```

  > **Note:** Before switching build modes, clean cmake cache by running `rm CMakeCache.txt` in the build directory and re-run cmake.

  | Parameter | Description | Available Values | Default |
  |------|------|---------|--------|
  | CMAKE_ASC_RUN_MODE | Run mode | npu, cpu, sim | npu |
  | CMAKE_ASC_ARCHITECTURES | NPU hardware architecture | dav-2201, dav-3510 | dav-2201 |

  The following execution result indicates successful precision comparison.
  ```bash
  test pass!
  ```