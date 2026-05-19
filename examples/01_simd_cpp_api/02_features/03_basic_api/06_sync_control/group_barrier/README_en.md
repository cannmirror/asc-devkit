# GroupBarrier Example

## Overview

This example implements correct synchronization between two groups of AIVs with dependencies. After Group A AIVs complete their computation, Group B AIVs rely on the computation results from Group A AIVs for subsequent computation. Group A is called the Arrive group, and Group B is called the Wait group.

> **Note:** This example only applies to the programming model based on TPipe and TQue.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── group_barrier
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── group_barrier.asc       // Ascend C example implementation & invocation example
```

## Example Description

- Example Functionality:
  The GroupBarrier example implements correct synchronization between two groups of AIVs with dependencies. After Group A AIVs complete their computation, Group B AIVs rely on the computation results from Group A AIVs for subsequent computation. Group A is called the Arrive group, and Group B is called the Wait group. This example does not perform input/output computation; it only verifies by having the Arrive group write specified values, then the Wait group reads those values and prints the correct values via printf.

- Example Specifications:
  <table>
  <tr><td rowspan="2" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">barGm</td><td align="center">[3072]</td><td align="center">uint8_t</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">z</td><td align="center">[8]</td><td align="center">int32_t</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">kernel_group_barrier</td></tr>
  </table>

- Example Implementation:

  The GroupBarrier example enables 8 AIV cores, where 2 AIV cores act as the Arrive group enabling atomic accumulation to write specified values to Global Memory and call the Arrive instruction; the remaining 6 AIV cores first call the Wait instruction to wait for the Arrive group to complete writing, then read from Global Memory and print the results via printf.

## Build and Run

Execute the following steps in the root directory of this example to build and run the example.

- Configure Environment Variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development toolkit on your current environment.
  - Default path, CANN package installed by root user

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, CANN package installed by non-root user

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Custom path install_path, CANN package installed

    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Example Execution

  ```bash
  mkdir -p build && cd build;      # Create and enter build directory
  cmake .. -DCMAKE_ASC_ARCHITECTURES=dav-2201;make -j;    # Build project
  ./demo                           # Execute the compiled executable program
  ```

  To use NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:

  ```bash
  cmake .. -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory and re-running cmake.

- Build Options Description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `sim` | Run mode: NPU execution, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` |  NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  The following output indicates successful example execution.

  ```bash
  [Block (0/6)]: OUTPUT = 24
  [Block (1/6)]: OUTPUT = 24
  [Block (2/6)]: OUTPUT = 24
  [Block (3/6)]: OUTPUT = 24
  [Block (4/6)]: OUTPUT = 24
  [Block (5/6)]: OUTPUT = 24
  ```