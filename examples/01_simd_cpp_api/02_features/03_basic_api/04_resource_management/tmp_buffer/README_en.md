# VectorAdd Example Using TmpBuf

## Overview

This example demonstrates the initialization of TBuf memory space using the `TPipe::InitBuffer` interface and uses the TBuf temporary buffer for data conversion during computation, implementing a vector addition (Add) example for the bfloat16_t data type.

> **Note:** This example only applies to the programming model based on TPipe and TQue.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── tmp_buffer
│   ├── scripts
│   │   ├── gen_data.py         // Script to generate input data and golden data
│   │   └── verify_result.py    // Script to verify output data matches golden data
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── tmp_buffer.asc          // Ascend C example implementation & invocation example
```

## Example Description

- Example Functionality

  This example calls the Cast interface to convert bfloat16_t input data to float type and stores it in the TBuf temporary buffer. After completing the Add computation, it calls the Cast interface to convert the result back to bfloat16_t type.

- Example Specifications

  <table>
    <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="4" align="center">Add</td></tr>
    <tr><td rowspan="3" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
    <tr><td align="center">x</td><td align="center">[1, 2048]</td><td align="center">bfloat16_t</td><td align="center">ND</td></tr>
    <tr><td align="center">y</td><td align="center">[1, 2048]</td><td align="center">bfloat16_t</td><td align="center">ND</td></tr>
    <tr><td rowspan="1" align="center">Example Output</td><td align="center">z</td><td align="center">[1, 2048]</td><td align="center">bfloat16_t</td><td align="center">ND</td></tr>
    <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">tmp_buffer_custom</td></tr>
  </table>

- Example Implementation

  - Kernel Implementation

    - Call the TPipe::InitBuffer interface to allocate memory space for TQue and TBuf, where TBuf is used to store temporary data.

    - Call the DataCopy basic API to move data from GM (Global Memory) to UB (Unified Buffer).

    - Call the Cast interface to convert bfloat16_t data to float type and store it in the TBuf temporary buffer.

    - Call the Add interface to perform addition on two input tensors.

    - Call the Cast interface to convert float type computation results to bfloat16_t type and store it in the UB (Unified Buffer) space allocated by TQue.

    - Call the DataCopy basic API to move computation results from UB (Unified Buffer) to GM (Global Memory).

  - Invocation Implementation

    Use the kernel call operator `<<<>>>` to invoke the kernel function.

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
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                           # Execute the compiled executable program
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output correctness
  ```

  To use CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  Examples:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory and re-running cmake.

- Build Options Description

  | Parameter | Description | Available Values | Default Value |
  |------|------|---------|--------|
  | CMAKE_ASC_RUN_MODE | Run mode | npu, cpu, sim | npu |
  | CMAKE_ASC_ARCHITECTURES | NPU hardware architecture | dav-2201, dav-3510 | dav-2201 |

- Execution Result

  The following output indicates successful precision comparison:
  ```bash
  test pass!
  ```