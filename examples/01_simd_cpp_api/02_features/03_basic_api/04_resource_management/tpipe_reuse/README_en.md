# TPipe Reuse Example

## Overview

This example demonstrates repeated allocation and release of TPipe using the `TPipe::Init` and `TPipe::Destroy` interfaces.

> **Note:** This example only applies to the programming model based on TPipe and TQue.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── tpipe_reuse
│   ├── scripts
│   │   ├── gen_data.py         // Script to generate input data and golden data
│   │   └── verify_result.py    // Script to verify output data matches golden data
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── tpipe_reuse.asc         // Ascend C example implementation & invocation example
```

## Example Description

- Example Functionality

  This example demonstrates repeated allocation and release of TPipe using Muls computation.

- Example Specifications

  <table>
    <tr>
      <td align="center">Category</td>
      <td align="center">name</td>
      <td align="center">shape</td>
      <td align="center">data type</td>
      <td align="center">format</td>
    </tr>
    <tr>
      <td align="center">Example Input</td>
      <td align="center">x</td>
      <td align="center">[1, 128]</td>
      <td align="center">float</td>
      <td align="center">ND</td>
    </tr>
    <tr>
      <td align="center">Example Output</td>
      <td align="center">z</td>
      <td align="center">[1, 128]</td>
      <td align="center">float</td>
      <td align="center">ND</td>
    </tr>
    <tr>
      <td align="center">Kernel Function Name</td>
      <td colspan="4" align="center">tpipe_reuse_custom</td>
    </tr>
  </table>

- Example Implementation

  - Kernel Implementation

    - Call the TPipe::Init interface to initialize the TPipe object, and call the TPipe::InitBuffer interface to allocate memory space for TQue.

    - Call the DataCopy basic API to move data from GM (Global Memory) to UB (Unified Buffer).

    - Call the Muls interface to multiply the input tensor by a scalar value of 3.

    - Call the DataCopy basic API to move computation results from UB (Unified Buffer) to GM (Global Memory).

    - Call the TPipe::Destroy interface to destroy the TPipe object, enabling TPipe reuse.

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