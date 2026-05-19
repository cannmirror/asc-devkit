# Where Example

## Overview

This example implements the function of selecting elements from two source operands based on a specified condition using the Where high-level API. Both source operands can be LocalTensor or scalars.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```plain
├── where
│   ├── scripts
│   │   └── gen_data.py         // Input data and ground truth data generation script
│   ├── CMakeLists.txt          // Build configuration file
│   ├── data_utils.h            // Data read and write functions
│   └── where.asc               // Ascend C example implementation & invocation example
```

## Example Description

- Example Function:
  Selects elements from two source operands based on a specified condition to generate the destination operand. Both source operands can be LocalTensor or scalars.

  The calculation formula is as follows:
  $$dst_i = \begin{cases}
  src0, & if condition \\
  src1, & otherwise
  \end{cases}$$

- Example Specification:
  <table>
  <tr><td rowspan="1" align="center">Example Type(OpType)</td><td colspan="4" align="center"> where </td></tr>

  <tr><td rowspan="5" align="center">Example Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src0</td><td align="center">[1, 32]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">src1</td><td align="center">[1, 32]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">condition</td><td align="center">[1, 32]</td><td align="center">bool</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">Example Output</td></tr>
  <tr><td align="center">dst</td><td align="center">[1, 32]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">where_custom</td></tr>
  </table>

- Example Implementation:
  This example implements the where_custom example with a fixed shape of inputs src0[1, 32], src1[1, 32], condition[1, 32], and output dst[1, 32].

  - Kernel Implementation

    Uses the Where high-level API to select elements from two source operands based on a condition. Supports tensor and scalar mixed mode.

  - Invocation Implementation
    Uses the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this example to build and run the example.

- Configure Environment Variables
  Select the corresponding command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development toolkit on your current environment.
  - Default path, CANN package installed by root user

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, CANN package installed by non-root user

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, CANN package installed

    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Example Execution

  ```bash
  mkdir -p build && cd build;      # Create and enter the build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;    # Build the project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                           # Execute the compiled executable to run the example
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  For example:

  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clear the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build Options Description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` (default) | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  The execution result is as follows, indicating successful precision comparison.

  ```bash
  test pass!
  ```