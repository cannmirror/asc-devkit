# BitwiseAnd Sample

## Overview

This sample implements bitwise AND operation on two inputs using the BitwiseAnd high-level API.

> **Interface Note:** In addition to the `BitwiseAnd` interface used in this sample, Ascend C also provides the following bitwise operation high-level API interfaces:
>
> - **BitwiseNot**: Bitwise NOT operation.
> - **BitwiseOr**: Bitwise OR operation.
> - **BitwiseXor**: Bitwise XOR operation.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```plain
├── bitwiseand
│   ├── scripts
│   │   └── gen_data.py         // Input data and golden data generation script
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read and write functions
│   └── bitwiseand.asc          // Ascend C operator implementation & call sample
```

## Sample Description

- Sample Function:
  Performs bitwise AND operation on two inputs.

  The calculation formula is as follows:
  $$
  dst_i = src0_i \& src1_i
  $$

- Sample Specifications:
  <table>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="4" align="center"> bitwiseand </td></tr>

  <tr><td rowspan="4" align="center">Sample Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src0</td><td align="center">[1, 1024]</td><td align="center">int32_t</td><td align="center">ND</td></tr>
  <tr><td align="center">src1</td><td align="center">[1, 1024]</td><td align="center">int32_t</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">Sample Output</td></tr>
  <tr><td align="center">dst</td><td align="center">[1, 1024]</td><td align="center">int32_t</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">bitwiseand_custom</td></tr>
  </table>

- Sample Implementation:
  This sample implements bitwiseand_custom with a fixed shape of input src0[1, 1024], src1[1, 1024] and output dst[1, 1024].

  - Kernel Implementation

    Uses the BitwiseAnd high-level API interface to complete the bitwise AND calculation, then moves the final result to external storage.

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
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                           # Execute the generated executable program to run the sample
  ```

  For CPU debugging or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:

  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debugging mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clear the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build Option Description

  | Option | Available Values | Description |
  |--------|------------------|-------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debugging, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` (default) | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  The execution result is as follows, indicating that the accuracy comparison passed.

  ```bash
  test pass!
  ```