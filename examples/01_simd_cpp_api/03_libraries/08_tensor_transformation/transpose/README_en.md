# Transpose Example

## Overview

This example converts data layout from NZ format to ND format and performs dimension swapping using the Transpose high-level API. The example implements a scenario where an NZ format Tensor is converted to an ND format Tensor and its first and second dimensions are swapped.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```plain
├── transpose
│   ├── scripts
│   │   └── gen_data.py         // Input data and ground truth data generation script
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read and write functions
│   └── transpose.asc           // Ascend C example implementation & invocation example
```

## Example Description

- Example Function:
  This example uses the Transpose high-level API to convert data layout from NZ to ND and swap the first and second dimensions of the ND matrix.

- Example Specifications:
  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="4" align="center">transpose</td></tr>

  <tr><td rowspan="3" align="center">Example Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 2, 2, 4, 16, 16]</td><td align="center">half</td><td align="center">NZ</td></tr>
  <tr><td rowspan="2" align="center">Example Output</td></tr>
  <tr><td align="center">y</td><td align="center">[1, 64, 2, 32]</td><td align="center">half</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">transpose_custom</td></tr>
  </table>

- Example Implementation:
  This example implements a Transpose example with fixed shapes: input x[1, 2, 2, 4, 16, 16], output y[1, 64, 2, 32]. For the NZ2ND scenario, it swaps axes 1 and 2.

  Input shape is [B, N, H/N/16, S/16, 16, 16] = [1, 2, 2, 4, 16, 16]

  After NZ2ND conversion, the shape is [B, N, S, H/N] = [1, 2, 64, 32]

  After transposing axes 1 and 2, the output shape is [B, S, N, H/N] = [1, 64, 2, 32]

  - Kernel Implementation
    Use the Transpose high-level API to complete the Transpose computation and obtain the final result, then move it to external memory.

  - Tiling Implementation
    Use the GetTransposeTilingInfo interface provided by Ascend C to obtain the required Tiling parameters, and call the GetTransposeMaxMinTmpSize interface to obtain the temporary space size required for Transpose interface computation.

  - Invocation Implementation
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

- Example Execution

  ```bash
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                           # Execute the compiled executable program to run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output result correctness, confirm algorithm logic is correct
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  For example:

  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clear the cmake cache by running `rm CMakeCache.txt` in the build directory and then re-run cmake.

- Build Options Description

  | Option | Available Values | Description |
  |--------|------------------|-------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` (default), `dav-2201` | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT, dav-2201 corresponds to Atlas A2/A3 series |

- Execution Result

  The execution result is as follows, indicating successful accuracy comparison.

  ```bash
  test pass!
  ```