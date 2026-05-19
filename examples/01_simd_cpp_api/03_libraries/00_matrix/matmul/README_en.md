# High-Level API Basic Matmul Sample

## Overview

This sample calls the Matmul API to implement a matmul example. It performs matrix multiplication and bias addition on input matrices A and B, with the calculation formula: C = A * B + Bias.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products
- Atlas Inference Series Products AI Core

## Directory Structure

```
├── matmul
│   ├── scripts
│   │   ├── gen_data.py         // Input data and golden data generation script
│   │   └── verify_result.py    // Golden value comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   ├── matmul.asc              // Ascend C sample implementation & call sample
│   └── README.md               // Sample introduction
```

## Sample Description

- Sample Function:
  The Matmul sample calls the Matmul API to perform matrix multiplication and bias addition on input matrices A and B, with the calculation formula: C = A * B + Bias.

- Sample Specifications:
  In this sample: M = 128, N = 2048, K = 1024.
<table>
<tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="5" align="center">MatmulCustom</td></tr>
<tr><td rowspan="4" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
<tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
<tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
<tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
<tr><td rowspan="1" align="center">Sample Output</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_custom</td></tr>
</table>

- Sample Implementation:
  - Kernel Key Steps
    - The specific steps are as follows:
      - Create a Matmul object.
      - Perform initialization operations.
      - Set left matrix A, right matrix B, and bias matrix Bias.
      - Complete the matrix multiplication operation.
      - End the matrix multiplication operation.

  - Tiling Key Steps
      - Ascend C provides a set of Matmul Tiling APIs to help users obtain the Tiling parameters required for Matmul kernel computation. Simply pass in the A/B/C matrix information and call the API interface to obtain the relevant parameters in the TCubeTiling structure.
      - The process for obtaining Tiling parameters is as follows:
        - Create a Tiling object.
        - Set the parameter type information for A, B, C, Bias; and M, N, Ka, Kb shape information.
        - Call the GetTiling interface to obtain Tiling information.

  - Call Implementation
    Use the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this sample to build and run the sample.

- Configure Environment Variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on your current environment.
  - Default path, root user installed CANN software package
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, non-root user installed CANN software package
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, installed CANN software package
    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Sample Execution

  ```bash
  mkdir -p build && cd build;    # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py    # Generate test input data
  ./demo                        # Execute the compiled executable program to run the sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin    # Verify output result correctness, confirm algorithm logic is correct
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  For example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clear the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build Options Description

  | Parameter | Description | Options | Default |
  |------|------|---------|--------|
  | CMAKE_ASC_RUN_MODE | Run mode | npu, cpu, sim | npu |
  | CMAKE_ASC_ARCHITECTURES | NPU hardware architecture | dav-2201, dav-3510 | dav-2201 |

- Execution Result

  The execution result is as follows, indicating successful precision comparison.

  ```bash
  test pass!
  ```