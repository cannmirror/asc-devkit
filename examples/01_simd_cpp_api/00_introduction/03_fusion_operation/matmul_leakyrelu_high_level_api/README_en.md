# Matmul and LeakyRelu Fusion Sample

## Overview

This sample uses high-level APIs to implement Matmul and LeakyRelu activation function fusion computation, implementing fusion computation for matrix computation units and vector computation units.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── matmul_leakyrelu_high_level_api
│   ├── CMakeLists.txt          // Compilation project file
│   ├── data_utils.h            // Data read/write functions
│   ├── matmul_leakyrelu.asc    // Ascend C sample implementation & invocation sample
│   └── scripts
│       ├── gen_data.py         // Input data and golden data generation script file
│       └── verify_result.py    // Golden data comparison file
```

## Sample Description

- Sample Function:
  The MatmulLeakyRelu computation formula is:
  ```
  C = A * B + Bias
  C = C > 0 ? C : C * 0.001
  ```
  Sample parameters M = 512, K = 512, N = 1024. Sample specifications are shown in the table below:
  <table>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="4" align="center">MatmulLeakyRelu</td></tr>
  <tr><td rowspan="4" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">B</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">Bias</td><td align="center">[N]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">C</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">matmul_leakyrelu_custom</td></tr>
  </table>

- Sample Implementation:

  - Implementation Flow
    - Implement host-side Tiling computation through GenerateTiling
    - Complete core partitioning computation through CalcGMOffset
    - Complete matrix multiplication computation through Iterate interface
    - Implement activation function computation through LeakyRelu

  - Invocation Implementation
    Use the kernel invocation operator <<<>>> to call the kernel function.

## Compilation and Execution

Execute the following steps in the root directory of this sample to compile and run the sample.

- Configure Environment Variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on the current environment.
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
  mkdir -p build && cd build;                                               # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;                      # Compile project (default npu mode)
  python3 ../scripts/gen_data.py                                            # Generate test input data
  ./demo                                                                    # Execute compiled program, run sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output results are correct, confirm algorithm logic is correct
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note**: Before switching compilation modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then run cmake again.

- Compilation Options Description

| Option | Available Values | Description |
|--------|------------------|-------------|
| `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
| `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result
  The execution result is as follows, indicating that the accuracy comparison passed.
  ```bash
  test pass!
  ```