# Matmul NBuffer33 Template Strategy Direct Call Example

## Overview
Matmul example using the NBuffer33 algorithm, achieving load/store bandwidth balance to improve efficiency.

## Supported Products
- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure
```
├── matmul_nbuffer33
│   └── scripts
│       ├── gen_data.py         // Script to generate input data and ground truth data
│       └── verify_result.py    // Ground truth comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read and write functions
│   └── matmul_nbuffer33.asc    // Ascend C example implementation & invocation example
```
## Example Description
- Example Function:
  The Matmul example uses the template parameter MatmulPolicy configured as NBuffer33MatmulPolicy. The single-core computation divides matrix A into 3x3 basic blocks. These 3x3 basic blocks of matrix A are fully loaded and kept in L1 Buffer. Each time, they are multiplied with 3x1 basic blocks of matrix B, while DoubleBuffer parallelly loads the next 3x1 basic blocks of matrix B needed for computation, until the matrix multiplication in the singleCoreN direction is completed.

- Constraints
  - Only the MDL template is supported.
  - Only pure Cube mode (matrix computation only) is supported; MIX mode (including matrix computation and vector computation) is not currently supported.
  - Only the IterateAll interface is supported to obtain the computation result matrix C from Matmul.
  - stepM, stepKa, and stepKb must be less than or equal to 3.
  - Users need to ensure that the sum of the fully loaded basic block size of matrix A and the loaded basic block size of matrix B does not exceed the L1 Buffer size.

- Example Specifications:
  In this example: M = 256, N = 512, K = 192.
  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="5" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_nbuffer33_custom</td></tr>
  </table>
- Example Implementation:
  - Kernel Key Steps
    - Specific Steps:
      - Create a Matmul object and configure the MDL template and NBuffer33MatmulPolicy.
          ```cpp
          AscendC::Matmul<AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType, IS_TRANS_A>,
                          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType, IS_TRANS_B>,
                          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>,
                          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>, CFG_MDL,
                          AscendC::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                          AscendC::Impl::Detail::NBuffer33MatmulPolicy>
              matmulObj;
          ```

  - Tiling Key Steps
    - Set the parameter type information for A, B, C, and Bias; shape information for M, N, Ka, and Kb; and enable NBuffer33 mode.
      ```cpp
      matmul_tiling::MatmulConfigParams matmulConfigParams(1, false, matmul_tiling::ScheduleType::N_BUFFER_33,
                                                            matmul_tiling::MatrixTraverse::NOSET, false);
      tilingApi.SetMatmulConfigParams(matmulConfigParams);
      ```

  - Invocation Implementation
    Use the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run
Execute the following steps in the root directory of this example to build and run the example.
- Configure Environment Variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on your current environment.
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

- Run the Example

  ```bash
  mkdir -p build && cd build;    # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py    # Generate test input data
  ./demo                        # Execute the compiled program to run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin    # Verify output result correctness and confirm algorithm logic
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  For example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clear the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build Options

  | Parameter | Description | Available Values | Default |
  |--------|--------|---------|--------|
  | CMAKE_ASC_RUN_MODE | Run mode | npu, cpu, sim | npu |
  | CMAKE_ASC_ARCHITECTURES | NPU hardware architecture | dav-2201, dav-3510 | dav-2201 |

- Execution Result

  The following execution result indicates that the precision comparison passed:
  ```bash
  test pass!
  ```
  The following execution result indicates that the precision comparison passed.
  ```bash
  test pass!
  ```