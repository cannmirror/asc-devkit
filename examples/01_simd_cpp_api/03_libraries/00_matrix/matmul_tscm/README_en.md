# Matmul Custom TSCM Input Direct Invocation Sample

## Overview

This sample demonstrates Matmul using user-defined TSCM input with data sourced from GM. Developers can manage L1 Buffer independently for efficient hardware resource utilization. TSCM stands for Temp Swap Cache Memory, used to temporarily swap data to additional space. In this scenario, the developer manages the L1 Buffer and then provides the L1 Buffer address corresponding to the input matrix data as Matmul input.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── matmul_tscm
│   └── scripts
│       ├── gen_data.py         // Input data and golden data generation script
│       └── verify_result.py    // Golden data verification file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read and write functions
│   └── matmul_tscm.asc         // Ascend C sample implementation & invocation sample
```

## Sample Description

- Sample Function:
  This Matmul sample customizes the data transfer of matrix A from Global Memory to L1 Buffer, keeping all matrix A data resident in L1 Buffer. When calling the Matmul API for computation, matrix A is the TSCM input and matrix B is set as GM input, performing matrix multiplication and bias addition on input matrices A and B.

- Constraints
  - Matrices with TSCM input must be fully loadable in L1 Buffer.

- Sample Specifications:
  In this sample: M = 64, N = 64, K = 64.
  <table>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="5" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_tscm_custom</td></tr>
  </table>

- Sample Implementation:
  - Key Kernel Steps
    - Create the Matmul object. In the MatmulType of left matrix A, POSITION is set to TSCM and SRC_POSITION defaults to GM.
      ```cpp
      AscendC::MatmulType<AscendC::TPosition::TSCM, CubeFormat::NZ, AType>
      ```
    - Customize the data transfer of left matrix A from GM to L1, set the left matrix A, right matrix B, and Bias, where left matrix A is TSCM input.
      ```cpp
      AscendC::TSCM<AscendC::TPosition::GM, 1> scm;
      pipe->InitBuffer(scm, 1, tiling.M * tiling.Ka * sizeof(AType));
      auto scmTensor = scm.AllocTensor<AType>();
      DataCopy(scmTensor, aGlobal, tiling.M * tiling.Ka);
      scm.EnQue(scmTensor);
      AscendC::LocalTensor<AType> scmLocal = scm.DeQue<AType>();

      matmulObj.SetTensorA(scmLocal, IS_TRANS_A);
      matmulObj.SetTensorB(bGlobal, IS_TRANS_B);
      if (tiling.isBias) {
          matmulObj.SetBias(biasGlobal);
      }
      matmulObj.IterateAll(cGlobal);
      matmulObj.End();

      scm.FreeTensor(scmLocal);
      ```

  - Invocation Implementation
    Use the kernel call operator <<<>>> to invoke the kernel function.

## Compilation and Execution

Execute the following steps in the root directory of this sample to compile and run the sample.

- Configure Environment Variables
  Select the corresponding environment variable configuration command based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on your current environment.
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
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin    # Verify output correctness, confirm algorithm logic
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  For example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching compilation modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Compilation Options Description

  | Parameter | Description | Options | Default |
  |------|------|---------|--------|
  | CMAKE_ASC_RUN_MODE | Run mode | npu, cpu, sim | npu |
  | CMAKE_ASC_ARCHITECTURES | NPU hardware architecture | dav-2201, dav-3510 | dav-2201 |

- Execution Result

  The following execution result indicates successful precision comparison.

  ```bash
  test pass!
  ```