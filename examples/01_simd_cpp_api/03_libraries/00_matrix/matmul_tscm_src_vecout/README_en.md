# Matmul Direct Call Example Using TSCM Input with VECOUT Data Source

## Overview

This is a Matmul example using a user-defined TSCM input with data sourced from VECOUT. Developers can manage the L1 Buffer independently to efficiently utilize hardware resources. TSCM stands for Temp Swap Cache Memory, which is used to temporarily swap data to additional space. In this scenario, the developer manages the L1 Buffer and then provides the L1 Buffer address corresponding to the input matrix data as the Matmul input.

## Supported Products
- Ascend 950PR/Ascend 950DT

## Directory Structure
```
├── matmul_tscm_src_vecout
│   └── scripts
│       ├── gen_data.py         // Script for generating input data and golden data
│       └── verify_result.py    // Golden data verification file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_tscm_src_vecout.asc              // Ascend C example implementation & invocation example
```

## Example Description
- Example Functionality:
  This Matmul example customizes the data movement of matrix A from VECOUT to L1, ensuring that all data of matrix A resides in L1. When calling the Matmul API for computation, matrix A is set as TSCM input and matrix B is set as GM input. The example performs matrix multiplication on input matrices A and B with bias addition.

- Constraints
  - The matrix with TSCM input must be fully loadable into the L1 Buffer.

- Example Specifications:
  In this example: M = 32, N = 256, K = 32.
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
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_tscm_src_vecout_custom</td></tr>
  </table>

- Example Implementation:
  - Kernel Key Steps
    - Create a Matmul object. In the MatmulType for the left matrix A, POSITION is set to TSCM and SRC_POSITION is set to VECOUT.
      ```cpp
      AscendC::MatmulType<AscendC::TPosition::TSCM, CubeFormat::NZ, AType, 
                          IS_TRANS_A, LayoutMode::NONE, false, AscendC::TPosition::VECOUT>
      ```
    - Customize the data copy of left matrix A from VECOUT to TSCM, and set left matrix A, right matrix B, and Bias, where left matrix A is the TSCM input.
      ```cpp
      // Copy aMatrix from vecout to tscm
      AscendC::TSCM<AscendC::TPosition::VECOUT, 1> scm;
      pipe->InitBuffer(scm, 1, tiling.M * tiling.Ka * sizeof(AType));
      auto scmTensor = scm.AllocTensor<AType>();
      DataCopy(scmTensor, vecoutLocal, tiling.M * tiling.Ka);
      scm.EnQue(scmTensor);
      AscendC::LocalTensor<AType> scmLocal = scm.DeQue<AType>();

      matmulObj.SetTensorA(scmLocal, isTransA); // Set aMatrix tscm input
      matmulObj.SetTensorB(bGlobal, isTransB);
      if (tiling.isBias) {
          matmulObj.SetBias(biasGlobal);
      }
      ```

  - Invocation Implementation
    Use the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run
Execute the following steps in the root directory of this example to build and run the example.
- Configure Environment Variables
  Please select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on your current environment.
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
  mkdir -p build && cd build;    # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py    # Generate test input data
  ./demo                        # Execute the compiled executable program to run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin    # Verify output correctness, confirm algorithm logic is correct
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  For example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching compilation modes, you need to clean the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build Option Description

  | Parameter | Description | Available Values | Default Value |
  |------|------|---------|--------|
  | CMAKE_ASC_RUN_MODE | Run mode | npu, cpu, sim | npu |
  | CMAKE_ASC_ARCHITECTURES | NPU hardware architecture | dav-3510 | dav-3510 |

- Execution Result

  The following execution result indicates successful precision comparison.

  ```bash
  test pass!
  ```