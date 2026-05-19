# Matmul User-Managed CO1 Direct Invocation Sample

## Overview

This sample demonstrates Matmul with user-managed CO1 (L0C Buffer). The computation result matrix C is saved in the CO1 position, and then the basic API Fixpipe is called to transfer the result from CO1 to Global Memory.

## Supported Products

- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── matmul_co1_output
│   └── scripts
│       ├── gen_data.py         // Input data and golden data generation script
│       └── verify_result.py    // Golden value comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_co1_output.asc   // Ascend C sample implementation & invocation sample
```

## Sample Description

- **Sample Function:**
  The Matmul sample calls the Matmul API to perform matrix multiplication and add bias offset on input matrices A and B. The computation result matrix C is saved in the CO1 position, and then the basic API Fixpipe is called to transfer the result from CO1 to Global Memory.

- **Sample Specifications:**
  In this sample: M = 32, N = 256, K = 128.

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
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_co1_output_custom</td></tr>
  </table>

- **Sample Implementation:**
  - The Iterate interface supports two usage modes:
    - Internal CO1 management: Users do not need to manage the allocation and deallocation of CO1 memory for storing matrix multiplication results, as it is managed internally by the Matmul API.
    - User-managed CO1: Users can flexibly and autonomously control the transfer of matrix multiplication results.
      This sample implements the second usage of the Iterate interface: caching multiple matrix multiplication results from Iterate calls in user-allocated CO1 memory, and transferring multiple baseM * baseN C matrix blocks at once when needed.

  - Kernel Key Steps
    - When creating a Matmul object, you must define the memory logical position of matrix C as TPosition::CO1 and the data layout format as CubeFormat::NZ.
        ```cpp

        AscendC::Matmul<AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>,
                        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>,
                        AscendC::MatmulType<AscendC::TPosition::CO1, CubeFormat::NZ, L0cT>,
                        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>, CFG_NORM>
            matmulObj;
        ```
    - Complete the matrix multiplication operation.
      - Call the user-managed CO1 interface to get the result of one Iterate computation.
        ```cpp
        matmulObj.Iterate(false, l0cTensor[l0cOffset]);
        ```
      - Call the Fixpipe interface to transfer the computation result matrix C from CO1.
        ```cpp
        FixpipeParamsV220 params;
        params.mSize = tiling.baseM;
        params.nSize = tiling.singleCoreN;
        params.srcStride = (params.mSize + BLOCK_CUBE - 1) / BLOCK_CUBE * BLOCK_CUBE;
        params.dstStride = tiling.N;
        CO1_.EnQue(l0cTensor);
        CO1_.template DeQue<L0cT>();
        Fixpipe<CType, L0cT, CFG_ROW_MAJOR>(cGlobal, l0cTensor, params);
        CO1_.FreeTensor(l0cTensor);
        CO1_.FreeAllEvent();
        ```

  - Invocation Implementation
    Use the kernel invocation operator <<<>>> to call the kernel function.

## Compilation and Execution

Execute the following steps in the sample root directory to compile and run the sample.

- Configure Environment Variables
  Select the appropriate environment variable configuration command based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package in your current environment.

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

- Sample Execution

  ```bash
  mkdir -p build && cd build;    # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py    # Generate test input data
  ./demo                        # Execute the compiled binary to run the sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin    # Verify output correctness
  ```

  For CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching compilation modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Compilation Options Description

  | Parameter | Description | Options | Default |
  |------|------|---------|--------|
  | CMAKE_ASC_RUN_MODE | Run mode | npu, cpu, sim | npu |
  | CMAKE_ASC_ARCHITECTURES | NPU hardware architecture | dav-2201 | dav-2201 |

- Execution Result

  The following output indicates successful precision comparison:
  ```bash
  test pass!
  ```