# Matmul ND_ALIGN Format Output Direct Invocation Sample

## Overview

This sample demonstrates Matmul with N-direction alignment enabled for output in scenarios where the input matrix N direction is not aligned. By setting the matrix multiplication result matrix C to ND_ALIGN format output, the Matmul API outputs matrix C according to the 32-byte alignment padding rule in the N direction.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── matmul_format_nd_align
│   └── scripts
│       ├── gen_data.py         // Input data and golden data generation script
│       └── verify_result.py    // Golden value comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_format_nd_align.asc              // Ascend C sample implementation & invocation sample
```

## Sample Description

- **Sample Function:**
  When the Matmul sample calls the Matmul API for computation, it enables the N-direction 32-byte alignment functionality for matrix multiplication output by setting the Format parameter of matrix C to CubeFormat::ND_ALIGN, performing matrix multiplication and adding bias offset on input matrices A and B.

- **Sample Specifications:**
  In this sample: M = 128, N = 7679, K = 128.

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
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_format_nd_align</td></tr>
  </table>

- **Sample Implementation:**
  - Kernel Key Steps
    - Create a Matmul object: Enable ND_ALIGN for matrix C.
        ```cpp
        AscendC::Matmul<
          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, ATYPE>,
          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>,
          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND_ALIGN, CType>,
          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>> matmulObj;
        ```

  - Tiling Key Steps
    - Create a Tiling object: Enable ND_ALIGN for matrix C.
      ```cpp
      cubeTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN,
      matmul_tiling::DataType::DT_FLOAT);
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
  | CMAKE_ASC_ARCHITECTURES | NPU hardware architecture | dav-2201, dav-3510 | dav-2201 |

- Execution Result

  The following output indicates successful precision comparison:
  ```bash
  test pass!
  ```