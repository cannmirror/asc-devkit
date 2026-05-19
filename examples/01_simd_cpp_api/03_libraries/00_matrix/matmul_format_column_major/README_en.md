# Matmul ColumnMajor Format Direct Invocation Sample

## Overview

This sample demonstrates Matmul with input and output matrices in COLUMN_MAJOR (column-major) format. Unlike matrix multiplication with ND (row-major) format, for matrices in COLUMN_MAJOR (column-major) format, the Matmul API supports setting matrices to COLUMN_MAJOR format.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── matmul_format_column_major
│   └── scripts
│       ├── gen_data.py             // Input data and golden data generation script
│       └── verify_result.py        // Golden value comparison file
│   ├── CMakeLists.txt              // Build project file
│   ├── data_utils.h                // Data read/write functions
│   └── matmul_format_column_major.asc     // Ascend C sample implementation & invocation sample
```

## Sample Description

- **Sample Function:**
  The MatmulColumnMajorCustom sample calls the Matmul API for computation and sets the Format parameter of matrices A, B, and C (where elements in the column direction are contiguous in memory) to CubeFormat::COLUMN_MAJOR, implementing column-major matrix multiplication. The sample performs matrix multiplication and adds bias offset on input matrices A and B.

- **Sample Specifications:**
    In this sample: M = 428, N = 479, K = 528.

    <table>
    <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="5" align="center">MatmulColumnMajor</td></tr>
    </tr>
    <tr><td rowspan="4" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
    <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">COLUMN_MAJOR</td><td align="center">false</td></tr>
    <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">COLUMN_MAJOR</td><td align="center">false</td></tr>
    <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
    </tr>
    </tr>
    <tr><td rowspan="1" align="center">Sample Output</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">COLUMN_MAJOR</td><td align="center">-</td></tr>
    </tr>
    <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmulColumnMajorCustom</td></tr>
    </table>

- **Sample Implementation:**
  - Kernel Key Steps
    - Create a Matmul object: Set the Format of matrix C to COLUMN_MAJOR.
      ```cpp
      AscendC::Matmul<
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::COLUMN_MAJOR, ATYPE>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::COLUMN_MAJOR, BType>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::COLUMN_MAJOR, CType>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>> matmulObj;
      ```

  - Tiling Key Steps
    - Set the parameter type information for A, B, C, and Bias, where the Format of matrices A, B, and C is set to COLUMN_MAJOR.
      ```cpp
      cubeTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::COLUMN_MAJOR,
          matmul_tiling::DataType::DT_FLOAT16, isAtrans);
      cubeTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::COLUMN_MAJOR,
          matmul_tiling::DataType::DT_FLOAT16, isBtrans);
      cubeTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::COLUMN_MAJOR,
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
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py    # Generate test input data
  ./demo                        # Execute the compiled binary to run the sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin    # Verify output correctness
  ```

  For CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching compilation modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Compilation Options Description

  | Parameter | Description | Options | Default |
  |------|------|---------|--------|
  | CMAKE_ASC_RUN_MODE | Run mode | npu, cpu, sim | npu |
  | CMAKE_ASC_ARCHITECTURES | NPU hardware architecture | dav-3510 | dav-3510 |

- Execution Result

  The following output indicates successful precision comparison:
  ```bash
  test pass!
  ```