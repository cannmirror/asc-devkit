# Matmul FP8 Direct Call Example

## Overview

This is a Matmul example with A and B matrices using hifloat8, fp8_e4m3fn, and fp8_e5m2 data types as input.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── matmul_fp8
│   └── scripts
│       ├── gen_data.py         // Input data and golden data generation script
│       └── verify_result.py    // Golden data comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_fp8.asc          // Ascend C example implementation & call example
```

## Example Description

- Example Function:
  Performs matrix multiplication and bias addition on input A and B matrices, implementing a Matmul example with hifloat8, fp8_e4m3fn, and fp8_e5m2 data types as input. When the input data type is hifloat8, the A and B data types must be consistent.

- Example Specifications:
  In this example: M = 428, N = 479, K = 158.
  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="5" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">hifloat8, fp8_e4m3fn, fp8_e5m2</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">hifloat8, fp8_e4m3fn, fp8_e5m2</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_fp8_custom</td></tr>
  </table>
- Example Implementation:
  - Kernel Key Steps
    - Create a Matmul object with data type set to various fp8 data types according to DT_MODE.
      ```cpp
      #if DT_MODE == 1
      MatmulFp8Kernel<fp8_e4m3fn_t, fp8_e4m3fn_t, float, float> MatmulFp8Kernel;
      #elif DT_MODE == 2
      MatmulFp8Kernel<fp8_e5m2_t, fp8_e5m2_t, float, float> MatmulFp8Kernel;
      #elif DT_MODE == 3
      MatmulFp8Kernel<fp8_e4m3fn_t, fp8_e5m2_t, float, float> MatmulFp8Kernel;
      #elif DT_MODE == 4
      MatmulFp8Kernel<fp8_e5m2_t, fp8_e4m3fn_t, float, float> MatmulFp8Kernel;
      #else
      MatmulFp8Kernel<hifloat8_t, hifloat8_t, float, float> MatmulFp8Kernel;
      #endif

      ...

      AscendC::Matmul<AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cType>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>, mmCfg> matmulObj;
      ```
  - Call Implementation
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

  - DT_MODE: Set the data type combination for A and B matrices
    - 0: Corresponds to A hifloat8, B hifloat8 scenario
    - 1: Corresponds to A fp8_e4m3fn, B fp8_e4m3fn scenario
    - 2: Corresponds to A fp8_e5m2, B fp8_e5m2 scenario
    - 3: Corresponds to A fp8_e4m3fn, B fp8_e5m2 scenario
    - 4: Corresponds to A fp8_e5m2, B fp8_e4m3fn scenario

  ```bash
  mkdir -p build && cd build;        # Create and enter build directory
  cmake -DDT_MODE=0 -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;      # Build project, 0 can be 01234, e.g., cmake .. -DDT_MODE=1;make -j;
  python3 ../scripts/gen_data.py  0  # Generate test input data, 0 can be 01234, e.g., python3 ../scripts/gen_data.py  1
  ./demo                             # Execute the compiled program to run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output correctness and confirm algorithm logic
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake -DDT_MODE=0 -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  cmake -DDT_MODE=0 -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clean the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then re-run cmake.

- Build Option Description

  | Option | Available Values | Description |
  | ----------------| -----------------------------| --------------------------------------------------------------------------------------|
  | `DT_MODE` | `0`, `1`, `2`, `3`, `4` | Scenario number, see example description for details |
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201`, `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 training series products/Atlas A2 inference series products and Atlas A3 training series products/Atlas A3 inference series products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  The following execution result indicates successful precision comparison:
  ```bash
  test pass!
  ```