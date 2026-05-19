# Matmul Constant Tiling Sample

## Overview

This sample demonstrates Matmul with constant tiling. Constant tiling refers to converting some or all tiling parameters from variables to constant values during compilation, and the constant tiling parameters will be used during sample execution.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── matmul_constant_tiling
│   └── scripts
│       ├── gen_data.py         // Input data and golden data generation script
│       └── verify_result.py    // Golden value comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_constant_tiling.asc     // Ascend C sample implementation & invocation sample
```

## Sample Description

- **Sample Function:**
  The MatmulConstantCustom sample performs matrix multiplication and adds bias offset on input matrices A and B, while using constant tiling for static compilation on the Kernel side. Tiling information is derived before sample execution. By implementing Matmul constant tiling, scalar operations in the sample are reduced, improving sample performance. In constant tiling scenarios, the SingleShape set using the SetSingleShape interface on the Kernel side is the maximum shape for single-core computation at runtime, and the actual computation shape should be less than or equal to this shape.

- **Sample Specifications:**
  In this sample: M = 128, N = 30720, K = 64.

  <table>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="4" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">matmul_constant_tiling_custom</td></tr>
  </table>

- **Sample Implementation:**

  - Kernel Key Steps
    - Specific steps:
      - Configure constant MatmulShapeParams to obtain a custom MatmulConfig.
        ```cpp
        constexpr int32_t MAX_M = 10000; // custom matmul kernel support max value of M Dim shape
        constexpr int32_t MAX_N = 10000; // custom matmul kernel support max value of N Dim shape
        constexpr int32_t MAX_K = 10000; // custom matmul kernel support max value of K Dim shape
        constexpr int32_t BASE_M = 128;  // BASEM * BASE_K * sizeof(typeC) <=L0A size
        constexpr int32_t BASE_N = 256;  // BASEN * BASE_K * sizeof(typeB) <=L0B size
        constexpr int32_t BASE_K = 64;   // BASEM * BASE_N * sizeof(typeC) <=L0C size
        constexpr MatmulShapeParams shapeParams = { MAX_M,
                                                      MAX_N,
                                                      MAX_K,
                                                      BASE_M,
                                                      BASE_N,
                                                      BASE_K };
        constexpr MatmulConfig CUSTOM_CFG = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(shapeParams);
        ```
      - Obtain constant tiling information through the GetMatmulApiTiling interface.
        ```cpp
        auto constantCFG = AscendC::GetMatmulApiTiling<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmCFG);
        ```
      - Create a Matmul object using the custom MatmulConfig template.
        ```cpp
        using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>;
        using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>;
        using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cType>;
        using BIAS_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>;
        constexpr static auto CONSTANT_CFG = GetCustomConstantCFG<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>();
        AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CONSTANT_CFG> matmulObj;
        ```
      - Initialize operations and pass constant tiling information.
      - Set the left matrix A, right matrix B, and Bias.
        ```cpp
        matmulObj.SetTail(tailM, tailN, shapes.k);
        matmulObj.SetTensorA(aGlobal, false);
        matmulObj.SetTensorB(bGlobal, false);
        if (shapes.isBias) {
            matmulObj.SetBias(biasGlobal);
        }
        ```
      - Complete the matrix multiplication operation.
        ```cpp
        matmulObj.IterateAll(cGlobal);
        ```
      - End the matrix multiplication operation.
        ```cpp
        matmulObj.End();
        ```

  - Tiling Key Steps
      - Ascend C provides a set of Matmul Tiling APIs to facilitate users in obtaining the tiling parameters required for Matmul kernel computation. By simply passing information such as A/B/C matrices and calling the API interface, users can obtain relevant parameters in the TCubeTiling structure. For constant tiling, only the multi-core tiling operation needs to be implemented in the Tiling phase. Users can obtain the optimal multi-core tiling strategy through the multi-core tiling interface. Other tiling information is derived through constant tiling on the Kernel side, so the Kernel side no longer needs runtime tiling information.

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
  The following output indicates successful precision comparison.
  ```bash
  test pass!
  ```