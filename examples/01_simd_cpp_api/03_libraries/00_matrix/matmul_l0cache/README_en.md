# Matmul L0 Cache Feature Direct Call Example
## Overview
This is a Matmul example that enables the L0 cache feature to reduce MTE1 redundant transfers. Taking the left matrix A as an example, L0 cache refers to the MTE1 transfer process when data is moved from A1 to A2. After data in A1 is transferred to A2, the cached data remains in A2, and the cached data sequentially performs multiply-accumulate operations with different data in B2.

L0 cache has no external switch and is automatically derived by the Matmul API internally based on the shape information configured by the user. Currently, only L0A cache is supported. The enabling requirements are as follows:
- All constant Tiling scenarios need to be configured, i.e., singleCoreM/singleCoreN/singleCoreK and baseM/baseN/baseK must all be set through constant interfaces;
- singleCoreM=baseM and singleCoreK=baseK.

## Supported Products
- Ascend 950PR/Ascend 950DT
- Atlas A3 training series products/Atlas A3 inference series products
- Atlas A2 training series products/Atlas A2 inference series products
## Directory Structure
```
├── matmul_l0cache
│   └── scripts
│       ├── gen_data.py         // Input data and golden data generation script
│       └── verify_result.py    // Golden data comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_l0cache.asc      // Ascend C example implementation & call example
```
## Example Description
- Example Function:
  The Matmul example calls the Matmul high-level API to perform matrix multiplication and bias addition on input A and B matrices, enabling L0 cache for the A matrix to reduce MTE1 redundant transfers and improve example performance.

- Example Specifications:
  In this example: M = 2560, N = 2048, K = 128.
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
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_l0cache_custom</td></tr>
  </table>
- Example Implementation:
  - Example Kernel Implementation
    - Specific Steps:
      - Create Matmul object.
        Set MatmulShapeParams to satisfy: singleCoreM=baseM, singleCoreK=baseK, call the GetMMConfig interface to obtain custom MatmulConfig. Call the GetMatmulApiTiling interface to obtain constant Matmul Tiling parameters.
          ```cpp
          constexpr static int32_t SINGLE_M = 256; // custom matmul kernel support max value of M Dim shape
          constexpr static int32_t SINGLE_N = 1024; // custom matmul kernel support max value of N Dim shape
          constexpr static int32_t SINGLE_K = 128; // custom matmul kernel support max value of K Dim shape
          constexpr static int32_t BASE_M = 256;  // BASEM * BASE_K * sizeof(typeC) <=L0A size
          constexpr static int32_t BASE_N = 128;  // BASEN * BASE_K * sizeof(typeB) <=L0B size
          constexpr static int32_t BASE_K = 128;   // BASEM * BASE_N * sizeof(typeC) <=L0C size
          constexpr static MatmulShapeParams shapeParams = { SINGLE_M, SINGLE_N, SINGLE_K, BASE_M, BASE_N, BASE_K };
          constexpr static MatmulConfig CUSTOM_CFG = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(shapeParams);
          constexpr static auto CONSTANT_CFG = AscendC::GetMatmulApiTiling<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(CUSTOM_CFG);
          AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CONSTANT_CFG> matmulObj;
          ```
      - Initialize operations.
      - Set left matrix A, right matrix B, and Bias.
      - Perform matrix multiplication operation.
      - End matrix multiplication operation.

  - Example Tiling Implementation
    - Ascend C provides a set of Matmul Tiling APIs to facilitate users in obtaining the Tiling parameters required for Matmul kernel computation. Simply pass in A/B/C matrix information and call the API interface to obtain the relevant parameters in the TCubeTiling structure.
    - The process of obtaining Tiling parameters is as follows:
      - Create a Tiling object.
      - Set parameter type information for A, B, C, Bias; M, N, Ka, Kb shape information, etc.
      - Call the GetTiling interface to obtain Tiling information.

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

  ```bash
  mkdir -p build && cd build;    # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py    # Generate test input data
  ./demo                        # Execute the compiled program to run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin    # Verify output correctness and confirm algorithm logic
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clean the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then re-run cmake.

- Build Option Description

  | Parameter | Description | Available Values | Default Value |
  |------|------|---------|--------|
  | CMAKE_ASC_RUN_MODE | Run mode | npu, cpu, sim | npu |
  | CMAKE_ASC_ARCHITECTURES | NPU hardware architecture | dav-2201, dav-3510 | dav-2201 |

- Execution Result

  The following execution result indicates successful precision comparison:
  ```bash
  test pass!
  ```