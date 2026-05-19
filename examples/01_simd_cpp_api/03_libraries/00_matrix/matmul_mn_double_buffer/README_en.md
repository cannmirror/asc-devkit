# Matmul M/N Axis Pipeline Parallelism Sample
## Overview
This is a Matmul sample demonstrating pipeline parallelism in the M/N axis direction. The application scenario for this feature is when the K dimension of the input matrix is small but M or N is large, that is, singleCoreK<=baseK, but singleCoreM is much larger than baseM or singleCoreN is much larger than baseN. Enabling M/N direction pipeline parallelism may improve performance.

## Supported Products
- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure
```
├── matmul_mn_double_buffer
│   └── scripts
│       ├── gen_data.py         // Script for generating input data and golden data
│       └── verify_result.py    // Golden data comparison file
│   ├── CMakeLists.txt          // Build configuration file
│   ├── data_utils.h            // Data read and write functions
│   └── matmul_mn_double_buffer.asc         // Ascend C sample implementation and invocation sample
```
## Sample Description
- Sample Function:  
  This sample calls the Matmul high-level API to perform matrix multiplication on input matrices A and B with bias addition. When defining the Matmul object, a MatmulConfig with parameters such as scheduleType is passed as a template parameter to enable M/N axis pipeline parallelism. This sample uses the MDL template with N-direction pipeline parallelism as an example. The implementation can also be referenced for the Norm template or M-direction pipeline parallelism.

- Sample Specifications:  
  In this sample: M = 128, N = 7680, K = 16
  <table>
  <tr><td rowspan="1" align="center">Sample Type(OpType)</td><td colspan="4" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">matmul_mn_double_buffer_custom</td></tr>
  </table>
- Sample Implementation
  - Key Kernel Steps
    - When creating the Matmul object, customize the MatmulConfig parameters by setting MatmulConfigMode to CONFIG_MDL, scheduleType parameter to ScheduleType::OUTER_PRODUCT, and iterateOrder parameter to IterateOrder::ORDER_M. This enables the N-direction pipeline parallelism feature of the MDL template, obtaining a customized Matmul object using the MDL template.
      ```cpp
      constexpr static MatmulConfigMode configModeMDL = MatmulConfigMode::CONFIG_MDL;
      constexpr static MatmulFuncParams funcParamsOrderM{false, false, false, false, 0, IterateOrder::ORDER_M, ScheduleType::OUTER_PRODUCT, true, true};
      constexpr static MatmulConfig CFG_MDL_OUTER_PRODUCT_ORDER_M = GetMMConfig<configModeMDL>(funcParamsOrderM);

      using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, ATYPE, false>;
      using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType, false>;
      using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>;
      using BIAS_TYPE =  AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>;
      AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL_OUTER_PRODUCT_ORDER_M> matmulObj;
      ```

  - Key Tiling Steps
    - Set custom MatmulConfig parameters and synchronize parameters configured on the Kernel side such as scheduleType to the Tiling side.
      ```cpp
      matmul_tiling::MatmulConfigParams matmulConfigParams(1, false, matmul_tiling::ScheduleType::OUTER_PRODUCT,
          matmul_tiling::MatrixTraverse::FIRSTM, false);
      cubeTiling.SetMatmulConfigParams(matmulConfigParams);
      ```

- Invocation Implementation  
  Use the kernel call operator `<<<>>>` to invoke the kernel function.

## Build and Run
Execute the following steps in the root directory of this sample to build and run the sample.
- Configure Environment Variables  
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on your current environment.
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
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build the project, default npu mode
  python3 ../scripts/gen_data.py    # Generate test input data
  ./demo                        # Execute the compiled executable to run the sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin    # Verify output results against golden data to confirm algorithm correctness
  ```

  For CPU debugging or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  For example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debugging mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean the cmake cache by executing `rm CMakeCache.txt` in the build directory and then re-run cmake.

- Build Options Description

  | Parameter | Description | Available Values | Default |
  |------|------|---------|--------|
  | CMAKE_ASC_RUN_MODE | Run mode | npu, cpu, sim | npu |
  | CMAKE_ASC_ARCHITECTURES | NPU hardware architecture | dav-2201, dav-3510 | dav-2201 |

- Execution Result

  The following output indicates successful accuracy comparison:
  ```bash
  test pass!
  ```