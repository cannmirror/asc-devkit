# Mx Matmul Scale Multi-Buffer Direct Invocation Sample

## Overview

This is an MxMatmul sample that enables multi-buffer caching for the quantization coefficient matrix scale data on the L1 Buffer in MXFP4/MXFP8 data format. Enabling scale multi-buffer caching reduces MTE2 redundant copying, thereby improving performance.

Taking the left quantization coefficient matrix scaleA with K-direction multi-buffer caching and a multiplier of scaleFactorK as an example: During the MTE2 transfer process when data is moved from GM to A1 (L1 Buffer), the A matrix transfers stepM * stepK base blocks at once, while the scaleA matrix transfers stepM * (scaleFactorK * stepK) base blocks at once and caches them in A1. During subsequent Iterate calculations, the cached scaleA matrix data sequentially performs broadcast multiplication operations with different data in the A matrix.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure
```
├── matmul_mx_scale_cache
│   └── scripts
│       ├── gen_data.py         // Script for generating input data and golden data
│       └── verify_result.py    // Golden data comparison file
│   ├── CMakeLists.txt          // Build configuration file
│   ├── data_utils.h            // Data read and write functions
│   └── matmul_mx_scale_cache.asc  // Ascend C sample implementation and invocation sample
```

## Sample Description

- Sample Function:  
  The mxTypePara parameter in Matmul Tiling represents scale multi-buffer caching. Where:  
  - mxTypePara[0:7] represents scaleFactorKa, the ratio coefficient of scaleA to A matrix in K-direction data loading.
  - mxTypePara[8:15] represents scaleFactorKb, the ratio coefficient of scaleB to B matrix in K-direction data loading.
  - mxTypePara[16:23] represents scaleFactorM, the ratio coefficient of scaleA to A matrix in M-direction data loading.
  - mxTypePara[24:31] represents scaleFactorN, the ratio coefficient of scaleA to A matrix in N-direction data loading.
  
  For example, if tilingData.mxTypePara = 0x01010104, then scaleFactorKa = 4, indicating that scaleA has 4x caching enabled in the K direction.

- Constraints
  - For the scaleA matrix, M-direction multi-buffer caching is only allowed when the K-direction of scaleA is fully loaded on L1.
  - For the scaleB matrix, N-direction multi-buffer caching is only allowed when the K-direction of scaleB is fully loaded on L1.

- Sample Specifications:  
  In this sample: M = 32, N = 128, K = 128, scaleK = 4. Where scaleK is the result of K divided by 32, which is 4.
  <table>
  <tr><td rowspan="1" align="center">Sample Type(OpType)</td><td colspan="5" align="center">MatmulMxTypeParaCustom</td></tr>
  </tr>
  <tr><td rowspan="6" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">fp8_e5m2_t</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">scaleA</td><td align="center">[M, scaleK]</td><td align="center">fp8_e8m0_t</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">fp8_e5m2_t</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">scaleB</td><td align="center">[scaleK, N]</td><td align="center">fp8_e8m0_t</td><td align="center">ND</td><td align="center">true</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_mx_scale_cache_custom</td></tr>
  </table>
- Sample Implementation: 
  - Key Kernel Steps
      - Create Matmul object: Use MatmulTypeWithScale to define parameter type information for A, scaleA, B, scaleB, including: memory logical position, data format, data type, and transpose information.
        ```cpp
        typedef AscendC::MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::GM, CubeFormat::ND, fp8_e5m2_t, false, AscendC::TPosition::GM, CubeFormat::ND, false> aType;
        typedef AscendC::MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::GM, CubeFormat::ND, fp8_e5m2_t, false, AscendC::TPosition::GM, CubeFormat::ND, true> bType;
        typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> cType;
        typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType;
        // When defining the matmul object, pass MatmulWithScalePolicy to enable the MxMatmul template policy
        AscendC::Matmul<aType, bType, cType, biasType, CFG_MDL, AscendC::MatmulCallBackFunc<nullptr, nullptr, nullptr>, AscendC::Impl::Detail::MatmulWithScalePolicy> matmulObj;
        ```
      - Set the left matrix A and left quantization coefficient matrix scaleA, right matrix B and right quantization coefficient matrix scaleB, and Bias.
        ```cpp
        matmulObj.SetTensorA(aGlobal, isTransA);
        matmulObj.SetTensorB(bGlobal, isTransB);
        matmulObj.SetTensorScaleA(asGlobal, isTransScaleA);
        matmulObj.SetTensorScaleB(bsGlobal, isTransScaleB);

        if (tiling.isBias) {
            matmulObj.SetBias(biasGlobal);
        }
        ```

  - Key Tiling Steps
    - Create a Tiling object: Use SetMadType to enable Mx feature, use SetScaleAType to set ScaleA information, and use SetScaleBType to set scaleB information.
      ```cpp
      cubeTiling.SetMadType(matmul_tiling::MatrixMadType::MXMODE);
      cubeTiling.SetScaleAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, false);
      cubeTiling.SetScaleBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, true);
      if (cubeTiling.GetTiling(tilingData) == -1) {
          std::cout << "Generate tiling failed." << std::endl;
          return {};
      }
      // 0-6bit represents the ratio coefficient of scaleA to A matrix in K-direction data loading, 8-14bit represents the ratio coefficient of scaleB to B matrix in K-direction data loading. 260 represents binary 0000 0001 0000 0100, where 100 is used in 0-6bit, enabling 4x caching.
      tilingData.mxTypePara = 16843012;
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
  mkdir -p build && cd build;   # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;             # Build the project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
 1000                        # Execute the compiled executable to run the sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output results against golden data to confirm algorithm correctness
  ```

  For CPU debugging or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  For example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debugging mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean the cmake cache by executing `rm CMakeCache.txt` in the build directory and then re-run cmake.

- Build Options Description

  | Parameter | Description | Available Values | Default |
  |------|------|---------|--------|
  | CMAKE_ASC_RUN_MODE | Run mode | npu, cpu, sim | npu |
  | CMAKE_ASC_ARCHITECTURES | NPU hardware architecture | dav-3510 | dav-3510 |

- Execution Result

  The following output indicates successful accuracy comparison:
  ```bash
  test pass!
  ```