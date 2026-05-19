# Mx Matmul NZ Input Direct Call Sample

## Overview

This sample demonstrates MxMatmul with user-defined TSCM and VECOUT inputs using MXFP4/MXFP8 data formats.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── matmul_mx_ub_tscm_nz
│   └── scripts
│       ├── gen_data.py         // Input data and golden data generation script
│       └── verify_result.py    // Golden value comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_mx_ub_tscm_nz.asc              // Ascend C sample implementation & call sample
```

## Sample Description

- Sample Function:
  The MatmulMxUbTscmNzCustom sample calls the Matmul API for computation. The memory logical positions for matrices A and B use VECOUT, while the memory logical positions for scaleA and scaleB matrices use TSCM. All four input matrices are in NZ format. The left quantization coefficient matrix multiplies with the left matrix, and the right quantization coefficient matrix multiplies with the right matrix. The matrix multiplication is performed on the results of these two products.

- Sample Specifications:
  In this sample: M = 64, N = 128, K = 128, scaleK = 4. Where scaleK is the result of K divided by 32, which is 4.
  <table>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="5" align="center">MatmulMxUbTscmNzCustom</td></tr>
  </tr>
  <tr><td rowspan="6" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">fp4x2_e1m2_t</td><td align="center">NZ</td><td align="center">false</td></tr>
  <tr><td align="center">scaleA</td><td align="center">[M, scaleK]</td><td align="center">fp8_e8m0_t</td><td align="center">NZ</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">fp4x2_e1m2_t</td><td align="center">NZ</td><td align="center">false</td></tr>
  <tr><td align="center">scaleB</td><td align="center">[scaleK, N]</td><td align="center">fp8_e8m0_t</td><td align="center">NZ</td><td align="center">true</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_mx_ub_tscm_nz_custom</td></tr>
  </table>

- Sample Implementation:
  - Kernel Key Steps
    - Create a Matmul object: Use MatmulTypeWithScale to enable scaleA and scaleB. Set the memory logical positions for the left and right matrices to VECOUT, and the memory logical positions for the left and right quantization coefficient matrices to TSCM. The physical layout format of all input matrix data is NZ, and set the SCALE_ISTRANS parameter of scaleB to true.
      ```cpp
      using aType = AscendC::MatmulTypeWithScale<AscendC::TPosition::VECOUT, AscendC::TPosition::TSCM, CubeFormat::NZ, fp4x2_e1m2_t, false, AscendC::TPosition::GM, CubeFormat::NZ, false, AscendC::TPosition::GM>;
      using bType = AscendC::MatmulTypeWithScale<AscendC::TPosition::VECOUT, AscendC::TPosition::TSCM, CubeFormat::NZ, fp4x2_e1m2_t, false, AscendC::TPosition::GM, CubeFormat::NZ, true, AscendC::TPosition::GM>;
      using cType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
      using biasType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
      // When defining the matmul object, pass MatmulWithScalePolicy to indicate enabling the MxMatmul template policy
      AscendC::Matmul<aType, bType, cType, biasType, CFG_MDL, AscendC::MatmulCallBackFunc<nullptr, nullptr, nullptr>, AscendC::Impl::Detail::MatmulWithScalePolicy> matmulObj;
      ```
    - Set the left matrix A and left quantization coefficient matrix scaleA, right matrix B and right quantization coefficient matrix scaleB, and Bias.
      ```cpp
      // SetTensorA
      pipe->InitBuffer(leftMatrixQue, 1, tiling.singleCoreM * tiling.singleCoreK);
      bufferLeft = leftMatrixQue.AllocTensor<fp4x2_e1m2_t>();
      DataCopy(bufferLeft, aGlobal, tiling.singleCoreM * tiling.singleCoreK);
      AscendC::PipeBarrier<PIPE_ALL>();
      matmulObj.SetTensorA(bufferLeft, isTransA);
      
      // SetTensorB
      pipe->InitBuffer(rightMatrixQue, 1, tiling.singleCoreK * tiling.singleCoreN);
      bufferRight = rightMatrixQue.AllocTensor<fp4x2_e1m2_t>();
      DataCopy(bufferRight, bGlobal, tiling.singleCoreK * tiling.singleCoreN);
      AscendC::PipeBarrier<PIPE_ALL>();
      matmulObj.SetTensorB(bufferRight, isTransB);
      
      // SetTensorScaleA
      pipe->InitBuffer(qidMxA1, 1, alignSingleCoreM * alignSingleCoreK / 32);
      bufferLeftScale = qidMxA1.AllocTensor<fp8_e8m0_t>();
      DataCopy(bufferLeftScale, asGlobal, tiling.singleCoreM * tiling.singleCoreK / 32);
      AscendC::PipeBarrier<PIPE_ALL>();
      matmulObj.SetTensorScaleA(bufferLeftScale, isTransScaleA);

      // SetTensorScaleB
      pipe->InitBuffer(qidMxB1, 1, alignSingleCoreN * alignSingleCoreK / 32);
      bufferRightScale = qidMxB1.AllocTensor<fp8_e8m0_t>();
      DataCopy(bufferRightScale, bsGlobal, tiling.singleCoreK * tiling.singleCoreN / 32);
      AscendC::PipeBarrier<PIPE_ALL>();
      matmulObj.SetTensorScaleB(bufferRightScale, isTransScaleB);

      if (tiling.isBias) {
          matmulObj.SetBias(biasGlobal);
      }
      ```

  - Tiling Key Steps
    - Create a Tiling object: Use SetMadType to enable Mx features, use SetScaleAType to set scaleA information, and use SetScaleBType to set scaleB information.
      ```cpp
      cubeTiling.SetAType(matmul_tiling::TPosition::VECOUT, matmul_tiling::CubeFormat::NZ,
          matmul_tiling::DataType::DT_FLOAT8_E5M2, isAtrans);
      cubeTiling.SetBType(matmul_tiling::TPosition::VECOUT, matmul_tiling::CubeFormat::NZ,
          matmul_tiling::DataType::DT_FLOAT8_E5M2, isBtrans);
      cubeTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
          matmul_tiling::DataType::DT_FLOAT);
      cubeTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
          matmul_tiling::DataType::DT_FLOAT);
      cubeTiling.SetScaleAType(matmul_tiling::TPosition::TSCM, matmul_tiling::CubeFormat::NZ, isScaleATrans);
      cubeTiling.SetScaleBType(matmul_tiling::TPosition::TSCM, matmul_tiling::CubeFormat::NZ, isScaleBTrans);
      cubeTiling.SetMadType(matmul_tiling::MatrixMadType::MXMODE);
      ```

  - Call Implementation
    Use the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this sample to build and run the sample.

- Configure Environment Variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on your current environment.
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
  mkdir -p build && cd build;   # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;             # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                        # Execute the compiled executable program to run the sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output result correctness, confirm algorithm logic is correct
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  For example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clear the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build Options Description

  | Parameter | Description | Options | Default |
  |------|------|---------|--------|
  | CMAKE_ASC_RUN_MODE | Run mode | npu, cpu, sim | npu |
  | CMAKE_ASC_ARCHITECTURES | NPU hardware architecture | dav-3510 | dav-3510 |

- Execution Result

  The execution result is as follows, indicating successful precision comparison:
  ```bash
  test pass!
  ```