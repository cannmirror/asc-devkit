# Mx Matmul Direct Call Example

## Overview

Matrix multiplication with quantization coefficients in MXFP4/MXFP8 data format, known as the MxMatmul example. The computation formula is: C = (scaleA ⊗ A) * (scaleB ⊗ B) + Bias. "⊗" represents broadcast multiplication.  
When the value of K rounded up to the nearest multiple of 32 is odd, the value of scaleK needs to be aligned up to the next even number, and the scaleA and scaleB matrices need to be expanded accordingly.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── matmul_mx
│   └── scripts
│       ├── gen_data.py         // Input data and ground truth data generation script
│       └── verify_result.py    // Ground truth comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_mx.asc              // Ascend C example implementation & call example
```

## Example Description

- Example Function:  
  The MatmulMxNormEvenCustom example calls the Matmul API to compute the product of the left quantization coefficient matrix and the left matrix, and the product of the right quantization coefficient matrix and the right matrix, then perform matrix multiplication on the two products.

- Example Specifications:  
  In this example: M = 32, N = 128, K = 128, scaleK = 4. scaleK is the result of K divided by 32, which is 4.
  <table>
  <tr><td rowspan="1" align="center">Example Type(OpType)</td><td colspan="5" align="center">MatmulMxNormEvenCustom</td></tr>
  </tr>
  <tr><td rowspan="6" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">fp8_e5m2_t</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">scaleA</td><td align="center">[M, scaleK]</td><td align="center">fp8_e8m0_t</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">fp8_e5m2_t</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">scaleB</td><td align="center">[scaleK, N]</td><td align="center">fp8_e8m0_t</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_mx_custom</td></tr>
  </table>

- Example Implementation: 
  - Kernel Key Steps
    - Create a Matmul object: Use MatmulTypeWithScale to enable scaleA and scaleB.
      ```cpp
      typedef AscendC::MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::GM, CubeFormat::ND, fp8_e5m2_t, false> aType;
      typedef AscendC::MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::GM, CubeFormat::ND, fp8_e5m2_t, false> bType;
      typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> cType;
      typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType;
      // When defining the matmul object, pass MatmulWithScalePolicy to indicate enabling the MxMatmul template policy
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

  - Tiling Key Steps
      - Create a Tiling object: Use SetMadType to set the Matmul mode, enable the MxMatmul scenario, use SetScaleAType to set ScaleA information, and use SetScaleBType to set scaleB information.
        ```cpp
        cubeTiling.SetMadType(matmul_tiling::MatrixMadType::MXMODE);
        cubeTiling.SetScaleAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, false);
        cubeTiling.SetScaleBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, false);
        ```

  - Invocation Implementation  
    Use the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this example to build and run the example.
- Configure Environment Variables  
  Select the appropriate environment variable configuration command based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on your current environment.
  - Default path, CANN software package installed by root user
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```
  - Default path, CANN software package installed by non-root user
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```
  - Specified path install_path, CANN software package installed
    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Example Execution

  ```bash
  mkdir -p build && cd build;   # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;             # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                        # Execute the compiled executable program, run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output result correctness, confirm algorithm logic is correct
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  For example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching compilation modes, you need to clear the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

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