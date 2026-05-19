# Matmul A2 and B2 Global Management Direct Invocation Sample

## Overview

A Matmul sample with A2 (L0A Buffer) and B2 (L0B Buffer) global management enabled. In this scenario, all Matmul objects share A2 (L0A Buffer) and B2 (L0B Buffer).

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── matmul_a2b2_share
│   └── scripts
│       ├── gen_data.py         // Input data and ground truth data generation script
│       └── verify_result.py    // Ground truth comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_a2b2_share.asc   // Ascend C sample implementation & invocation sample
```

## Sample Description

- Sample Function:
  The Matmul sample contains two Matmul objects, each with different left and right matrices but the same bias.

  By setting the isA2B2Shared parameter in MatmulConfig to true for each Matmul object, A2 and B2 global management is enabled, meaning both Matmul objects share A2 and B2.

- Sample Specifications:
  In this sample: M = 7680, N = 480, K = 320.
  <table>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="5" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="6" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a1</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b1</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">a2</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b2</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="2" align="center">Sample Output</td>
  <td align="center">c1</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  <td align="center">c2</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_a2b2_share_custom</td></tr>
  </table>

- Sample Implementation:
  - Kernel Key Steps
    - Create a MatmulConfig object, configure the NORM template, set isA2B2Shared parameter to true, and create two Matmul objects.
      ```cpp
      // In the first matmul calculation, `a1 * b1 + bias = c1`.
      AscendC::Matmul<AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType, IS_TRANS_A>,
                      AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType, IS_TRANS_B>,
                      AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>,
                      AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>, CFG_MDL>
          matmulObj1;
      // In the second matmul calculation, `a2 * b2 + bias = c2`.
      AscendC::Matmul<AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType, IS_TRANS_A>,
                      AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType, IS_TRANS_B>,
                      AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>,
                      AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>, CFG_MDL>
          matmulObj2;
      ```
    - Set the left matrices A1, A2 and right matrices B1, B2 for two matrix multiplications, sharing the same Bias.
      ```cpp
      matmulObj1.SetTensorA(a1Global);
      matmulObj1.SetTensorB(b1Global);
      matmulObj2.SetTensorA(a2Global);
      matmulObj2.SetTensorB(b2Global);
      if (tiling.isBias) {
          matmulObj1.SetBias(biasGlobal);
          matmulObj2.SetBias(biasGlobal);
      }
      matmulObj1.IterateAll(c1Global);
      matmulObj1.End();
      matmulObj2.IterateAll(c2Global);
      matmulObj2.End();
      ```

  - Invocation Implementation
    Use the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this sample to build and run the sample.

- Configure Environment Variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development toolkit on your current environment.
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

- Run the Sample

  ```bash
  mkdir -p build && cd build;   # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;             # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                        # Execute the compiled executable to run the sample
  python3 ../scripts/verify_result.py output/output1.bin output/golden1.bin output/output2.bin output/golden2.bin  # Verify output correctness and confirm algorithm logic
  ```

  For CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then run cmake again.

- Build Options Description

  | Parameter | Description | Options | Default |
  |-----------|-------------|---------|---------|
  | CMAKE_ASC_RUN_MODE | Run mode | npu, cpu, sim | npu |
  | CMAKE_ASC_ARCHITECTURES | NPU hardware architecture | dav-2201, dav-3510 | dav-2201 |

- Execution Result

  The following output indicates successful accuracy verification:
  ```bash
  test pass!
  ```