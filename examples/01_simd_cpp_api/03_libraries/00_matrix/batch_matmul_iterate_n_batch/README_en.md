# Batch Matmul IterateNBatch Direct Invocation Sample

## Overview

A sample for processing Matmul computations in multiple batches, including implementations for both synchronous and asynchronous scenarios.

## Supported Products

- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── batch_matmul_iterate_n_batch
│   └── scripts
│       ├── gen_data.py              // Input data and ground truth data generation script
│       └── verify_result.py         // Ground truth comparison file
│   ├── CMakeLists.txt               // Build project file
│   ├── data_utils.h                 // Data read/write functions
│   └── batch_matmul_iterate_n_batch.asc   // Ascend C sample implementation & invocation sample
```

## Sample Description

- Sample Function:
  This sample splits the BatchMatmul computation batch into nNum main batches, where each main batch includes BatchNum sub-batches of Matmul computations. By modifying the IS_SYNCH parameter in the batch_matmul_iterate_n_batch.asc code, you can control whether to use the asynchronous computation flow.

- Constraints
  - Single batch Matmul computation must follow the constraints of IterateBatch;
  - For BSNGD, SBNGD, BNGS1S2 Layout formats, the total multi-batch data of input A and B matrices should be less than the L1 Buffer size;

- Sample Specifications:
    In this sample: BatchNum = 3, M = 32, N = 256, K = 64.
    <table>
    <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="6" align="center">BatchMatmulCustom</td></tr>
    </tr>
    <tr><td rowspan="4" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td><td align="center">layout</td></tr>
    <tr><td align="center">a</td><td align="center">[BatchNum, M, K]</td><td align="center">half</td><td align="center">ND</td><td align="center">true</td><td align="center">BSNGD</td></tr>
    <tr><td align="center">b</td><td align="center">[BatchNum, K, N]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td><td align="center">BSNGD</td></tr>
    <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td><td align="center">-</td></tr>
    </tr>
    </tr>
    <tr><td rowspan="1" align="center">Sample Output</td><td align="center">c</td><td align="center">[BatchNum, M, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td><td align="center">BSNGD</td></tr>
    </tr>
    <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="6" align="center">batch_matmul_iterate_n_batch_custom</td></tr>
    </table>

- Sample Implementation:
  - Kernel Key Steps
    - When creating the Matmul object, customize the MatmulConfig parameters by setting the isNBatch parameter to true to enable multi-batch input and multi-batch output; setting the isBiasBatch parameter to false to enable the Bias reuse feature for BatchMatmul, obtaining a customized Matmul object using the NORM template.
      ```cpp
      constexpr MatmulConfigMode configMode = MatmulConfigMode::CONFIG_NORM;
      constexpr MatmulBatchParams batchParams = {
        true, BatchMode::BATCH_LESS_THAN_L1, false /* isNBatch, batchMode, isBiasBatch */
      };
      constexpr MatmulConfig CFG_MM = GetMMConfig<configMode>(batchParams);
      AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_PARTIAL> matmulObj;
      ```
    - Calculate nNum size based on the Layout information of A and B matrices and BatchNum.
      ```cpp
      int32_t g_lay = tiling.ALayoutInfoG > tiling.BLayoutInfoG ? tiling.ALayoutInfoG : tiling.BLayoutInfoG;
      int32_t nNum = tiling.ALayoutInfoB * tiling.ALayoutInfoN * g_lay / tiling.BatchNum;
      ```
    - Set workspace.
      ```cpp
        matmulObj.SetWorkspace(cGlobal);
      ```
    - Complete the multi-batch matrix multiplication operation and call the GetBatchTensorC interface to get the result.
      ```cpp
      matmulObj.template IterateNBatch<false>(nNum, batchA, batchB, false);
      for(int32_t j = 0; j < nNum; ++j){
          matmulObj.template GetBatchTensorC<false>(batchA, batchB, false);
      }
      ```

  - Tiling Key Steps
    - Call SetALayout, SetBLayout, SetCLayout, and SetBatchNum to set the Layout axis information and maximum BatchNum for A/B/C.
      ```cpp
      cubeTiling.SetALayout(1, M, 1, batchNum, K);
      cubeTiling.SetBLayout(1, N, 1, batchNum, K);
      cubeTiling.SetCLayout(1, M, 1, batchNum, N);
      cubeTiling.SetBatchNum(batchNum);
      ```

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
  mkdir -p build && cd build;    # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py    # Generate test input data
  ./demo                        # Execute the compiled executable to run the sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin    # Verify output correctness and confirm algorithm logic
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