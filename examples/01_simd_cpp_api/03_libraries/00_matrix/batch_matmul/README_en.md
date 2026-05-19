# Batch Matmul Direct Invocation Sample

## Overview

A sample for batch processing Matmul computations.

By transferring multiple Matmul input data in a single transfer, the number of transfers is reduced and performance is improved. This is applicable to scenarios where multiple Matmul computations are required and the input shape of a single Matmul computation is small, where transfer overhead accounts for a significant portion of the total time.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── batch_matmul
│   └── scripts
│       ├── gen_data.py              // Input data and ground truth data generation script
│       └── verify_result.py         // Ground truth comparison file
│   ├── CMakeLists.txt               // Build project file
│   ├── data_utils.h                 // Data read/write functions
│   └── batch_matmul.asc             // Ascend C sample implementation & invocation sample
```

## Sample Description

- Sample Function:
  Call the Matmul high-level API to implement batch processing of 3 groups of Matmul computations, where each group performs matrix multiplication and bias addition on A and B matrices in BSNGD format.

  For details on BSNGD format, refer to the data layout description in [IterateBatch](../../../../../docs/api/context/IterateBatch.md).

- Sample Specifications:
  <table>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="6" align="center">BatchMatmulCustom</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td><td align="center">layout</td></tr>
  <tr><td align="center">a</td><td align="center">[2, 32, 1, 3, 64]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td><td align="center">BSNGD</td></tr>
  <tr><td align="center">b</td><td align="center">[2, 256, 1, 3, 64]</td><td align="center">half</td><td align="center">ND</td><td align="center">true</td><td align="center">BSNGD</td></tr>
  <tr><td align="center">bias</td><td align="center">[2, 1, 1, 3, 256]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">c</td><td align="center">[2, 32, 1, 3, 256]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td><td align="center">BSNGD</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="6" align="center">batch_matmul_custom</td></tr>
  </table>

- Sample Implementation:
  - Kernel Key Steps
    - Complete the multi-batch matrix multiplication operation.
      ```cpp
      matmulObj.IterateBatch(cGlobal[batchOffsetC], batchA, batchB, false);
      ```

  - Tiling Key Steps
    - Call SetALayout, SetBLayout, SetCLayout, and SetBatchNum to set the Layout axis information and maximum BatchNum for A/B/C.
      ```cpp
      cubeTiling->SetALayout(A_BNUM, A_SNUM, 1, A_GNUM, A_DNUM);
      cubeTiling->SetBLayout(B_BNUM, B_SNUM, 1, B_GNUM, B_DNUM);
      cubeTiling->SetCLayout(C_BNUM, C_SNUM, 1, C_GNUM, C_DNUM);
      cubeTiling->SetBatchNum(BATCH_NUM);
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
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output correctness and confirm algorithm logic
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