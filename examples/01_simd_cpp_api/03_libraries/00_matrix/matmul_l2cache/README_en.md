# Matmul L2 Cache Splitting Direct Invocation Sample

## Overview

This sample demonstrates the Matmul with L2 Cache splitting enabled to improve L2 Cache utilization.

The Matmul sample splits the input matrix along the M or N direction, dividing the matrix into multiple blocks. The computation is performed multiple times based on the number of split blocks. Before each computation, when the first core accesses the matrix in Global Memory for the first time, it loads a split block of matrix data into L2 Cache. Subsequent accesses by other cores or the first core can then hit the L2 Cache, improving sample performance.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── matmul_l2cache
│   └── scripts
│       ├── gen_data.py         // Input data and golden data generation script
│       └── verify_result.py    // Golden data verification file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read and write functions
|   ├── l2_cache_optimizer.h    // L2Cache splitting algorithm implementation functions
│   └── matmul_l2cache.asc      // Ascend C sample implementation & invocation sample
```

## Sample Description

- Sample Function:
  This sample uses the AI processor's L2 Cache size of 192M as an example. Based on the sample's input and output shapes, the total data volume for input and output is calculated as ((30720 * 4096) + (4096 * 1024) + (1024) + (30720 * 1024)) * 2 = 322963456 bytes (approximately 308M), which exceeds the L2 Cache (192M). This cannot guarantee that data read before computation will hit the L2 Cache. Since Global Memory bandwidth is lower than L2 Cache with a significant gap, data transfer becomes the performance bottleneck of the sample execution. Therefore, the input data needs to be split into multiple blocks so that the computation data volume (including input and output) of each data block can hit the L2 Cache. This sample provides the L2CacheOptimizer class, where the GetTileNum interface is used to automatically obtain the total number of L2 split portions for the left and right matrices based on their shapes, the GetBlockShape interface obtains the lengths of the M, N, and K axes after L2 splitting, and the GetBlockCoord interface returns the position coordinates of the corresponding block, that is, the offsets relative to the matrix starting position in the M, N, and K directions.

- Sample Specifications:
  In this sample: M = 30720, N = 1024, K = 4096.
  <table>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="5" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">c</td>
  <td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_l2cache_custom</td></tr>
  </table>

- Sample Implementation:
  - Key Kernel Steps
    - Set the left matrix A, right matrix B, and Bias. Use the GetTileNum interface of the L2CacheOptimizer class to obtain the total number of L2 split portions for the left and right matrices, and perform computations in a loop multiple times.
      ```cpp
      L2CacheOpt l2Opt(shapes, blockNum);
      matmulObj.SetOrgShape(shapes.m, shapes.n, shapes.k);
      for (int64_t tileIdx = curBlockIdx; tileIdx < l2Opt.GetTileNum(); tileIdx += blockNum) {
          auto blockShape = l2Opt.GetBlockShape(tileIdx);  // Get L2 split block size for single computation
          if (Get<0>(blockShape) <= 0 || Get<1>(blockShape) <= 0) {
              return;
          }
          auto blockCoord = l2Opt.GetBlockCoord(tileIdx);  // Get current computation index blockCoord
          matmulObj.SetTail(Get<0>(blockShape), Get<1>(blockShape), Get<2>(blockShape));
          const auto& offsetCoord = CalcOffset(shapes, blockCoord); // Calculate matrix offset based on index
          int64_t offsetA = Get<0>(offsetCoord);
          int64_t offsetB = Get<1>(offsetCoord);
          int64_t offsetC = Get<2>(offsetCoord);
          matmulObj.SetTensorA(aGlobal[offsetA], false);
          matmulObj.SetTensorB(bGlobal[offsetB], false);
          if (shapes.isBias) {
              matmulObj.SetBias(biasGlobal);
          }
          matmulObj.IterateAll(cGlobal[offsetC]);  // Compute L2 split block
      }
      matmulObj.End();
      ```

  - Key Tiling Steps
    - This sample uses constant Tiling computation. On the kernel side, a set of fixed basic block information is set, and other Tiling information is derived through constant computation on the kernel side, eliminating the need for runtime Tiling information on the kernel side. Based on this set of optimal basic block information, it is applicable to scenarios where M and N in the input shape are large. The sample provides an L2Cache splitting algorithm (refer to the L2CacheOptimizer class in the sample). This algorithm currently calculates the number of L2 split blocks on the kernel side, and can also be migrated to the host side for calculation.
    - L2CacheOptimizer specific calculation steps:
      - Determine whether L2 blocking is needed
        ```cpp
        bool smallDim = mTileNum_ < L1_MIN_UST_DIM && nTileNum_ < L1_MIN_UST_DIM;
        if (smallDim || (!EnableL2Tile())) { // Check if total computation data is less than L2Cache threshold
            mL2TileNum_ = mTileNum_;
            nL2TileNum_ = nTileNum_;
            mL2BlockNum_ = 1;
            nL2BlockNum_ = 1;
            return; // No splitting needed, return early
        }
        InitL2TileTail(); // Calculate L2 splitting
        ```
      - Calculate optimal L2 block based on load balancing
        ```cpp
        int64_t mConflict = INT64_MAX;
        int64_t nConflict = INT64_MAX;
        constexpr bool isNMajor = l1N > l1M; // Determine major dimension based on shape size
        for (int64_t i = maxMajor; i >= L1_MIN_UST_DIM; i--) {
            for (int64_t j = maxMinor; j >= minMinor; j--) {
                if (GetTotalSize(j * l1M, i * l1N, k_) <= L2_TILE_THRESHOLD) { // Ensure block is less than L2Cache threshold
                    uint64_t mConflictTmp = AscendC::Ceil(blockNum_, mL2TileNumTailTmp); // Calculate load conflict value
                    uint64_t nConflictTmp = AscendC::Ceil(blockNum_, nL2TileNumTailTmp);
                    if (mConflict >= mConflictTmp && nConflict >= nConflictTmp) { // If conflict value is smaller, update block count
                        mConflict = mConflictTmp;
                        nConflict = nConflictTmp;
                        mL2TileNum_ = curMajorDim;
                        nL2TileNum_ = curMinorDim;
                    }
                }
            }
        }
        ```

  - Invocation Implementation
    Use the kernel call operator <<<>>> to invoke the kernel function.

## Compilation and Execution

Execute the following steps in the root directory of this sample to compile and run the sample.

- Configure Environment Variables
  Select the corresponding environment variable configuration command based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on your current environment.
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
  mkdir -p build && cd build;    # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py    # Generate test input data
  ./demo                        # Execute the compiled executable program to run the sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin    # Verify output correctness, confirm algorithm logic
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  For example:
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

  The following execution result indicates successful precision comparison:
  ```bash
  test pass!
  ```