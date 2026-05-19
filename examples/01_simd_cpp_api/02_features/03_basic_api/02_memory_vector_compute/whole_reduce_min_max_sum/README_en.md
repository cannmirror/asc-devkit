# Reduce Interface Multi-Scenario Example

## Overview

This example demonstrates the usage of reduce interfaces in various scenarios, including WholeReduceMax, WholeReduceMin, WholeReduceSum, RepeatReduceSum, and the combination of WholeReduceMin with GetReduceRepeatMaxMinSpr for obtaining global minimum values and indices. These interfaces perform reduction operations (finding maximum, minimum, or sum) on all elements within each repeat of a LocalTensor, with results stored in the destination LocalTensor.

Note: `GetReduceRepeatMaxMinSpr` is the renamed API in CANN 9.0.0. For CANN 8.5.0 and earlier versions, use `GetReduceMaxMinCount`.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── whole_reduce_min_max_sum
│   ├── scripts
│   │   ├── gen_data.py                    // Input and golden data generation script
│   │   └── verify_result.py              // Output and golden data verification script
│   ├── CMakeLists.txt                    // Build project file
│   ├── data_utils.h                      // Data read/write functions
│   └── whole_reduce_min_max_sum.asc      // Ascend C implementation & invocation example
```

## Scenario Details

This example selects different reduction scenarios through the compilation parameter `SCENARIO_NUM`. All scenarios use ND data format with kernel function name `reduce_custom`.

**Scenario 1: WholeReduceMax**
- Input: [1, 1024] half elements, mask=128, repeat=8 (1024/128)
- Output: [1, 8] half elements (8 maximum values, no index returned)
- Implementation: `WholeReduceMax<half>(dstLocal, srcLocal, mask=128, repeat=8, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_VALUE)`
- Description: Uses ORDER_ONLY_VALUE parameter to return only maximum values without indices, output stored as [max0, max1, max2, ...]

**Scenario 2: WholeReduceMin**
- Input: [1, 1024] half elements, mask=128, repeat=8 (1024/128)
- Output: [1, 16] half elements (8 minimum values + 8 indices, interleaved storage)
- Implementation: `WholeReduceMin<half>(dstLocal, srcLocal, mask=128, repeat=8, 1, 1, 8)`
- Description: Uses default order (ORDER_VALUE_INDEX), output interleaved as [min0, idx0, min1, idx1, ...]

**Scenario 3: WholeReduceSum**
- Input: [1, 2048] float elements, mask=64 (float type is 32-bit, mask range is [1,64]), repeat=32 (2048/64)
- Output: [1, 32] float elements (sum result for each repeat)
- Implementation: `WholeReduceSum<float>(dstLocal, srcLocal, mask=64, repeat=32, 1, 1, 8)`
- Description: Each repeat sums independently, outputting 32 sum results total

**Scenario 4: RepeatReduceSum**
- Input: [1, 2048] float elements, mask=64 (float type is 32-bit, mask range is [1,64]), repeat=32 (2048/64)
- Output: [1, 32] float elements (accumulation mode)
- Implementation: First use `Duplicate` to initialize dstLocal to 0, then call `RepeatReduceSum<float>(dstLocal, srcLocal, repeat=32, mask=64, dstBlkStride=0, 1, 1, 8)`
- Description: dstBlkStride=0 means accumulating all repeat results to the same position, with each repeat result stored sequentially

**Scenario 5: WholeReduceMin + GetReduceRepeatMaxMinSpr**
- Input: [1, 1024] half elements, mask per-bit mode (uint64_t[2] all 1s), repeat=8 (1024/128)
- Output: [1, 16] half elements (only first 2 elements valid: global minimum + global minimum index)
- Implementation: First call `WholeReduceMin<half>(dstLocal, srcLocal, mask=uint64_t[2]{-1,-1}, repeat=8, 1, 1, 8)`, then call `GetReduceRepeatMaxMinSpr<half>(val, idx)` to get global minimum and its index, synchronize vector to scalar computation via `SetFlag<HardEvent::V_S>` / `WaitFlag<HardEvent::V_S>`
- Description: WholeReduceMin computes local minimum for each of 8 repeats, GetReduceRepeatMaxMinSpr reads the global minimum from all repeats and its index position in source data, writing results to the first two elements of dstLocal

**Scenario 6: WholeReduceSum Non-Aligned Scenario**
- Input: [13, 57] float elements (13 rows x 57 columns, column width 57x4 bytes = 228 bytes, not 32-byte aligned)
- Output: [1, 13] float elements (sum result for each row)
- Implementation: Use `DataCopyPad` to move non-aligned data, `WholeReduceSum<float>(dstLocal, srcLocal, mask=57, repeat=13, 1, 1, srcStride)` to sum each row
- Description: Demonstrates reduction operations with non-aligned data. Each row has 57 float elements (228 bytes), using DataCopyPad to pad to 232 bytes (58 float elements) for 32-byte alignment, WholeReduceSum's srcRepeatStride calculated based on aligned block count

## Example Specifications

<table border="2">
<caption>Table 1: Example Input/Output Specifications (Scenario 1)</caption>
<tr><td rowspan="2" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">[1, 1024]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">Example Output</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">z</td><td align="center">[1, 8]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">reduce_custom</td></tr>
</table>

<table border="2">
<caption>Table 2: Example Input/Output Specifications (Scenario 2/5)</caption>
<tr><td rowspan="2" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">[1, 1024]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">Example Output</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">z</td><td align="center">[1, 16]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">reduce_custom</td></tr>
</table>

<table border="2">
<caption>Table 3: Example Input/Output Specifications (Scenario 3/4)</caption>
<tr><td rowspan="2" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">[1, 2048]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">Example Output</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">z</td><td align="center">[1, 32]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">reduce_custom</td></tr>
</table>

<table border="2">
<caption>Table 4: Example Input/Output Specifications (Scenario 6)</caption>
<tr><td rowspan="2" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">[13, 57]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">Example Output</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">z</td><td align="center">[1, 13]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">reduce_custom</td></tr>
</table>

## Build and Run

Execute the following steps in the example root directory to build and run the example.
- Configure Environment Variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development toolkit on your current environment.
  - Default path, CANN package installed by root user

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, CANN package installed by non-root user

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Custom path install_path, CANN package installed

    ```bash
    source ${install_path}/cann/set_env.sh
    ```
    
- Example Execution

  ```bash
  SCENARIO_NUM=1  # Set scenario number
  mkdir -p build && cd build;      # Create and enter build directory
  cmake .. -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j;    # Build project
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO_NUM   # Generate test input data
  ./demo                           # Execute the compiled program
  python3 ../scripts/verify_result.py -scenarioNum=$SCENARIO_NUM ./output/output.bin ./output/golden.bin  # Verify output correctness
  ```

  For CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  Example:

  ```bash
  cmake .. -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j; # CPU debug mode
  cmake .. -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching compilation modes, clear the cmake cache by running `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Compilation Options Description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products; dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  | `SCENARIO_NUM` | `1` (default), `2`, `3`, `4`, `5`, `6` | Scenario number: 1 (WholeReduceMax), 2 (WholeReduceMin), 3 (WholeReduceSum), 4 (RepeatReduceSum), 5 (WholeReduceMin+GetReduceRepeatMaxMinSpr), 6 (Non-aligned WholeReduceSum) |

- Execution Result

  The following output indicates successful precision comparison:

  ```bash
  test pass!
  ```