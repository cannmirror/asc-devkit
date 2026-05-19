# BlockReduce Class API Multi-Scenario Example

## Overview

This sample demonstrates multi-scenario reduction functionality using BlockReduceMax, BlockReduceMin, and BlockReduceSum APIs in reduction scenarios. It performs reduction operations (finding maximum, minimum, or sum) on all elements within each datablock (32 bytes) of the input tensor.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── block_reduce_min_max_sum
│   ├── scripts
│   │   ├── gen_data.py                    // Script to generate input data and golden data
│   │   └── verify_result.py              // Script to verify output data against golden data
│   ├── CMakeLists.txt                    // Build configuration file
│   ├── data_utils.h                      // Data read/write functions
│   └── block_reduce_min_max_sum.asc      // Ascend C sample implementation & invocation example
```

## Scenario Description

This sample selects different reduction scenarios through the compile parameter `SCENARIO_NUM`. All scenarios use ND data format, and the kernel function name is `block_reduce_custom`.

**Scenario 1: BlockReduceMax<half>**
- Input: [1, 256] half elements, mask=128 (256/sizeof(half)), repeat=2
- Output: [1, 16] half elements (maximum values for each of the 16 datablocks)
- Implementation: `BlockReduceMax<half>(dstLocal, srcLocal, repeat=2, mask=128, dstRepStride=1, srcBlkStride=1, srcRepStride=8)`
- Description: Finds the maximum value among all elements in each datablock. One datablock processes 32 bytes (16 half elements). With 256 elements across 16 datablocks, it outputs 16 maximum values.

**Scenario 2: BlockReduceMin<half>**
- Input: [4, 128] half elements, mask=128 (256/sizeof(half)), repeat=4
- Output: [4, 8] half elements (minimum values for each of the 32 datablocks)
- Implementation: `BlockReduceMin<half>(dstLocal, srcLocal, repeat=4, mask=128, dstRepStride=1, srcBlkStride=1, srcRepStride=8)`
- Description: Finds the minimum value among all elements in each datablock. One datablock processes 32 bytes (16 half elements). With 512 elements across 32 datablocks, it outputs 32 minimum values.

**Scenario 3: BlockReduceSum<float>**
- Input: [1, 128] float elements, mask=64 (256/sizeof(float)), repeat=2
- Output: [1, 16] float elements (sum results for each of the 16 datablocks)
- Implementation: `BlockReduceSum<float>(dstLocal, srcLocal, repeat=2, mask=8, dstRepStride=1, srcBlkStride=1, srcRepStride=8)`
- Description: Sums all elements in each datablock using a binary tree pairwise addition approach. One datablock processes 32 bytes (8 float elements). With 128 elements across 16 datablocks, it outputs 16 sum results.

## Sample Specifications

<table border="2">
<caption>Table 1: Sample Input/Output Specifications (Scenario 1: BlockReduceMax)</caption>
<tr><td rowspan="2" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">[1, 256]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">Sample Output</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">y</td><td align="center">[1, 16]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Kernel Name</td><td colspan="4" align="center">block_reduce_custom</td></tr>
</table>

<table border="2">
<caption>Table 2: Sample Input/Output Specifications (Scenario 2: BlockReduceMin)</caption>
<tr><td rowspan="2" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">[4, 128]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">Sample Output</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">y</td><td align="center">[4, 8]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Kernel Name</td><td colspan="4" align="center">block_reduce_custom</td></tr>
</table>

<table border="2">
<caption>Table 3: Sample Input/Output Specifications (Scenario 3: BlockReduceSum)</caption>
<tr><td rowspan="2" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">[1, 128]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">Sample Output</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">y</td><td align="center">[1, 16]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Kernel Name</td><td colspan="4" align="center">block_reduce_custom</td></tr>
</table>

## Build and Run

Execute the following steps in the sample root directory to build and run the sample.

- Configure Environment Variables  
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development toolkit on your current environment.
  - Default path, root user installed CANN package

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, non-root user installed CANN package

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Custom path install_path, installed CANN package

    ```bash
    source ${install_path}/cann/set_env.sh
    ```
    
- Sample Execution

  ```bash
  SCENARIO_NUM=1
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO_NUM   # Generate test input data
  ./demo                           # Execute the compiled executable to run the sample
  python3 ../scripts/verify_result.py -scenarioNum=$SCENARIO_NUM ./output/output.bin ./output/golden.bin  # Verify output results
  ```

  For CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  Example:

  ```bash
  cmake -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clear the cmake cache. Execute `rm CMakeCache.txt` in the build directory and run cmake again.

- Build Options

  | Option | Available Values | Description |
  |--------|------------------|-------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 for Atlas A2 Training Series/Atlas A2 Inference Series and Atlas A3 Training Series/Atlas A3 Inference Series, dav-3510 for Ascend 950PR/Ascend 950DT |
  | `SCENARIO_NUM` | `1` (default), `2`, `3` | Scenario number |

- Execution Result

  The following output indicates successful accuracy comparison.

  ```bash
  test pass!
  ```