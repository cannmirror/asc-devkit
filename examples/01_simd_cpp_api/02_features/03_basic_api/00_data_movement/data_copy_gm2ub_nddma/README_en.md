# DataCopy Multi-dimensional Data Transfer Interface Example

## Overview

This example introduces how to use the multi-dimensional data transfer interface to implement GM (Global Memory) to UB (Unified Buffer) data transfer. By freely configuring the transfer dimension information and corresponding stride, it can be used for various data transformation operations such as Padding, Transpose, BroadCast, and Slice.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── data_copy_gm2ub_nddma
│   ├── scripts
│   │   ├── gen_data.py         // Input data and golden data generation script
│   │   └── verify_result.py    // Verification script for comparing output data with golden data
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── multidimensional_data_movement.asc      // Ascend C example implementation & invocation example
```

## Scenario Description

This example selects different scenarios through the compilation parameter `SCENARIO_NUM`. All scenarios use 2D ND format data with kernel function name `datacopy_custom`.

<table border="2">
<caption>Table 1: Scenario Configuration Reference</caption>
<tr><th>scenarioNum</th><th>Input Shape</th><th>Output Shape</th><th>Data Type</th><th>Description</th></tr>
<tr><td>1</td><td>[16, 32]</td><td>[32, 64]</td><td>float32</td><td>Padding scenario</td></td>
<tr><td>2</td><td>[28, 15]</td><td>[32, 32]</td><td>float</td><td>Padding scenario with nearest value fill mode enabled</td></tr>
<tr><td>3</td><td>[16, 64]</td><td>[64, 16]</td><td>float32</td><td>Transpose scenario</td></tr>
<tr><td>4</td><td>[1, 16]</td><td>[3, 16]</td><td>float32</td><td>BroadCast scenario</td></tr>
<tr><td>5</td><td>[32, 64]</td><td>[16, 16]</td><td>float32</td><td>Slice scenario</td></tr>
</table>

### Scenario Details

**Scenario 1: Padding Scenario**
- Input: [16, 32] float32 elements
- Output: [32, 64] float32 elements
- Parameter Configuration: NdDmaLoopInfo={{1, 32}, {1, 64}, {32, 16}, {15, 13}, {17, 3}}, paddingValue=0
- Description: Transfer [16, 32] data from GM to UB and pad to [32, 64], with left padding 15, top padding 13, right padding 17, bottom padding 3, padding value filled with 0

**Scenario 2: Padding Scenario with Nearest Value Fill Mode Enabled**
- Input: [28, 15] float32 elements
- Output: [32, 32] float32 elements
- Parameter Configuration: NdDmaLoopInfo={{1, 15}, {1, 32}, {15, 28}, {11, 3}, {6, 1}}, isNearestValueMode=true
- Description: Transfer [28, 15] data from GM to UB and pad to [32, 32], enable nearest value fill mode, padding positions filled with boundary data instead of 0

**Scenario 3: Transpose Scenario**
- Input: [16, 64] float32 elements
- Output: [64, 16] float32 elements
- Parameter Configuration: NdDmaLoopInfo={{1, 64}, {16, 1}, {64, 16}, {0, 0}, {0, 0}}
- Description: Transfer [16, 64] data from GM to UB and transpose to [64, 16], achieving row-column swap through stride configuration

**Scenario 4: BroadCast Scenario**
- Input: [1, 16] float32 elements
- Output: [3, 16] float32 elements
- Parameter Configuration: NdDmaLoopInfo={{1, 0}, {1, 16}, {16, 3}, {0, 0}, {0, 0}}
- Description: Transfer [1, 16] data from GM to UB and broadcast to [3, 16], achieving row data replication through stride configuration set to 0

**Scenario 5: Slice Scenario**
- Input: [32, 64] float32 elements
- Output: [16, 16] float32 elements
- Parameter Configuration: NdDmaLoopInfo={{1, 64}, {1, 16}, {16, 16}, {0, 0}, {0, 0}}
- Description: Transfer [32, 64] data from GM to UB and slice to [16, 16], achieving final data slicing through transfer amount configuration

## Compilation and Execution

Execute the following steps in the root directory of this example to compile and run the example.
- Configure Environment Variables
  Select the appropriate environment variable configuration command based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on the current environment.
  - Default path, CANN software package installed by root user

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, CANN software package installed by non-root user

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, CANN software package installation

    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Example Execution

  ```bash
  SCENARIO_NUM=1
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO_NUM   # Generate test input data
  ./demo                           # Execute compiled executable to run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin  # Verify output correctness
  ```

  When using CPU debug mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` parameter.

  Example:

  ```bash
  cmake -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  ```

  > **Note:** Before switching compilation modes, clean the cmake cache by executing `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Compilation Options Description

  | Option | Possible Values | Description |
  |--------|-----------------|-------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu` | Run mode: NPU execution, CPU debug |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` (default) | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  | `SCENARIO_NUM` | `1` (default), `2`, `3`, `4`, `5` | Scenario number |

- Execution Result

  The following result indicates successful precision comparison:

  ```bash
  test pass!
  ```