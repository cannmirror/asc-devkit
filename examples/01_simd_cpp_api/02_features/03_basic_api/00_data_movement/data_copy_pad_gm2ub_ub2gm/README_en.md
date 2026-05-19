# DataCopyPad Example

## Overview

This example demonstrates data transfer and padding functionality for non-32-byte aligned data using the DataCopyPad API in data transfer scenarios. The DataCopyPad API supports non-aligned data transfer from Global Memory to Local Memory, with the ability to pad specified values on the left or right side of the data.

The data transfer process includes: Global Memory (GM) -> Unified Buffer (UB) (using DataCopyPad for non-aligned transfer with padding) -> Global Memory (GM). This example uses static Tensor allocation for UB memory and supports switching between different scenarios through compile parameters to demonstrate different usage modes of DataCopyPad.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── data_copy_pad_gm2ub_ub2gm
│   ├── scripts
│   │   ├── gen_data.py         // Input data and ground truth data generation script
│   │   └── verify_result.py    // Verification script for comparing output data with ground truth
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── data_copy_pad.asc       // Ascend C example implementation & invocation example
```

## Scenario Description

This example selects different scenarios through the compile parameter `SCENARIO_NUM`. All scenarios use ND data format with kernel function name `data_copy_pad_custom`.

<table border="2">
<caption>Table 1: Scenario Configuration Reference</caption>
<tr><th>scenarioNum</th><th>Input Shape</th><th>Output Shape</th><th>Data Type</th><th>Padding/Transfer Mode</th><th>Description</th></tr>
<tr><td>1</td><td>[1, 20]</td><td>[1, 32]</td><td>half</td><td>SetPadValue Padding</td><td>Right padding of 12 elements, requires SetPadValue to set padding value to 1</td></tr>
<tr><td>2</td><td>[32, 59]</td><td>[32, 64]</td><td>float</td><td>rightPadding</td><td>Right padding of 5 elements, default padding value is 0, no SetPadValue required</td></tr>
<tr><td>3</td><td>[3, 24]</td><td>[1, 80]</td><td>half</td><td>Compact Padding</td><td>Compact mode, last data block right padded with 16 bytes (dav-3510 only)</td></tr>
<tr><td>4</td><td>[1, 320]</td><td>[1, 576]</td><td>int8</td><td>LoopMode Transfer (Compact)</td><td>SetLoopModePara enables loop mode, Compact mode, implements GM->UB non-contiguous stride transfer (dav-3510 only)</td></tr>
<tr><td>5</td><td>[1, 320]</td><td>[1, 576]</td><td>int8</td><td>LoopMode Transfer (Normal)</td><td>SetLoopModePara enables loop mode, Normal mode, implements GM->UB non-contiguous stride transfer (dav-3510 only)</td></tr>
<tr><td>6</td><td>[2, 4, 3, 128, 126]</td><td>[512, 128]</td><td>int8</td><td>LoopMode Transfer (Normal)</td><td>5D data transfer, transfers [2, 2, 2, 64, 126], each row padded with 2 bytes to 128 bytes (dav-3510 only)</td></tr>
</table>

### Detailed Scenario Description

**Scenario 1: Custom Padding Using SetPadValue**
- Input shape: [1, 20]
- Output shape: [1, 32]
- Data type: half
- Parameter configuration: isPad=false, leftPadding=0, rightPadding=12
- Description: Use SetPadValue to set padding value to 1, right pad 12 elements. **SetPadValue requires user to explicitly call and set the padding value**, used with isPad=false.

**Scenario 2: Default Padding Using rightPadding**
- Input shape: [32, 59]
- Output shape: [32, 64]
- Data type: float
- Parameter configuration: isPad=true, leftPadding=0, rightPadding=5
- Description: **No need to use SetPadValue**, when isPad=true, padding value defaults to 0, right pad 5 elements.

**Scenario 3: Data Transfer Using Compact Mode ----This scenario is only supported on Ascend 950PR/Ascend 950DT products**
- Input shape: [3, 24]
- Output shape: [1, 80]
- Data type: half
- Parameter configuration: blockLen=48, blockCount=3, leftPadding=0, rightPadding=16, isPad=false
- Description: Compact mode allows single non-aligned transfer, with unified padding at the end of the entire data block to 32-byte alignment. In this example, leftPadding is 0, rightPadding is 16, padding 16 bytes on the right side of the last data block. The destination operand data size is 160 bytes.

**Scenario 4: Using SetLoopModePara to Enable Loop Mode (Compact Mode) ----This scenario is only supported on Ascend 950PR/Ascend 950DT products**
- Input shape: [1, 320], as shown in Figure 1
- Output shape: [1, 576], as shown in Figure 2
- Data type: int8
- Parameter configuration:
  - GM->UB: LoopModeParams{loop1Size=2, loop2Size=2, loop1SrcStride=80, loop1DstStride=128, loop2SrcStride=160, loop2DstStride=288}, DataCopyMVType::OUT_TO_UB
  - DataCopyExtParams: BLOCK_COUNT=2, BLOCK_LEN=40 (using constexpr constant)
  - DataCopyPadExtParams: isPad=true, leftPadding=0, rightPadding=0, padValue=-1
- Description: Enable loop mode through SetLoopModePara, use Compact mode to implement GM->UB non-contiguous stride data transfer. In Compact mode, each inner loop transfers 80B then pads 16B for 96B alignment, padding value set to -1.

**Figure 1** Source Operand Transfer Scenario Example

<img src="figure/datacopypad1.png" width="80%">

**Figure 2** Destination Operand Compact Mode Transfer Scenario Example

<img src="figure/datacopypad2.png">

**Scenario 5: Using SetLoopModePara to Enable Loop Mode (Normal Mode) ----This scenario is only supported on Ascend 950PR/Ascend 950DT products**
- Input shape: [1, 320], as shown in Figure 1
- Output shape: [1, 576], as shown in Figure 3
- Data type: int8
- Parameter configuration:
  - GM->UB: LoopModeParams{loop1Size=2, loop2Size=2, loop1SrcStride=80, loop1DstStride=128, loop2SrcStride=160, loop2DstStride=288}, DataCopyMVType::OUT_TO_UB
  - DataCopyExtParams: BLOCK_COUNT=2, BLOCK_LEN=40 (using constexpr constant)
  - DataCopyPadExtParams: isPad=true, leftPadding=0, rightPadding=0, padValue=-1
- Description: Enable loop mode through SetLoopModePara, use Normal mode to implement GM->UB non-contiguous stride data transfer. In Normal mode, each block is padded with 24B after transfer for 64B alignment, padding value set to -1.

**Figure 3** Destination Operand Normal Mode Transfer Scenario Example

<img src="figure/datacopypad3.png">

**Scenario 6: Using SetLoopModePara to Enable Loop Mode (Normal Mode) for 5D Data Transfer ----This scenario is only supported on Ascend 950PR/Ascend 950DT products**
- Input shape: [2, 4, 3, 128, 126], 5D data
- Output shape: [512, 128], stored contiguously in UB
- Data type: int8
- Transfer specification: [2, 2, 2, 64, 126], each 126 bytes padded with 2 bytes to 128 bytes
- Parameter configuration:
  - GM->UB: LoopModeParams{loop1Size=2, loop2Size=2, loop1SrcStride=128*126, loop1DstStride=64*128, loop2SrcStride=3*128*126, loop2DstStride=2*64*128}, DataCopyMVType::OUT_TO_UB
  - DataCopyExtParams: blockCount=64, blockLen=126, srcStride=0, dstStride=0
  - DataCopyPadExtParams: isPad=true, leftPadding=0, rightPadding=0, padValue=0
  - Dimension 0 uses for loop to transfer 2 times

## Build and Run

Execute the following steps in the root directory of this example to build and run the example.

- Configure Environment Variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on your current environment.
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

- Example Execution

  ```bash
  SCENARIO_NUM=1  # Set scenario number
  mkdir -p build && cd build;      # Create and enter build directory
  cmake .. -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j;    # Build project
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO_NUM   # Generate test input data
  ./demo                           # Execute the compiled executable to run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin -scenarioNum=$SCENARIO_NUM  # Verify output results
  ```

  When using NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:

  ```bash
  cmake .. -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Build Option Description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `sim` | Run mode: NPU execution, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  | `SCENARIO_NUM` | `1` (default), `2`, `3`, `4`, `5`, `6` | Scenario number: 1 (SetPadValue padding), 2 (rightPadding), 3 (Compact mode), 4 (SetLoopModePara loop mode Compact), 5 (SetLoopModePara loop mode Normal), 6 (5D LoopMode Normal) |

- Execution Result

  When the execution result shows the following, it indicates successful precision comparison.

  ```bash
  test pass!
  ```