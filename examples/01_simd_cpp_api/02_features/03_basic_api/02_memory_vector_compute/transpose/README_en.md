# Transpose Example

## Overview

This example demonstrates data transposition functionality using the Transpose and TransDataTo5HD interfaces. It covers three scenarios: basic transpose, enhanced transpose, and 5HD format conversion. The example addresses three types of data transformation requirements: (1) transposing 16x16 2D matrix blocks, (2) converting between [N,C,H,W] and [N,H,W,C] 4D matrix formats, and (3) converting NCHW format to NC1HWC0 format.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── transpose
│   ├── scripts
│   │   ├── gen_data.py         // Input and golden data generation
│   │   └── verify_result.py    // Output and golden data verification
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── transpose.asc           // Ascend C implementation & invocation example
```

## Example Description

- Example Function:
  The transpose example implements matrix transposition, supporting three scenarios: basic transpose, enhanced transpose, and 5HD format conversion:

  1. Basic transpose: Supports transposing 16x16 2D matrix blocks

  2. Enhanced transpose: Supports 16x16 2D matrix block transposition through transposeParams specification, and supports conversion between [N,C,H,W] and [N,H,W,C] 4D matrices

  3. 5HD format conversion: Supports 16x16 2D matrix block transposition and conversion from [N,C,H,W] 4D format to [N,C1,H,W,C0] 5D format

- Scenario Description:

  The example can switch between different scenarios using the compilation parameter `SCENARIO_NUM`. For parameter details, see the table below:

  | Scenario Number | Scenario Name | Input Shape | Output Shape | Data Type | Input Format | Output Format | Transpose Type |
  |------|--------|------|------|--------|------|------|------|
  | 1 | Basic Transpose | [16,16] | [16,16] | half | ND | ND | / |
  | 2 | Enhanced Transpose | [3,3,2,8] | [3,2,8,3] | half | NCHW | NHWC | TRANSPOSE_NCHW2NHWC |
  | 3 | 5HD Format Conversion | [2,32,16,16] | [2,2,16,16,16] | half | NCHW | NC1HWC0 | / |

- Data Format Description:
  Feature maps in convolutional neural networks are typically stored as 4D arrays (4D format), explained as follows:
    - N: Batch size.
    - H: Height, the feature map height.
    - W: Width, the feature map width.
    - C: Channels, the feature map channels.

  Since data can only be stored linearly, these four dimensions have corresponding orders. Different deep learning frameworks store feature map data in different orders. For example, TensorFlow uses [Batch, Height, Width, Channels], i.e., NHWC.

  The 5HD format (NC1HWC0) is a data layout format specific to Ascend NPU, where:
    - N: Batch size.
    - C1: C1 = ceil(C / C0). If the result is not evenly divisible, round down.
    - H: Height, the feature map height.
    - W: Width, the feature map width.
    - C0: Equal to the matrix computation unit size in AI Core. If the data type bit width is 32-bit or 16-bit, C0=16; if the data type bit width is 8-bit, C0=32. In this example, C0=16.

    The NHWC/NCHW -> NC1HWC0 conversion process: Split the data along the C dimension into C1 parts of NHWC0/NC0HW, then arrange the C1 parts of NHWC0/NC0HW consecutively in memory as NC1HWC0.

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
  SCENARIO_NUM=1      # Set scenario number (1=basic transpose, 2=enhanced transpose, 3=5HD format conversion)
  mkdir -p build && cd build;      # Create and enter build directory
  cmake .. -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j;    # Build project
  python3 ../scripts/gen_data.py -scenario_num=$SCENARIO_NUM   # Generate test input data
  ./demo                           # Execute the compiled program
  python3 ../scripts/verify_result.py ./output/output.bin ./output/golden.bin  # Verify output correctness
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
  | `SCENARIO_NUM` | `1` (default), `2`, `3` | Scenario number: 1 (basic transpose), 2 (enhanced transpose), 3 (5HD format conversion) |

- Execution Result

  The following output indicates successful precision comparison:
  ```bash
  test pass!
  ```