# BroadCast Example

## Overview

This example implements data broadcast functionality based on the Broadcast high-level API. It supports expanding the input Tensor to a target shape along a specified axis, suitable for scenarios such as data alignment and dimension expansion.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```plain
├── broadcast
│   ├── scripts
│   │   └── gen_data.py         // Input data and ground truth data generation script
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read and write functions
│   └── broadcast.asc           // Ascend C sample implementation & call example
```

## Example Description

- Example Function:  
  Perform broadcast computation on the input Tensor.

- Example Specifications:

<table>
  <caption>Table 1: Example Specification Description - Scenario 0</caption>
  <tr>
    <td align="center">Example Type(OpType)</td>
    <td colspan="4" align="center">broadcast</td>
  </tr>
  <tr>
    <td rowspan="2" align="center">Example Input</td>
    <td align="center">name</td>
    <td align="center">shape</td>
    <td align="center">data type</td>
    <td align="center">format</td>
  </tr>
  <tr>
    <td align="center">x</td>
    <td align="center">[1, 48]</td>
    <td align="center">float</td>
    <td align="center">ND</td>
  </tr>
  <tr>
    <td align="center">Example Output</td>
    <td align="center">y</td>
    <td align="center">[96, 48]</td>
    <td align="center">float</td>
    <td align="center">ND</td>
  </tr>
  <tr>
    <td align="center">Kernel Function Name</td>
    <td colspan="4" align="center">broadcast_custom</td>
  </tr>
</table>

<table>
  <caption>Table 2: Example Specification Description - Scenario 1</caption>
  <tr>
    <td align="center">Example Type(OpType)</td>
    <td colspan="4" align="center">broadcast</td>
  </tr>
  <tr>
    <td rowspan="2" align="center">Example Input</td>
    <td align="center">name</td>
    <td align="center">shape</td>
    <td align="center">data type</td>
    <td align="center">format</td>
  </tr>
  <tr>
    <td align="center">x</td>
    <td align="center">[96, 1]</td>
    <td align="center">float</td>
    <td align="center">ND</td>
  </tr>
  <tr>
    <td align="center">Example Output</td>
    <td align="center">y</td>
    <td align="center">[96, 96]</td>
    <td align="center">float</td>
    <td align="center">ND</td>
  </tr>
  <tr>
    <td align="center">Kernel Function Name</td>
    <td colspan="4" align="center">broadcast_custom</td>
  </tr>
</table>

- Scenario Description:

  <table>
  <caption>Table 3: TESTCASE Parameter Description</caption>
  <tr><td align="center">TESTCASE</td><td align="center">Input Shape</td><td align="center">Output Shape</td><td align="center">Broadcast Axis(axis)</td><td align="center">Description</td></tr>
  <tr><td align="center">0</td><td align="center">[1, 48]</td><td align="center">[96, 48]</td><td align="center">0</td><td align="center">Broadcast along the first dimension, expanding 1 to 96</td></tr>
  <tr><td align="center">1</td><td align="center">[96, 1]</td><td align="center">[96, 96]</td><td align="center">1</td><td align="center">Broadcast along the second dimension, expanding 1 to 96</td></tr>
  </table>

- Example Implementation:  
  This example implements two broadcast scenarios: broadcasting from [1, 48] to [96, 48] and from [96, 1] to [96, 96].

  - Kernel Implementation  
    Use the Broadcast high-level API interface to complete the broadcast, expanding the input Tensor to the target shape along the specified axis.

  - Tiling Implementation  
    The tiling implementation flow for the broadcast example is as follows: First, obtain the 2D shapes of the input and output, then populate the broadcast axis and input/output tensor dimensions into TilingData.

  - Call Implementation  
    Use the kernel call operator <<<>>> to call the kernel function.

## Build and Run  

Execute the following steps in the root directory of this example to build and run the example.

- Configure Environment Variables  
  Select the appropriate environment variable configuration command based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on your current environment.
  - Default path, root user installed CANN software package

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, non-root user installed CANN software package

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, CANN software package installed

    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Example Execution

  ```bash
  TESTCASE=1                    # 0: shape[1, 48]->[96,48]  1: shape[96,1]->[96,96]
  mkdir -p build && cd build;   # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py --testcase $TESTCASE  # Generate test input data
  ./demo $TESTCASE              # Execute the compiled executable program to run the example
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  Examples:

  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build Option Description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2/A3 series, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  The execution result is shown below, indicating the accuracy comparison passed.

  ```bash
  test pass!
  ```