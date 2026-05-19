# Single-sided Clamp Sample

## Overview

This sample implements the functionality of single-sided truncation of a tensor to a scalar using the ClampMin/ClampMax high-level API.
ClampMin replaces elements in the input tensor that are less than the scalar with the scalar, while ClampMax replaces elements in the input tensor that are greater than the scalar with the scalar. You can choose to use the ClampMin or ClampMax functionality through parameter configuration.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```plain
├── clampmin
│   ├── scripts
│   │   └── gen_data.py   // Input data and golden data generation script
│   ├── CMakeLists.txt    // Build project file
│   ├── data_utils.h      // Data read and write functions
│   └── clampmin.asc      // Ascend C sample implementation & call sample
```

## Sample Description

- Sample Function:
  Choose to use ClampMin or ClampMax functionality through parameter configuration:
  - ClampMin: Replaces numbers in srcTensor that are less than scalar with scalar, while numbers greater than or equal to scalar remain unchanged, and outputs as dstTensor
  - ClampMax: Replaces numbers in srcTensor that are greater than scalar with scalar, while numbers less than or equal to scalar remain unchanged, and outputs as dstTensor

  The calculation formula is as follows:

  $$
  ClampMin(srcTensor_i, scalar) =
  \begin{cases}
  srcTensor_i, & srcTensor_i \geq scalar \\
  scalar, & srcTensor_i < scalar
  \end{cases}
  $$

  $$
  ClampMax(srcTensor_i, scalar) =
  \begin{cases}
  srcTensor_i, & srcTensor_i \leq scalar \\
  scalar, & srcTensor_i > scalar
  \end{cases}
  $$

- Sample Specifications:
  <table>
  <caption>Table 1: Sample Specifications</caption>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="4" align="center"> clampmin/clampmax </td></tr>

  <tr><td rowspan="3" align="center">Sample Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src</td><td align="center">[1, 256]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">Sample Output</td></tr>
  <tr><td align="center">dst</td><td align="center">[1, 256]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">clampmin_clampmax_custom</td></tr>
  </table>

- Sample Implementation:

  - Kernel Implementation

    Uses the ClampMin/ClampMax high-level API interface to complete single-sided truncation calculation. You can choose to use ClampMin or ClampMax functionality through parameter configuration.

  - Tiling Implementation

    The host side uses the GetClampMaxMinTmpSize interface to obtain the maximum and minimum temporary space required for the ClampMin/ClampMax interface calculation.

  - Call Implementation
    Uses the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this sample to build and run the sample.

- Configure Environment Variables
  Select the appropriate environment variable configuration command based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on your current environment.
  - Default path, CANN software package installed by root user

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, CANN software package installed by non-root user

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, CANN software package installed

    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Sample Execution

  ```bash
  SCENARIO=0  # 0: ClampMin, 1: ClampMax
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DSCENARIO=$SCENARIO -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py --scenario $SCENARIO  # Generate test input data
  ./demo                           # Execute the generated executable program to run the sample
  ```

  For CPU debugging or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  Example:

  ```bash
  cmake -DSCENARIO=$SCENARIO -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debugging mode
  cmake -DSCENARIO=$SCENARIO -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clear the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build Option Description

  | Option | Available Values | Description |
  |--------|------------------|-------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debugging, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2/A3 series, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  | `SCENARIO` | `0` (default), `1` | Scenario: 0 corresponds to ClampMin, 1 corresponds to ClampMax |

- Execution Result

  The execution result is as follows, indicating that the accuracy comparison passed.

  ```bash
  test pass!
  ```