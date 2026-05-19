# TransData Example

## Overview

This example implements data layout format conversion functionality based on the TransData high-level API. It supports converting the input data layout format to the target layout format.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```plain
├── transdata
│   ├── scripts
│   │   └── gen_data.py         // Input data and ground truth data generation script
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read and write functions
│   └── transdata.asc           // Ascend C sample implementation & call example
```

## Example Description

Convert the input data layout format to the target layout format.  
In addition to dimension order transformation, this involves splitting the C-axis and N-axis. The specific conversion method is: C-axis is split into C1-axis and C0-axis, N-axis is split into N1-axis and N0-axis. For data types with a bit width of 16, C0 and N0 are fixed at 16. The calculation formulas for C1 and N1 are as follows:

$$ C1 = (C + C0 - 1) / C0 $$

$$ N1 = (N + N0 - 1) / N0 $$

This example supports the following four data format conversion scenarios:

### Scenario 1: NCDHW -> FRACTAL_Z_3D (mode = 1)

- Example Specifications:
  <table>
  <caption>Table 1: Scenario 1 Example Input/Output Specifications</caption>
  <tr><td rowspan="1" align="center">Example Type(OpType)</td><td colspan="4" align="center"> transdata </td></tr>

  <tr><td rowspan="3" align="center">Example Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src</td><td align="center">[16, 16, 1, 3, 5]</td><td align="center">half</td><td align="center">NCDHW</td></tr>
  <tr><td rowspan="2" align="center">Example Output</td></tr>
  <tr><td align="center">dst</td><td align="center">[1, 1, 3, 5, 1, 16, 16]</td><td align="center">half</td><td align="center">FRACTAL_Z_3D</td></tr>

  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">transdata_custom</td></tr>
  </table>

  **Note**:
  - Input shape is [N, C, D, H, W] = [16, 16, 1, 3, 5]
  - Output shape is [D, C1, H, W, N1, N0, C0] = [1, 1, 3, 5, 1, 16, 16]

### Scenario 2: FRACTAL_Z_3D -> NCDHW (mode = 2)

- Example Specifications:
  <table>
  <caption>Table 2: Scenario 2 Example Input/Output Specifications</caption>
  <tr><td rowspan="1" align="center">Example Type(OpType)</td><td colspan="4" align="center"> transdata </td></tr>

  <tr><td rowspan="3" align="center">Example Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src</td><td align="center">[1, 1, 3, 5, 1, 16, 16]</td><td align="center">half</td><td align="center">FRACTAL_Z_3D</td></tr>
  <tr><td rowspan="2" align="center">Example Output</td></tr>
  <tr><td align="center">dst</td><td align="center">[16, 16, 1, 3, 5]</td><td align="center">half</td><td align="center">NCDHW</td></tr>

  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">transdata_custom</td></tr>
  </table>

  **Note**:
  - Input shape is [D, C1, H, W, N1, N0, C0] = [1, 1, 3, 5, 1, 16, 16]
  - Output shape is [N, C, D, H, W] = [16, 16, 1, 3, 5]
  - This is the inverse operation of Scenario 1

### Scenario 3: NCDHW -> NDC1HWC0 (mode = 3)

- Example Specifications:
  <table>
  <caption>Table 3: Scenario 3 Example Input/Output Specifications</caption>
  <tr><td rowspan="1" align="center">Example Type(OpType)</td><td colspan="4" align="center"> transdata </td></tr>

  <tr><td rowspan="3" align="center">Example Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src</td><td align="center">[16, 16, 1, 3, 5]</td><td align="center">half</td><td align="center">NCDHW</td></tr>
  <tr><td rowspan="2" align="center">Example Output</td></tr>
  <tr><td align="center">dst</td><td align="center">[16, 1, 1, 3, 5, 16]</td><td align="center">half</td><td align="center">NDC1HWC0</td></tr>

  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">transdata_custom</td></tr>
  </table>

  **Note**:
  - Input shape is [N, C, D, H, W] = [16, 16, 1, 3, 5]
  - Output shape is [N, D, C1, H, W, C0] = [16, 1, 1, 3, 5, 16]

### Scenario 4: NDC1HWC0 -> NCDHW (mode = 4)

- Example Specifications:
  <table>
  <caption>Table 4: Scenario 4 Example Input/Output Specifications</caption>
  <tr><td rowspan="1" align="center">Example Type(OpType)</td><td colspan="4" align="center"> transdata </td></tr>

  <tr><td rowspan="3" align="center">Example Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src</td><td align="center">[16, 1, 1, 3, 5, 16]</td><td align="center">half</td><td align="center">NDC1HWC0</td></tr>
  <tr><td rowspan="2" align="center">Example Output</td></tr>
  <tr><td align="center">dst</td><td align="center">[16, 16, 1, 3, 5]</td><td align="center">half</td><td align="center">NCDHW</td></tr>

  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">transdata_custom</td></tr>
  </table>

  **Note**:
  - Input shape is [N, D, C1, H, W, C0] = [16, 1, 1, 3, 5, 16]
  - Output shape is [N, C, D, H, W] = [16, 16, 1, 3, 5]
  - This is the inverse operation of Scenario 3

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
  SCENARIO=1
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO=$SCENARIO ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py --mode $SCENARIO  # Generate test input data
  ./demo                           # Execute the compiled executable program to run the example
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  Examples:

  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO=$SCENARIO ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO=$SCENARIO ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build Option Description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2/A3 series, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  | `SCENARIO` | `1` (default), `2`, `3`, `4` | Scenario: 1=NCDHW→FRACTAL_Z_3D, 2=FRACTAL_Z_3D→NCDHW, 3=NCDHW→NDC1HWC0, 4=NDC1HWC0→NCDHW |

- Execution Result

  The execution result is shown below, indicating the accuracy comparison passed.

  ```bash
  test pass!
  ```