# Compare Example

## Overview
This example demonstrates Compare and Compares interfaces using the Reg programming interface for data comparison in various scenarios.
This example supports two comparison scenarios, selected via the environment variable SCENARIO_NUM.
  <table>
    <tr>
      <td>SCENARIO_NUM</td>
        <td>Comparison Scenario</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Compare: Element-wise comparison between two vectors</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Compares: Element-wise comparison between a vector and a scalar</td>
    </tr>
    </table>

## Supported Products
- Ascend 950PR/Ascend 950DT

## Directory Structure
```
├── compare
│   ├── scripts
│   │   ├── gen_data.py                // Input and golden data generation script
│   ├── CMakeLists.txt                 // Build project file
│   ├── data_utils.h                   // Data read/write functions
│   ├── compare.asc                    // Ascend C implementation & invocation example
│   └── README.md                      // Example introduction
```

## Example Description
The Compare interface is typically used with the Select interface. This example only demonstrates the usage of Compare and Select together.
This example switches between different scenarios through the compilation parameter `SCENARIO_NUM`:

**Scenario 1: Compare**
- Example Function:
  Takes the element-wise maximum between two vectors xReg and yReg of the same size.
- Example Specifications:
  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="3" align="center">AIV Example</td></tr>
  <tr><td rowspan="3" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 256]</td><td align="center">float</td></tr>
  <tr><td align="center">y</td><td align="center">[1, 256]</td><td align="center">float</td></tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">z</td><td align="center">[1, 256]</td><td align="center">float</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">compare</td></tr>
  </table>
- Example Implementation:
  - Call the Compare interface in GT (greater than) mode to compare two vectors, output to maskReg: if xReg > yReg, write 1 to the corresponding bit in maskReg, otherwise write 0
  - Call the Select interface, passing the comparison result maskReg: if maskReg bit is 1, select the element from xReg at that position, otherwise select from yReg
  - For float data type, MaskReg format stores one mask per 4 bits, so Compare reads data sequentially from xReg and yReg, compares and writes to MaskReg at bit position 4 * N; Select determines whether to select data from xReg or yReg based on MaskReg bit at 4 * N.
  - Invocation Implementation: Use the kernel launch operator <<<>>> to invoke the kernel function.
  <img src="figure/compare.png">

**Scenario 2: Compares**
- Example Function:
  Compares each element of vector xReg with scalar 0. If xReg[i] > 0, zReg[i] takes xReg[i], otherwise takes yReg[i].
- Example Specifications:
  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="3" align="center">AIV Example</td></tr>
  <tr><td rowspan="3" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 256]</td><td align="center">float</td></tr>
  <tr><td align="center">y</td><td align="center">[1, 256]</td><td align="center">float</td></tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">z</td><td align="center">[1, 256]</td><td align="center">float</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">compare</td></tr>
  </table>
- Example Implementation:
  - Call the Compares interface in GT (greater than) mode to compare vector xReg with scalar 0, output to maskReg: if xReg > 0, write 1 to the corresponding bit in maskReg, otherwise write 0
  - Call the Select interface, passing the comparison result maskReg: if maskReg bit is 1, select the element from xReg at that position, otherwise select from yReg
  - For float data type, MaskReg format stores one mask per 4 bits, so Compare reads data sequentially from xReg and yReg, compares and writes to MaskReg at bit position 4 * N; Select determines whether to select data from xReg or yReg based on MaskReg bit at 4 * N.
  - Invocation Implementation: Use the kernel launch operator <<<>>> to invoke the kernel function.

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
  SCENARIO=1                                                                    # Execute scenario 1
  mkdir -p build && cd build;                                                   # Create and enter build directory
  cmake -DSCENARIO_NUM=$SCENARIO -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # Build project (default npu mode)
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO                         # Generate test input data
  ./demo                                                                        # Execute the compiled program
  ```

  For CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake -DSCENARIO_NUM=1 -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  cmake -DSCENARIO_NUM=1 -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching compilation modes, clear the cmake cache by running `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Compilation Options Description

| Option | Available Values | Description |
|------|--------|------|
| `SCENARIO_NUM` | 1, 2 | Example execution scenario: Scenario 1: Compare, Scenario 2: Compares |
| `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
| `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  The following output indicates successful precision comparison:
  ```bash
  test pass!
  ```