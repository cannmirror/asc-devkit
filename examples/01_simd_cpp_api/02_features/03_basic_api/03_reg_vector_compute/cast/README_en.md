# Cast Example

## Overview
This example demonstrates Cast operation using the Reg programming interface, primarily calling the Cast interface for data type conversion.
This example supports two data type conversion scenarios, selected via the environment variable SCENARIO_NUM.
  <table>
    <tr>
      <td>SCENARIO_NUM</td>
        <td>Data Type Conversion</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Smaller to larger bit-width, using half to int32_t as example</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Larger to smaller bit-width, using float to int16_t as example</td>
    </tr>
    </table>

## Supported Products
- Ascend 950PR/Ascend 950DT

## Directory Structure
```
├── cast
│   ├── scripts
│   │   ├── gen_data.py                // Input and golden data generation script
│   ├── CMakeLists.txt                 // Build project file
│   ├── data_utils.h                   // Data read/write functions
│   ├── cast.asc                       // Ascend C implementation & invocation example
│   └── README.md                      // Example introduction
```

## Example Description
This example performs data type conversion on input vectors. When input and output data types have different bit-widths, the Cast interface reads or writes at intervals. Therefore, this example uses corresponding compression or decompression modes during data load or store. LoadAlign/StoreAlign interface usage is for reference only. Details are as follows:

**Scenario 1: Smaller to Larger Bit-Width**
- Example Function: Converts half type data to int32_t type data.
- Parameter Description:
  - layoutMode = RegLayout::ZERO: Cast interface reads data from xReg at position 2\*N+0, used with LoadAlign interface to sequentially load input data to xReg at position 2\*N
  - satMode = SatMode::NO_SAT: This scenario demonstrates non-saturation mode for float-to-int conversion. When input data exceeds the output data type range, the result is truncated to the output data type width. For example, input half value 4294967297.0 corresponds to integer 4294967297 (0x100000001), takes the lower 32 bits, output int32_t value is 1
  - roundMode = RoundMode::CAST_FLOOR: This scenario demonstrates floor (round down) rounding mode. For example, input half value 2.5 outputs int32 value 2
  - mask: In Cast interface, mask filters based on the larger bit-width data type between input and output, so this scenario generates MaskReg based on int32_t data type
- Example Specifications:
  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="3" align="center">AIV Example</td></tr>
  <tr><td rowspan="2" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 256]</td><td align="center">half</td></tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">y</td><td align="center">[1, 256]</td><td align="center">int32_t</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">cast</td></tr>
  </table>
- Example Implementation:
  Cast from half to int32_t processes 64 elements at a time, with the following flow and diagram:
  - Load: Call LoadAlign interface with decompression mode to load data to position 2\*N, while setting position 2\*N+1 to 0
  - Compute: Call Cast interface with input/output bit-width ratio 1:2, so read data from xReg position 2\*N, after type conversion write sequentially to yReg
  - Store: Call StoreAlign interface for normal store
  - Invocation Implementation: Use the kernel launch operator <<<>>> to invoke the kernel function.
  <img src="figure/reg_cast_1.png">

**Scenario 2: Larger to Smaller Bit-Width**
- Example Function: Converts float type data to int16_t type data.
- Parameter Description:
  - layoutMode = RegLayout::ZERO: Cast interface writes data to yReg at position 2\*N+0, used with StoreAlign interface to store data from yReg position 2\*N
  - satMode = SatMode::SAT: This scenario demonstrates saturation mode for float-to-int conversion. When input data exceeds the output data type range, the result is clamped to the corresponding extreme value of the output type. For example, input float value 32768.0 corresponds to integer 32768, takes the maximum value of int16_t, output int16_t value is 32767
  - roundMode = RoundMode::CAST_ROUND: This scenario demonstrates round (round half up) rounding mode. For example, input half value 2.5 outputs int32 value 3
  - mask: In Cast interface, mask filters based on the larger bit-width data type between input and output, so this scenario generates MaskReg based on float data type
- Example Specifications:
  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="3" align="center">AIV Example</td></tr>
  <tr><td rowspan="2" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 256]</td><td align="center">float</td></tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">y</td><td align="center">[1, 256]</td><td align="center">int16_t</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">cast</td></tr>
  </table>
- Example Implementation:
  Cast from float to int16_t processes 64 elements at a time, with the following flow and diagram:
  - Load: Call LoadAlign interface for normal load
  - Compute: Call Cast interface with input/output bit-width ratio 2:1, so read data sequentially from xReg, after type conversion write to yReg position 2\*N, while setting position 2\*N+1 to 0
  - Store: Call StoreAlign interface with compression mode to store only data at position 2\*N
  - Invocation Implementation: Use the kernel launch operator <<<>>> to invoke the kernel function.
  <img src="figure/reg_cast_2.png">

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
  SCENARIO=1                                                                     # Execute scenario 1
  mkdir -p build && cd build;                                                    # Create and enter build directory
  cmake -DSCENARIO_NUM=$SCENARIO -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;  # Build project (default npu mode)
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO                          # Generate test input data
  ./demo                                                                         # Execute the compiled program
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
| `SCENARIO_NUM` | 1, 2 | Example execution scenario: Scenario 1: Smaller to larger bit-width, Scenario 2: Larger to smaller bit-width |
| `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
| `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result
  The following output indicates successful precision comparison:
  ```bash
  test pass!
  ```