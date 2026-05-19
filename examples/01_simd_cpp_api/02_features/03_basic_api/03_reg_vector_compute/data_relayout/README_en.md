# Data Relayout Example

## Overview
This example demonstrates data relayout functionality using the Reg programming interface, supporting multiple scenarios selected via environment variable.
    <table>
  	 	<tr>
  	 		<td>scenarioNum</td>
  	 		<td>Relayout Scenario</td>
  	 	</tr>
  	 	<tr>
  	 		<td>1</td>
  	 		<td>Interleave (interleave two uint16_t vectors)</td>
  	 	</tr>
  	 	<tr>
  	 		<td>2</td>
  	 		<td>Pack (extract low 16 bits from uint32_t vector to uint16_t vector)</td>
  	 	</tr>
  	 </table>

## Supported Products
- Ascend 950PR/Ascend 950DT

## Directory Structure
```
├── data_relayout
│   ├── scripts
│   │   ├── gen_data.py                // Input and golden data generation script
│   ├── CMakeLists.txt                 // Build project file
│   ├── data_utils.h                   // Data read/write functions
│   ├── data_relayout.asc              // Ascend C implementation & invocation example
│   └── README.md                      // Example introduction
```

## Example Description
- Example Function:
  Demonstrates the usage of data relayout interfaces (Interleave/Pack), supporting both Interleave and Pack scenarios.

  **Scenario 1: Interleave Mode**
  - Interleave two uint16_t vectors (each with 128 elements), output two uint16_t vectors
  - Interleave divides VL(128) elements into high and low halves (64 each) and interleaves separately:
    - dst0 = [src0[0], src1[0], src0[1], src1[1], ..., src0[63], src1[63]]
    - dst1 = [src0[64], src1[64], src0[65], src1[65], ..., src0[127], src1[127]]
  - Example Specifications:
    <table>
    <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="3" align="center">AIV Example</td></tr>
    <tr><td rowspan="3" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
    <tr><td align="center">x</td><td align="center">[1, 128]</td><td align="center">uint16_t</td></tr>
    <tr><td align="center">y</td><td align="center">[1, 128]</td><td align="center">uint16_t</td></tr>
    <tr><td rowspan="2" align="center">Example Output</td><td align="center">dst0</td><td align="center">[1, 128]</td><td align="center">uint16_t</td></tr>
    <tr><td align="center">dst1</td><td align="center">[1, 128]</td><td align="center">uint16_t</td></tr>
    <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">data_relayout</td></tr>
    </table>
  - Example Implementation:
    The InterleaveVF function calls the Interleave interface for data interleaving.
    - Invocation Implementation
      Use the kernel launch operator `<<<>>>` to invoke the kernel function, launching 1 core.

  **Scenario 2: Pack Mode**
  - Extract low 16 bits from one uint32_t vector (128 elements) to one uint16_t vector
  - Pack<uint16_t, uint32_t, LOWEST>: Extracts the low 16 bits of each uint32_t
  - Example Specifications:
    <table>
    <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="3" align="center">AIV Example</td></tr>
    <tr><td rowspan="2" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
    <tr><td align="center">x</td><td align="center">[1, 128]</td><td align="center">uint32_t</td></tr>
    <tr><td rowspan="1" align="center">Example Output</td><td align="center">dst</td><td align="center">[1, 128]</td><td align="center">uint16_t</td></tr>
    <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">data_relayout</td></tr>
    </table>
  - Example Implementation:
    The PackVF function calls the Pack interface to extract low 16 bits.
    - Invocation Implementation
      Use the kernel launch operator `<<<>>>` to invoke the kernel function, launching 1 core.

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
  SCENARIO=1                                                                     # Select execution scenario (1=Interleave, 2=Pack)
  mkdir -p build && cd build;                                                    # Create and enter build directory
  cmake -DSCENARIO_NUM=$SCENARIO -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;  # Build project (default npu mode)
  python3 ../scripts/gen_data.py -scenarioNum $SCENARIO                          # Generate test input data
  ./demo                                                                         # Execute the compiled program
  ```

  For CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake -DSCENARIO_NUM=$SCENARIO -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  cmake -DSCENARIO_NUM=$SCENARIO -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching compilation modes, clear the cmake cache by running `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Compilation Options Description

| Option | Available Values | Description |
|------|--------|------|
| `SCENARIO_NUM` | 1, 2 | Example execution scenario: Scenario 1=Interleave, Scenario 2=Pack |
| `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
| `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  The following output indicates successful precision comparison:
  ```bash
  test pass!
  ```