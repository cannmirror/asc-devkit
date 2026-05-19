# Cast Example

## Overview

This sample demonstrates data type and precision conversion between source and destination operand tensors using Cast. The sample implements two conversion scenarios: half to int32_t and half to int4b_t.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```plain
├── cast
│   ├── scripts
│   │   ├── gen_data.py         // Script to generate input data and golden data
│   │   └── verify_result.py    // Script to verify output data against golden data
│   ├── CMakeLists.txt          // Build configuration file
│   ├── data_utils.h            // Data read/write functions
│   └── cast.asc                // Ascend C sample implementation & invocation example
```

## Sample Description

- Sample Function:  
  The CastCustom sample performs precision conversion based on the data types of source and destination operand tensors.
- Sample Specifications:  
  <table>
  <caption>Table 1: Sample Specifications</caption>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="4" align="center"> cast </td></tr>

  <tr><td rowspan="3" align="center">Sample Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 512]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">Sample Output</td></tr>
  <tr><td align="center">y</td><td align="center">[1, 512]</td><td align="center">int32_t/int4b_t</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">Kernel Name</td><td colspan="4" align="center">cast_custom</td></tr>
  </table>

- Scenario Description:  
  <table>
  <caption>Table 2: SCENARIO_NUM Parameter Description</caption>
  <tr><td align="center">SCENARIO_NUM</td><td align="center">Input Type</td><td align="center">Output Type</td><td align="center">Description</td></tr>
  <tr><td align="center">0</td><td align="center">half</td><td align="center">int4b_t</td><td align="center">half to int4b_t conversion</td></tr>
  <tr><td align="center">1</td><td align="center">half</td><td align="center">int32_t</td><td align="center">half to int32_t conversion</td></tr>
  </table>

- Sample Implementation:  
  This sample implements a CastCustom example with fixed shape: input x[1, 512], output y[1, 512], supporting two conversion scenarios: half to int32_t and half to int4b_t.

  - Kernel Implementation  
    - Call DataCopy basic API to transfer data from GM (Global Memory) to UB (Unified Buffer)
    - Call Cast interface to perform data type conversion (half to int32_t or half to int4b_t)
    - Call DataCopy basic API to transfer results from UB (Unified Buffer) to GM (Global Memory)

  - Invocation Implementation  
    Use the kernel call operator <<<>>> to invoke the kernel function.

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
  SCENARIO_NUM=0
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO_NUM  # Generate test input data
  ./demo                        # Execute the compiled executable to run the sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output results
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
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 for Atlas A2/A3 series, dav-3510 for Ascend 950PR/Ascend 950DT |
  | `SCENARIO_NUM` | `0` (default), `1` | Scenario: 0 for half to int4b_t conversion, 1 for half to int32_t conversion |

- Execution Result

  The following output indicates successful accuracy comparison.

  ```bash
  test pass!
  ```