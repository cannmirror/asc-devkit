# DataCopy with Atomic Operations Sample

## Overview

This sample introduces the implementation flow of atomic accumulation and atomic maximum comparison when data is transferred from VECOUT to GM, based on `SetAtomicAdd` and `SetAtomicMax` atomic operation interfaces. Note that after calling atomic operation interfaces to complete related operations, you must call `DisableDmaAtomic()` to disable atomic mode to prevent affecting other subsequent computations.

> **Interface Note:** In addition to the `SetAtomicAdd` and `SetAtomicMax` interfaces used in this sample, Ascend C also provides the `SetAtomicMin` interface for configuring VECOUT-to-GM transfer rules. The calling method for `SetAtomicMin` is the same as `SetAtomicMax`; simply replace the function name to switch.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── data_movement_with_atomic_operations
│   ├── scripts
│   │   ├── gen_data.py         // Input data and golden data generation script
│   │   └── verify_result.py    // Verification script for comparing output data with golden data
│   ├── CMakeLists.txt          // Build configuration file
│   ├── data_utils.h            // Data read/write functions
│   └── data_movement_with_atomic_operations.asc  // Ascend C sample implementation & invocation sample
```

## Scenario Details

This sample switches between different scenarios via the compilation parameter `SCENARIO_NUM`:

<table border="2">
<caption>Table 1: Scenario Configuration Reference</caption>
<tr><th>scenarioNum</th><th>Atomic Operation Interface</th><th>Input Shape</th><th>Output Shape</th><th>Data Type</th><th>Description</th></tr>
<tr><td>1</td><td>SetAtomicAdd</td><td>[1, 256] (three cores read same)</td><td>[1, 256]</td><td>half</td><td>Three cores read same input, atomic accumulation operation</td></tr>
<tr><td>2</td><td>SetAtomicMax</td><td>[1, 256]×3 (three cores read different)</td><td>[1, 256]</td><td>half</td><td>Three cores read different inputs, atomic maximum comparison operation</td></tr>
</table>

**Scenario 1: SetAtomicAdd Atomic Accumulation Operation (three cores read same input)**
- Input shape: src=[1, 256] (three cores simultaneously read input_x.bin), dst=[1, 256] (input input_y.bin represents existing data on dst)
- Output shape: dst=[1, 256]
- Data type: half
- Description: Three cores simultaneously read the same input data (input_x.bin), enable atomic accumulation mode via `SetAtomicAdd`, and accumulate their respective data to the shared output buffer. Result is input_y + input_x*3


**Scenario 2: SetAtomicMax Atomic Maximum Comparison Operation (three cores read different inputs)**
- Input shape: src0=[1, 256], src1=[1, 256], src2=[1, 256] (three cores read input_x0.bin, input_x1.bin, input_x2.bin respectively)
- Output shape: dst=[1, 256]
- Data type: half
- Description: Three cores obtain their own index via `GetBlockIdx()`, read different input data (input_x0.bin, input_x1.bin, input_x2.bin), enable atomic maximum comparison mode via `SetAtomicMax`. Each position outputs the maximum value among the three inputs

## Sample Description

- Sample Specifications:
  <table>
  <caption>Table 2: Sample Specifications</caption>
  <tr><td rowspan="2" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td rowspan="1" align="center">src</td><td align="center">[1, 256]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">dst</td><td align="center">[1, 256]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">data_movement_with_atomic_operations_custom</td></tr>
  <tr><td rowspan="1" align="center">Parallel Block Count</td><td colspan="4" align="center">3</td></tr>
  </table>

## Build and Run

Execute the following steps in the sample root directory to build and run the sample.

- Configure Environment Variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on your current environment.
  - Default path, root user installed CANN package

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, non-root user installed CANN package

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, installed CANN package

    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Sample Execution

  ```bash
  SCENARIO_NUM=1  # Set scenario number (values: 1, 2)
  mkdir -p build && cd build;      # Create and enter build directory
  cmake .. -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j;    # Build project
  python3 ../scripts/gen_data.py -scenarioNum $SCENARIO_NUM   # Generate test input data
  ./demo                           # Execute the compiled program, run the sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin  # Verify output correctness
  ```

  When using CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:

  ```bash
  cmake .. -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j; # CPU debug mode
  cmake .. -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean cmake cache by running `rm CMakeCache.txt` in the build directory and re-run cmake.

- Build Options Description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products; dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  | `SCENARIO_NUM` | `1` (default), `2` | Scenario number: 1 (SetAtomicAdd atomic accumulation), 2 (SetAtomicMax atomic maximum comparison) |

- Execution Result

  The following execution result indicates successful precision comparison:

  ```bash
  test pass!
  ```