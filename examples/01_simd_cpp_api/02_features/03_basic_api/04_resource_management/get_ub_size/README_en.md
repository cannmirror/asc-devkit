# GetUBSize Example

## Overview

This example demonstrates the usage of GetUBSizeInBytes and GetRuntimeUBSize interfaces, which are used to obtain the maximum user-available UB (Unified Buffer) size in bytes. This example provides 2 different test scenarios.

<table>
  <tr>
    <td align="center">scenarioNum</td>
    <td align="center">API Interface</td>
    <td align="center">Description</td>
  </tr>
  <tr>
    <td align="center">1</td>
    <td align="center">GetUBSizeInBytes</td>
    <td>Returns a compile-time constant representing the maximum user-available UB (Unified Buffer) size. For example, in Ascend 950PR/Ascend 950DT scenario, the system reserves 8KB, total UB is 256KB, returning 248KB</td>
  </tr>
  <tr>
    <td align="center">2</td>
    <td align="center">GetRuntimeUBSize</td>
    <td>Returns a runtime variable representing the maximum user-available UB (Unified Buffer) size. Suitable for SIMT and SIMD hybrid programming scenarios. In SIMT scenarios, a portion of UB space is reserved for Dcache. For example, in Ascend 950PR/Ascend 950DT scenario, SIMT programming allocates 32KB for Dcache, system reserves 8KB, total UB is 256KB, returning 216KB</td>
  </tr>
</table>

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── get_ub_size
│   ├── scripts
│   │   ├── gen_data.py         // Input data and ground truth generation script
│   │   └── verify_result.py    // Verification script for checking output data against ground truth
│   ├── CMakeLists.txt          // Build configuration file
│   ├── data_utils.h            // Data read/write functions
│   └── get_ub_size.asc         // Ascend C example implementation & invocation example
```

## Example Description

- Example functionality

  The example demonstrates functionality based on the Abs absolute value operation with the calculation formula:

  ```
  z = Abs(x)
  ```

- Example specifications

  **Scenario 1: GetUBSizeInBytes**

  <table>
    <tr>
      <td align="center">Category</td>
      <td align="center">name</td>
      <td align="center">shape</td>
      <td align="center">data type</td>
      <td align="center">format</td>
    </tr>
    <tr>
      <td rowspan="1" align="center">Example Input</td>
      <td align="center">x</td>
      <td align="center">[1, 16384]</td>
      <td align="center">half</td>
      <td align="center">ND</td>
    </tr>
    <tr>
      <td align="center">Example Output</td>
      <td align="center">z</td>
      <td align="center">[1, 16384]</td>
      <td align="center">half</td>
      <td align="center">ND</td>
    </tr>
  </table>

  **Scenario 2: GetRuntimeUBSize**

  <table>
    <tr>
      <td align="center">Category</td>
      <td align="center">name</td>
      <td align="center">shape</td>
      <td align="center">data type</td>
      <td align="center">format</td>
    </tr>
    <tr>
      <td rowspan="1" align="center">Example Input</td>
      <td align="center">x</td>
      <td align="center">[1, 126976]</td>
      <td align="center">half</td>
      <td align="center">ND</td>
    </tr>
    <tr>
      <td align="center">Example Output</td>
      <td align="center">z</td>
      <td align="center">[1, 126976]</td>
      <td align="center">half</td>
      <td align="center">ND</td>
    </tr>
  </table>

- Example implementation

  - Kernel implementation

    - Calls GetUBSizeInBytes or GetRuntimeUBSize interface to obtain the available UB (Unified Buffer) size for calculating tileLength.

    - Calls DataCopy basic API to transfer data from GM (Global Memory) to UB (Unified Buffer).

    - Calls Abs interface to perform absolute value operation on the input tensor.

    - Calls DataCopy basic API to transfer the computation result from UB (Unified Buffer) to GM (Global Memory).

  - Invocation implementation

    Uses the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this example to build and run the example.

- Configure environment variables

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

- Execute the example

  ```bash
  SCENARIO=1
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DSCENARIO_NUM=$SCENARIO -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO   # Generate test input data
  ./demo                           # Execute the compiled executable to run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output result correctness, confirm algorithm logic
  ```

  The following execution result indicates successful precision comparison:

  ```bash
  test pass!
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:

  ```bash
  cmake -DSCENARIO_NUM=$SCENARIO -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  cmake -DSCENARIO_NUM=$SCENARIO -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clear the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build options description

  | Option | Possible Values | Description |
  | ----------------| -----------------------------| --------------------------------------------------------------------------------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` (default) | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  | `SCENARIO_NUM` | `1`, `2` | Scenario number: 1=GetUBSizeInBytes, 2=GetRuntimeUBSize |

- Execution result

  The following execution result indicates successful precision comparison:

  ```bash
  test pass!
  ```