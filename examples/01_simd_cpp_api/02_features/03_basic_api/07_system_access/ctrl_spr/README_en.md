# CtrlSpr Sample

## Overview

This sample demonstrates setting, reading, and resetting specific bit fields of the CTRL register (control register) using `SetCtrlSpr`, `GetCtrlSpr`, and `ResetCtrlSpr` interfaces, and verifies whether non-saturation mode is working correctly through floating-point computation.

This sample demonstrates setting, reading, and resetting CTRL[48] and CTRL[60] bit fields, verifying whether INF maintains its original value in non-saturation mode. It also verifies by reading CTRL[48] values after setting and after reset using GetCtrlSpr:
- CTRL[60] controls the global enable method for saturation mode. When set to 1, global saturation setting is enabled.
- CTRL[48] controls the saturation mode for floating-point computation and floating-point precision conversion, effective only when CTRL[60] is enabled.
  - When set to 0 (saturation mode): INF output is saturated to ±MAX (65504 for half type), NaN output is saturated to 0.
  - When set to 1 (non-saturation mode): INF/NaN maintains original output.
- After using the register, use ResetCtrlSpr interface to reset the register to default values to prevent affecting subsequent computations.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── ctrl_spr
│   ├── scripts
│   │   ├── gen_data.py         // Input data and golden data generation script
│   │   └── verify_result.py    // Verification script for comparing output data with golden data
│   ├── CMakeLists.txt          // Build configuration file
│   ├── data_utils.h            // Data read/write functions
│   └── ctrl_spr.asc            // Ascend C sample implementation & invocation sample
```

## Sample Description

- Sample Function:  
  Verifies the complete workflow and functional effects of CTRL register non-saturation mode operations:
  1. SetCtrlSpr: Set `CTRL[60]=1` to enable global effect, set `CTRL[48]=1` to select non-saturation mode (INF/NAN maintains original output)
  2. GetCtrlSpr: Read the value of `CTRL[48]` after setting, store in the first 4 bits of `ctrlLocal`
  3. Adds: Execute half-type floating-point addition computation with input containing INF values, store in `dstLocal`, used to verify whether INF maintains original value in non-saturation mode
  4. ResetCtrlSpr: Reset `CTRL[48]` and `CTRL[60]` to default values
  5. GetCtrlSpr: Read the value of `CTRL[48]` after reset, store in the last 4 bits of `ctrlLocal`

- Sample Specifications:
  <table>
  <caption>Table 1: Sample Specifications</caption>
  <tr><td rowspan="1" align="center">Number of Cores</td><td colspan="4" align="center">1</td></tr>
  <tr><td rowspan="2" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src</td><td align="center">[1, 256]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">Sample Output</td><td align="center">output_ctrl</td><td align="center">[1, 8]</td><td align="center">int64_t</td><td align="center">ND</td></tr>
  <tr><td align="center">output_sat</td><td align="center">[1, 256]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">kernel_ctrl_spr</td></tr>  
  </table>

- Output Description:
  - output_ctrl.bin: Register value verification output, 8 int64_t values (first 4 are the value after `CTRL[48]` setting=1, last 4 are the value after `CTRL[48]` reset=0)
  - output_sat.bin: Non-saturation mode functional verification output, first 128 are normal values+1, last 128 are INF (INF maintained in non-saturation mode)

- Verification Logic:
  - Input data: First 128 are normal values (0~127), last 128 are INF
  - Set `CTRL[48]=1` (non-saturation mode): INF + 1 = INF (INF maintained)
  - `CTRL[48]` value after setting=1, value after reset=0


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
  mkdir -p build && cd build;                                          # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;                 # Build project, default npu mode
  python3 ../scripts/gen_data.py                                       # Generate test input data and golden data
  ./demo                                                               # Execute the compiled program, run the sample
  python3 ../scripts/verify_result.py                                  # Verify output correctness
  ```

  When using CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  Examples:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;   # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;   # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean cmake cache by running `rm CMakeCache.txt` in the build directory and re-run cmake.

- Build Options Description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` (default) | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

  > **Note:** This sample is only supported on Ascend 950PR/Ascend 950DT.

- Execution Result

  The following execution result indicates successful precision comparison:
  ```bash
  test pass!
  ```