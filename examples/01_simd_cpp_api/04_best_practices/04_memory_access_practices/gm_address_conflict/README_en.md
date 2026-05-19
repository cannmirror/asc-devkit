# Same-Address Conflict Operator Direct Call Sample

## Overview

This sample introduces the impact of same-address conflicts and two solutions, providing kernel direct call methods.

## Supported Products
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure
```
├── gm_address_conflict
│   ├── scripts
│   │   ├── gen_data.py                    // Input data and golden data generation script
│   │   └── verify_result.py               // Verification script for output data and golden data
│   ├── CMakeLists.txt                     // Build project file
│   ├── data_utils.h                       // Data read/write functions
│   └── gm_address_conflict.asc            // Ascend C operator implementation & call sample
```

## Operator Description
- Operator Function:
  The Adds operator implements adding a tensor with a scalar value of 2.0 and returns the result.

  The corresponding mathematical expression is:

  ```python
  z = x + 2.0
  ```

  - x: input, shape [8192, 128], data type float
  - z: output, shape [8192, 128], data type float

- Operator Specification:

<table>
<tr><td rowspan="1" align="center">Operator Type(OpType)</td><td colspan="4" align="center">Adds</td></tr>
<tr><td rowspan="2" align="center">Operator Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">8192 * 128</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Operator Output</td><td align="center">z</td><td align="center">8192 * 128</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">adds_custom_v1/adds_custom_v2/adds_custom_v3</td></tr>
</table>

- Operator Implementation:

  - Kernel Implementation
   
    The computation logic is: This sample mainly introduces the impact of same-address conflicts on data copy efficiency in Global Memory data access. Data access requests (read/write) in the AI processor are address-aligned to 512 Bytes internally. When data access requests from multiple cores fall within consecutive 512 Bytes range after conversion at the same time, the AI processor serializes requests falling within the same 512 Bytes range for data consistency requirements, resulting in reduced copy efficiency, which is the same-address access phenomenon.

    This sample has 3 implementation versions:
    adds_custom_v1: Basic implementation version, each core has the same computation order, same-address conflicts exist, bandwidth efficiency is poor.
    adds_custom_v2: Avoids same-address conflicts by adjusting each core's computation order.
    adds_custom_v3: Avoids same-address conflicts by adjusting the partition order.

    The current operator execution mechanism ensures that user kernel input parameters (including workspace/tiling) addresses are 512 Bytes aligned. Therefore, users only need to determine based on the address offset whether two addresses will fall within consecutive 512 Bytes range.

## Build and Run
Execute the following steps in the sample root directory to build and run the operator.
- Configure Environment Variables
  Please select the corresponding environment variable configuration command based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on the current environment.
  - Default path, root user installed CANN software package
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, non-root user installed CANN software package
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, installed CANN software package
    ```bash
    source ${install_path}/cann/set_env.sh
    ```
    
- Sample Execution
  ```bash
  mkdir -p build && cd build;   # Create and enter build directory
  cmake ..;make -j;             # Build project
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                        # Execute the compiled executable program, run the sample
  python3 ../scripts/verify_result.py output/output_z_1.bin output/golden.bin   # Verify output result correctness, confirm algorithm logic is correct
  python3 ../scripts/verify_result.py output/output_z_2.bin output/golden.bin   # Verify output result correctness, confirm algorithm logic is correct
  python3 ../scripts/verify_result.py output/output_z_3.bin output/golden.bin   # Verify output result correctness, confirm algorithm logic is correct
  ```
  The execution result is as follows, indicating precision comparison succeeded.
  ```bash
  test pass!
  ```
