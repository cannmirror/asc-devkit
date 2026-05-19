# LoadData3DV2 Data Movement Example in Matrix Multiplication

## Overview

This example introduces the usage scenarios and methods of the LoadData3DV2 instruction in matrix multiplication. LoadData3DV2 can move two-dimensional matrices A and B from L1 to L0A/L0B, where A and B respectively represent the left and right input matrices of matrix multiplication.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── mmad_load3dv2
│   ├── scripts
│   │   ├── gen_data.py             // Input data and golden data generation script
│   │   └── verify_result.py        // Verification script for output data and golden data
│   ├── CMakeLists.txt              // Build project file
│   ├── data_utils.h                // Data read/write functions
│   └── mmad_load3dv2.asc           // Ascend C operator implementation & calling example
```

## Operator Description

The LoadData3DV2 instruction is referred to as load3dv2. The transposition capability and supported data types of this instruction for two-dimensional matrices are related to the storage location of the destination address. Specifically:

(1) When the destination address is on L0A, supported data types are: uint8_t/int8_t/half/bfloat16_t/uint32_t/int32_t/float/int4b_t.
    When the destination address is on L0B, supported data types are: half/bfloat16_t/uint32_t/int32_t/float.

(2) When the destination address is on L0A, enTranspose can determine whether to enable the transposition function.
    When the destination address is on L0B, the transposition function is enabled by default (even when enTranspose=false, the transposition function is still enabled).

Since this example does not support int4b_t input data type, this example demonstrates the following five uses of load3dv2 in matrix multiplication:

### Load3DV2 Interface Scenario Reference Table

| scenarioNum | Input Data Type | Matrix A Transposed | Matrix B Transposed |
| --- | --- | --- | --- |
| 1 | half | Not transposed | Not transposed |
| 2 | half | Transposed | Not transposed |
| 3 | float | Not transposed | Not transposed |
| 4 | float | Transposed | Not transposed |
| 5 | int8_t | Not transposed | Transposed |

Note: When input data type is B8 and destination address is on L0B, the load3dv2 instruction is not supported. Therefore, when scenarioNum=5, SplitB calls the load2d instruction.

In this example, scenarioNum=3 and 4 have matrix A consistent with scenarioNum=12 and 13 in the example [load_data_l12l0](./load_data_l12l0/README.md), and matrix B consistent with scenarioNum=13 in that example. Therefore, the specific parameter configuration and diagrams for the load3dv2 instruction can refer to the introduction of the Load3DV2 interface in the "3. L1 to L0" section of that example's readme.

Since different input data types have little impact on the configuration parameters of the load3dv2 instruction, other scenarios in this example can refer to scenarioNum=3 and 4.

## Build and Run

Execute the following steps in the root directory of this example to build and run the operator.

- Configure environment variables

  Please select the corresponding environment variable configuration command based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on the current environment.

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

- Example execution
  ```bash
  SCENARIO=4 M=30 K=40 N=70
  mkdir -p build && cd build;      # Create and enter build directory
  cmake .. -DSCENARIO_NUM=$SCENARIO -DM_SIZE=$M -DK_SIZE=$K -DN_SIZE=$N;make -j;    # Build project
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO -m=$M -k=$K -n=$N   # Generate test input data
  ./demo                           # Execute the compiled executable program, run the example
  python3 ../scripts/verify_result.py -scenarioNum=$SCENARIO output/output.bin output/golden.bin   # Verify output result correctness, confirm algorithm logic is correct
  ```

  The execution result is shown below, indicating the precision comparison passed.
  ```bash
  test pass!
  ```