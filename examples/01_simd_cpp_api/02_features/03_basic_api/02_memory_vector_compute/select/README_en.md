# Select Class Example

## Overview

This example implements data selection functionality in multiple scenarios using the Select interface, selecting elements from two vectors or between a vector and a scalar based on the `selMask` mask and writing them to a destination vector. The selection rule is: when a bit in selMask is 1, select from src0; when a bit is 0, select from src1.

The example supports switching between different scenarios via compilation parameters, helping developers understand the usage and implementation differences of the Select interface.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── select
│   ├── scripts
│   │   ├── gen_data.py         // Input data and golden data generation script
│   │   └── verify_result.py    // Verification script for comparing output data with golden data
│   ├── CMakeLists.txt          // Build configuration file
│   ├── data_utils.h            // Data read/write functions
│   └── select.asc              // Ascend C implementation & invocation example
```

## Example Description

- Scenario Details

  **Scenario 1: Select**

  - Description: Selects elements from two tensors based on selMask. In each iteration, the selection operation is performed based on the valid bits of selMask (limited to 256/sizeof(T) valid bits, where T is the input data type; in this example T is float).
  - Implementation:
    ```cpp
    // cmpMode = AscendC::SELMODE::VSEL_CMPMASK_SPR
    // Both src0 and src1 are tensors. When selMask bit is 1, select from src0; when bit is 0, select from src1
    AscendC::Select(dstl, selMask, src0, src1, cmpMode, count);
    ```
  
  **Scenario 2: Select (Scalar)**

  - Description: Selects elements from one tensor and one scalar based on selMask, with no valid data limit on selMask. In multiple iterations, each iteration continuously uses a different portion of selMask.
  - Implementation:
    ```cpp
    // cmpMode = AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE
    // src0 is a tensor, src1 is a scalar. When selMask bit is 1, select from src0; when bit is 0, equal to src1
    AscendC::Select(dst, selMask, src0, src1, cmpMode, count);
    ```

  **Scenario 3: Select (selMask without valid bit limit)**

  - Description: Selects elements from two tensors based on selMask, with no valid data limit on selMask. In multiple iterations, each iteration continuously uses a different portion of selMask.
  - Implementation:
    ```cpp
    // cmpMode = AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE
    // Both src0 and src1 are tensors. When selMask bit is 1, select from src0; when bit is 0, select from src1
    AscendC::Select(dst, selMask, src0, src1, cmpMode, count);
    ```

**Scenario 4: Select (Flexible Scalar Position) — Only supported on Ascend 950PR/Ascend 950DT**

  - Description: Similar functionality to Scenario 2, but with more flexible scalar position.
  - Implementation:
    ```cpp
    // cmpMode = AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE
    // Scalar at the end. src0 is a tensor, src1 is a scalar. When selMask bit is 1, select from src0; when bit is 0, equal to src1
    static constexpr AscendC::BinaryConfig config = { 1 };
    AscendC::Select<AscendC::BinaryDefaultType, uint8_t, config>(dst, selMask, src0, src1, cmpMode, count);

    // Scalar at the beginning. src0 is a scalar, src1 is a tensor. When selMask bit is 1, equal to src0; when bit is 0, select from src1
    static constexpr AscendC::BinaryConfig config = { 0 };
    AscendC::Select<AscendC::BinaryDefaultType, uint8_t, config>(dst, selMask, src0, src1, cmpMode, count); 
    ```

- Scenario Specifications

  The example can switch between different scenarios via the compilation parameter `SCENARIO_NUM`. Parameters are shown in the table below:

  | Scenario Number | Scenario Name | src0 shape | src0 Data Type | src1 shape | src1 Data Type | selMask shape | selMask Data Type | dst shape | dst Data Type |
  |------|------|------|------|------|------|------|------|------|------|
  | 1 | Select | [1, 256] | float | [1, 256] | float | [1, 8] valid bit limit | uint8_t |[1, 256] | float |
  | 2 | Select (Scalar) | [1, 256] | float | scalar | float | [1, 32] | uint8_t |[1, 256] | float |
  | 3 | Select (selMask without valid bit limit) | [1, 256] | float | [1, 256] | float | [1, 32] | uint8_t |[1, 256] | float |
  | 4 | Select (Flexible Scalar Position) | [1, 256] | float | scalar | float | [1, 32] | uint8_t |[1, 256] | float |

## Build and Run

Execute the following steps in the example root directory to build and run the example.

- Configure environment variables  
  Select the appropriate command based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development toolkit in your current environment.
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
    
- Example execution
  ```bash
  SCENARIO_NUM=1  # Set scenario number
  mkdir -p build && cd build;      # Create and enter build directory
  cmake .. -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j;    # Build project
  python3 ../scripts/gen_data.py -scenario_num=$SCENARIO_NUM   # Generate test input data
  ./demo                           # Execute the compiled program to run the example
  python3 ../scripts/verify_result.py ./output/output.bin ./output/golden.bin  # Verify output correctness
  ```

  When using CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  Example:
  ```bash
  cmake .. -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j; # CPU debug mode
  cmake .. -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j; # NPU simulation mode
  ```
  
  > **Note:** Before switching build modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Build options description

  | Option | Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  | `SCENARIO_NUM` | `1` (default), `2`, `3`, `4` | Scenario number: 1 corresponds to Select, 2 corresponds to Select (Scalar), 3 corresponds to Select (selMask without valid bit limit), 4 corresponds to Select (Flexible Scalar Position) |

- Execution result

  The following result indicates successful accuracy comparison.
  ```bash
  test pass!
  ```