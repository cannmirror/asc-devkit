# Gather Class Example

## Overview

This example demonstrates data selection functionality in various scenarios using GatherMask, Gather, and Gatherb interfaces, including built-in fixed patterns, user-defined patterns, tensor offset mode, and DataBlock offset mode. It implements element selection from source operands to destination operands. The example supports switching between different scenarios via compilation parameters, helping developers understand the usage and implementation differences of Gather class interfaces.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── gather
│   ├── scripts
│   │   ├── gen_data.py         // Input data and golden data generation script
│   │   └── verify_result.py    // Verification script for comparing output data with golden data
│   ├── CMakeLists.txt          // Build configuration file
│   ├── data_utils.h            // Data read/write functions
│   └── gather.asc              // Ascend C implementation & invocation example
```

## Scenario Details

This example switches between different mask generation scenarios via the compilation parameter `SCENARIO_NUM`:

**Scenario 1: Built-in Fixed Pattern**
- Description: Uses `src1Pattern` to select the corresponding binary as mask for data retrieval
- Input: src0Local=[1, 256]
- Output: [1, 256]
- Data type: uint32_t
- Implementation:
    ```cpp
    AscendC::GatherMask(dstLocal, src0Local, src1Pattern, reduceMode, mask, gatherMaskParams, rsvdCnt);
    ```
- Parameters: dstLocal and src0Local use address reuse, with built-in fixed pattern src1Pattern=2 for element selection, reduceMode=false (Normal mode), mask=0, gatherMaskParams={1, 4, 8, 0}

**Scenario 2: User-defined Pattern**
- Description: Uses the binary representation of user-provided `src1Local` as mask for data retrieval
- Input: src0Local=[1, 256], src1Local=[1, 32]
- Output: [1, 256]
- Data type: uint32_t
- Implementation:
    ```cpp
    AscendC::GatherMask (dstLocal, src0Local, src1Local, reduceMode, mask, gatherMaskParams, rsvdCnt);
    ```
- Parameters: Uses user-provided Tensor for element selection, reduceMode=true (Counter mode), mask=70, gatherMaskParams={1, 2, 4, 0}

**Scenario 3: Tensor Offset Mode**
- Description: Performs address offset based on user-provided address offset tensor `srcOffset` for data retrieval
- Input: src0Local=[1, 128], srcOffset=[1, 128]
- Output: [1, 128]
- Data type: Input and output uint16_t, srcOffset type is uint32_t
- Implementation:
    ```cpp
    AscendC::Gather(dstLocal, src0Local, srcOffset, srcBaseOffset, count);
    ```
- Parameters: Uses user-provided srcOffset for per-element address offset, srcBaseOffset=0 indicates the starting address of source operand, count=128 indicates the number of elements to process

**Scenario 4: DataBlock Offset Mode**
- Description: Performs address offset based on user-provided address offset tensor `srcOffset` (at DataBlock granularity) for data retrieval
- Input: src0Local=[1, 128], srcOffset=[1, 8]
- Output: [1, 128]
- Data type: Input and output uint16_t, srcOffset type is uint32_t
- Implementation:
    ```cpp
    AscendC::Gatherb<T>(dstLocal, src0Local, srcOffset, repeatTime, params);
    ```
- Parameters: User-provided srcOffset contains the address offset for each datablock in the source operand, repeatTime=1 indicates the number of repeat iterations, params={1,8}

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
  python3 ../scripts/verify_result.py ./output/output.bin ./output/golden.bin $SCENARIO_NUM  # Verify output correctness
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
  | `SCENARIO_NUM` | `1` (default), `2`, `3`, `4` | Scenario number: 1 (built-in fixed pattern), 2 (user-defined pattern), 3 (tensor offset mode), 4 (DataBlock offset mode) |

- Execution result

  The following result indicates successful accuracy comparison.
  ```bash
  test pass!
  ```