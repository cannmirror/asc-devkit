# Compare Class Example

## Overview

This sample demonstrates data comparison functionality across multiple scenarios using Compare and Compares interfaces, implementing element-wise comparison. If the comparison result is true, the corresponding bit in the output is set to 1; otherwise, it is set to 0. Comparison results are stored in 8-bit compressed format, where every 8 comparison results are packed into one byte (uint8_t/int8_t).

The sample supports switching between different scenarios through compile parameters, helping developers understand the usage and implementation differences of Compare class interfaces.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── compare
│   ├── scripts
│   │   ├── gen_data.py         // Script to generate input data and golden data
│   │   └── verify_result.py    // Script to verify output data against golden data
│   ├── CMakeLists.txt          // Build configuration file
│   ├── data_utils.h            // Data read/write functions
│   └── compare.asc             // Ascend C sample implementation & invocation example
```

## Scenario Description

This sample switches between different scenarios through the compile parameter `SCENARIO_NUM`:

**Scenario 1: Compare**
- Description: Element-wise comparison of two tensors `src0Local` and `src1Local`
- Input: src0Local=[1, 256], src1Local=[1, 256]
- Input data type: float
- Output: dstLocal=[1, 32]
- Output data type: uint8_t
- Implementation:
  ```cpp
  AscendC::Compare(dstLocal, src0Local, src1Local, cmpMode, srcDataSize);
  ```
- Parameters: cmpMode=AscendC::CMPMODE::LT, srcDataSize=256

**Scenario 2: Compare (result stored in register)**
- Description: Element-wise comparison of two tensors `src0Local` and `src1Local`, with result stored in cmpMask register
- Input: src0Local=[1, 64], src1Local=[1, 64] 
- Input data type: float
- Output: [1, 32] 
- Output data type: uint8_t
- Implementation:
  ```cpp
    AscendC::Compare(src0Local, src1Local, cmpMode, mask, repeatParams);  // Compare interface has no repeat input, repeat defaults to 1, supporting 256 bytes of data per instruction
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::GetCmpMask(dstLocal);  // Retrieve data saved in register via GetCmpMask interface, dstLocal requires at least 128 bytes, but actual result data only occupies 8 bytes
  ```
- Parameters: cmpMode=AscendC::CMPMODE::LT; repeatParams is default value, controlling operand address stride

**Scenario 3: Compares**
- Description: Element-wise comparison between elements in `src0Local` (tensor) and `src1Scalar` (scalar)
- Input: src0Local=[1, 256], src1Local=[1, 16] where `src1Scalar` obtains one element via GetValue(idx) method for comparison
- Input data type: float
- Output: dstLocal=[1, 32]
- Output data type: uint8_t
- Data type: float
- Implementation:
    ```cpp
    AscendC::Compares(dstLocal, src0Local, src1Scalar, cmpMode, srcDataSize);
    ```
- Parameters: src1Scalar=src1Local.GetValue(0), cmpMode=AscendC::CMPMODE::LT, srcDataSize=256

**Scenario 4: Compares (flexible scalar position) — Only supported on Ascend 950PR/Ascend 950DT**
- Description: Element-wise comparison between elements in `src0Local` (tensor) and `src1Scalar` (scalar), supporting scalar before or after
- Input: src0Local=[1, 256], src1Local=[1, 16] where `src1Scalar` obtains one element via src1Local[idx] method for comparison
- Input data type: float
- Output: dstLocal=[1, 32]
- Output data type: uint8_t
- Implementation:
    ```cpp
    AscendC::Compares(dstLocal, src0Local, src1Scalar, cmpMode, srcDataSize);  // Scalar after
    AscendC::Compares(dstLocal, src1Scalar, src0Local, cmpMode, srcDataSize);  // Scalar before
    ```
- Parameters: src1Scalar=src1Local[0], cmpMode=AscendC::CMPMODE::LT, srcDataSize=256

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
  SCENARIO_NUM=1  # Set scenario number
  mkdir -p build && cd build;      # Create and enter build directory
  cmake .. -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j;    # Build project
  python3 ../scripts/gen_data.py -scenario_num=$SCENARIO_NUM   # Generate test input data
  ./demo                           # Execute the compiled executable to run the sample
  python3 ../scripts/verify_result.py ./output/output.bin ./output/golden.bin -scenario_num=$SCENARIO_NUM  # Verify output results
  ```

  For CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  Example:
  ```bash
  cmake .. -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j; # CPU debug mode
  cmake .. -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j; # NPU simulation mode
  ```
  
  > **Note:** Before switching build modes, you need to clear the cmake cache. Execute `rm CMakeCache.txt` in the build directory and run cmake again.

- Build Options

  | Option | Available Values | Description |
  |--------|------------------|-------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 for Atlas A2 Training Series/Atlas A2 Inference Series and Atlas A3 Training Series/Atlas A3 Inference Series, dav-3510 for Ascend 950PR/Ascend 950DT |
  | `SCENARIO_NUM` | `1` (default), `2`, `3`, `4` | Scenario number<br>1: Compare<br>2: Compare (result stored in register)<br>3: Compares<br>4: Compares (flexible scalar position) |

- Execution Result  
  The following output indicates successful accuracy comparison.
  ```bash
  test pass!
  ```