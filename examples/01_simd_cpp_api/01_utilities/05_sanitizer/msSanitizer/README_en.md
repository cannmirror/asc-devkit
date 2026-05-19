# msSanitizer Sample

## Overview

Developers can use the operator anomaly detection tool msSanitizer to discover and fix anomalies at an early stage, ensuring operator quality and stability. This sample uses a static Tensor programming approach for the add operator to demonstrate how this tool detects anomalies.

Please refer to the "Environment Preparation" section in [Operator Development Tools](https://www.hiascend.com/document/redirect/CannCommercialToolOpDev) for detailed installation guides and steps.

## Supported Products

- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── msSanitizer
│   ├── scripts
│   │   ├── gen_data.py             // Script to generate input data and golden data
│   │   └── verify_result.py        // Script to verify output data against golden data
│   ├── CMakeLists.txt              // Build project file
│   ├── data_utils.h                // Data read/write functions
│   └── add_custom.asc              // Ascend C operator implementation & invocation sample
```

## Operator Description

The operator implements an Add operator with a fixed shape of 72x4096.

The Add computation formula is:

```python
z = x + y
```

- x: Input, shape [72, 4096], data type float
- y: Input, shape [72, 4096], data type float
- z: Output, shape [72, 4096], data type float

## Operator Specifications Description

<table>
<tr><td rowspan="1" align="center">Operator Type (OpType)</td><td colspan="4" align="center">Add</td></tr>
<tr><td rowspan="3" align="center">Operator Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">72 * 4096</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">y</td><td align="center">72 * 4096</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Operator Output</td><td align="center">z</td><td align="center">72 * 4096</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">add_custom</td></tr>
</table>

- Operator Implementation:
  The mathematical expression for the Add operator is:

  ```
  z = x + y
  ```

  The computation logic is: The vector computation interface provided by Ascend C operates on elements in LocalTensor. Input data must first be transferred to on-chip storage, then the computation interface is used to complete the addition of two input parameters to obtain the final result, which is then transferred to external storage.

  The Add operator implementation process is divided into 3 basic tasks: CopyIn, Compute, and CopyOut. The CopyIn task is responsible for transferring input Tensors xGm and yGm from Global Memory to Local Memory, storing them in xLocal and yLocal respectively. The Compute task is responsible for performing the addition operation on xLocal and yLocal, storing the computation result in zLocal. The CopyOut task is responsible for transferring output data from zLocal to the output Tensor zGm in Global Memory.

## Anomaly Scenario Detection

This sample code is a correct implementation. Users can reproduce various anomaly scenarios as described below to experience msSanitizer's anomaly detection capabilities.

- **Memory Detection**
  - Illegal Read/Write: Anomaly caused by accessing unallocated memory.

    Users can comment out the correct DataCopy and use the incorrect DataCopy to reproduce this scenario. LocalTensor xLocal is allocated with size TILE_LENGTH, but the transfer incorrectly specifies TILE_LENGTH * 2, which is larger than xLocal's allocated size, thus triggering illegal read/write.
    ```
    // 1. correct datacopy
    AscendC::DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);
    // 2. illegal read of xGm (TILE_LENGTH*2)
    // AscendC::DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH * 2);
    ```

    Tool error message:
    ```
    ====== ERROR: illegal write of size 16384
    ======    at 0x0 on UB in add_custom
    ======    in block aiv(0-7) on device 0
    ```

  - Misaligned Access: Memory access does not meet byte alignment requirements

    Users can comment out the correct DataCopy and use the incorrect DataCopy to reproduce this scenario. In the DataCopy GM->UB transfer, the UB-side address should satisfy 32B alignment, but xLocal[5] does not satisfy 32B alignment (5 * sizeof(float) = 20), thus triggering misaligned access.
    ```
    // 1. correct datacopy
    AscendC::DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);
    // 3. misaligned access of xLocal (should be 32Byte aligned)
    // AscendC::DataCopy(xLocal[5], xGm[i * TILE_LENGTH], TILE_LENGTH);
    ```

    Tool error message:
    ```
    ====== ERROR: misaligned access of size 32
    ======    at 0x14 on UB in add_custom
    ======    in block aiv(0-7) on device 0
    ```
  - Memory Leak: Allocated memory not released after use, causing continuous memory usage increase during program execution

    Users can comment out this aclrtFree line to reproduce this scenario. Before commenting, tilingDevice is properly released; after commenting, tilingDevice is not released after use, thus triggering memory leak.

    Note: When calling mssanitizer, you need to pass --leak-check=yes to enable allocated memory leak detection.
    ```
    // 1. correct free for memory. If deleted, it will trigger memory leak check.
    aclrtFree(tilingDevice);
    ```

    Tool error message:
    ```
    ====== ERROR: LeakCheck: detected memory leaks

    ======    Direct leak of 64 byte(s)
    ======      at 0x12c0c0013000 on GM
    ======      allocated in :0 (serialNo:0)
    ```
  - Unused Allocated Memory: Anomaly caused by allocated memory not being used after allocation

    Users can comment out the correct aclrtMalloc and use the incorrect aclrtMalloc to reproduce this scenario. inputDevice[i] needs to allocate size inputsInfo[i].length, but actually allocates inputsInfo[i].length * 5, of which inputsInfo[i].length * 4 is unused, thus triggering unused allocated memory.

    Note: When calling mssanitizer, you need to pass --check-unused-memory=yes to enable unused allocated memory detection.
    ```
    // 1. correct malloc for memory
    aclrtMalloc((void **)(&inputDevice[i]), inputsInfo[i].length, ACL_MEM_MALLOC_HUGE_FIRST);
    // 2. needed inputsInfo[i].length, but malloc length * 5, therefore trigger unused memory
    // aclrtMalloc((void **)(&inputDevice[i]), inputsInfo[i].length * 5, ACL_MEM_MALLOC_HUGE_FIRST);
    ```

    Tool error message:
    ```
    ====== WARNING: Unused memory of 4718624 byte(s)
    ======    at 0x12c041200000 on GM
    ======    code in :0 (serialNo:260)
    ```

- **Race Detection**
  - Race Detection: Used to resolve memory access race issues in parallel computing environments.

    Users can comment out SetFlag and WaitFlag to reproduce this scenario. SetFlag and WaitFlag are used to ensure the timing between MTE2 GM->UB transfer and Vector compute Add. Deleting them may cause Add computation before data transfer, resulting in accuracy anomalies, thus triggering race detection.

    Note: When calling mssanitizer, you need to pass --tool=racecheck to enable race detection.
    ```
    // dependency of PIPE_MTE2 & PIPE_V caused by xLocal/yLocal in one single loop
    // If SetFlag and WaitFlag are deleted, will trigger RAW
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    ```

    Tool error message:
    ```
    ====== ERROR: Potential RAW hazard detected at UB in add_custom on device 0:
    ======    PIPE_MTE2 Write at RAW()+0x4000 in block 0 (aiv) on device 0 at pc current 0xd08 (serialNo:25)
    ======    xxxxx
    ======    PIPE_V Read at RAW()+0x4000 in block 0 (aiv) on device 0 at pc current 0x1578 (serialNo:28)
    ======    xxxxx
    ```

- **Uninitialized Detection**
  - Uninitialized Detection: Memory is in an uninitialized state after allocation, and uninitialized values are read directly without writing to the memory, causing an anomaly.

    Users can comment out the SetGlobalBuffer below to reproduce this scenario. For the device-side zGm, it is not initialized before use, thus triggering uninitialized detection.

    Note: When calling mssanitizer, you need to pass --tool=initcheck to enable race detection.
    ```
    // correct initialize of zGm.
    // If deleted, it will trigger uninitialized read
    zGm.SetGlobalBuffer((__gm__ float *)z + AscendC::GetBlockIdx() * singleCoreLength, singleCoreLength);
    ```

    Tool error message:
    ```
    ====== ERROR: uninitialized read of size 1179648
    ======    at 0x12c041600000 on GM
    ```

## Build and Run

Execute the following steps in the sample root directory to build and run the operator.
- Configure Environment Variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package in your current environment.
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

- Sample Execution
  ```bash
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;                # Build project
  python3 ../scripts/gen_data.py   # Generate test input data

  # Execute the generated executable program using mssanitizer
  # Execute the corresponding mssanitizer command based on business requirements
  mssanitizer ./demo                             # Illegal read/write / Misaligned access
  mssanitizer ./demo --leak-check=yes            # Enable memory leak detection
  mssanitizer ./demo --check-unused-memory=yes   # Enable unused allocated memory detection
  mssanitizer ./demo --tool=racecheck            # Race detection
  mssanitizer ./demo --tool=initcheck            # Uninitialized detection

  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output results
  ```

- Build Options Description

| Option | Available Values | Description |
|--------|------------------|-------------|
| `CMAKE_ASC_ARCHITECTURES` | `dav-2201` | NPU Architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products |

- Execution Result
  The execution result is shown below, indicating successful accuracy comparison.
  ```bash
  test pass!
  ```