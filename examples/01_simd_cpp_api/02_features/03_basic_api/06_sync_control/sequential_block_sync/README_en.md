# Inter-core Sequential Synchronization Sample

## Overview

This sample implements inter-core sequential synchronization in deterministic computing scenarios using InitDetermineComputeWorkspace, WaitPreBlock, and NotifyNextBlock interfaces. **These three interfaces must be used together** and can ensure multiple AIV cores execute strictly in ascending blockIdx order, suitable for scenarios requiring deterministic computing. This sample simulates 8 cores performing data processing, using deterministic computing interfaces to guarantee inter-core execution order and perform atomic accumulation, ensuring deterministic computation results.

> **Note:** This sample is only applicable to programming models based on TPipe and TQue.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── sequential_block_sync
│   ├── scripts
│   │   ├── gen_data.py         // Input data and golden data generation script
│   │   └── verify_result.py    // Verification script for comparing output data with golden data
│   ├── CMakeLists.txt          // Build configuration file
│   ├── data_utils.h            // Data read/write functions
│   └── sequential_block_sync.asc      // Ascend C sample implementation & invocation sample
```

## Sample Function Description

This sample uses 8 cores working collaboratively, with each core processing 256 float data elements. It uses InitDetermineComputeWorkspace to initialize the synchronization state of GM shared memory, then ensures inter-core execution in ascending blockIdx order through WaitPreBlock and NotifyNextBlock. Each core writes input data to the output buffer through atomic accumulation in two tiles (128 elements each), guaranteeing deterministic computation results.

### Sample Specifications

<table>
<caption>Table 1: Sample Input/Output Specifications</caption>
<tr><td rowspan="3" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">[256]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">workspace</td><td align="center">[8]</td><td align="center">int32_t</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Sample Output</td><td align="center">z</td><td align="center">[256]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">kernel_sequential_block_sync</td></tr>
<tr><td rowspan="1" align="center">Number of Cores</td><td colspan="5" align="center">8</td></tr>
</table>

### Computation Flow

1. **Initialization Phase**: Call InitDetermineComputeWorkspace to initialize the synchronization state of GM shared memory
2. **Data Load**: Move 256 elements from GM to UB in two tiles (128 elements each)
3. **Inter-core Synchronization**: Wait for preceding core to complete via WaitPreBlock
4. **Atomic Accumulation**: Enable SetAtomicAdd and write data to GM via atomic accumulation
5. **Notify Succeeding Core**: Notify the succeeding core via NotifyNextBlock that it can begin execution


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
  mkdir -p build && cd build;      # Create and enter build directory
  cmake .. -DCMAKE_ASC_ARCHITECTURES=dav-2201;make -j;    # Build project
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                           # Execute the compiled program, run the sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin  # Verify output correctness
  ```

  When using CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:

  ```bash
  cmake .. -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201;make -j; # CPU debug mode
  cmake .. -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean cmake cache by running `rm CMakeCache.txt` in the build directory and re-run cmake.

- Build Options Description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products; dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  The following execution result indicates successful precision comparison.

  ```bash
  test pass!
  ```