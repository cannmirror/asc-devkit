# Scalar Atomic Operations Sample

## Overview

This sample demonstrates the implementation flow of scalar atomic addition and scalar atomic compare-and-swap on Global Memory (GM) addresses, based on the `AtomicAdd` and `AtomicCas` interfaces. Atomic operations ensure data consistency when multiple cores access the same memory address in parallel, avoiding data race issues. Note that atomic operations involve the scalar computation unit. If there are data dependencies with the data movement unit (MTE2/MTE3), synchronization events must be inserted manually.

> **Interface Note:** In addition to the `AtomicAdd` and `AtomicCas` interfaces used in this sample, Ascend C also provides `AtomicExch`, `AtomicMax`, and `AtomicMin` interfaces. They are called in the same way as `AtomicAdd`—simply replace the function name to switch between them.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── scalar_atomic_operations
│   ├── scripts
│   │   ├── gen_data.py         // Input data and golden data generation script
│   │   └── verify_result.py    // Verification script for comparing output data with golden data
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── scalar_atomic_operations.asc  // Ascend C sample implementation & invocation sample
```

## Scenario Details

This sample uses the compile parameter `SCENARIO_NUM` to switch between different scenarios:

<table border="2">
<caption>Table 1: Scenario Configuration Reference</caption>
<tr><th>scenarioNum</th><th>Atomic Operation Interface</th><th>Input Shape</th><th>Output Shape</th><th>Data Type</th><th>Description</th></tr>
<tr><td>1</td><td>AtomicAdd</td><td>[1, 256]</td><td>[1, 256]</td><td>int32</td><td>Three cores perform atomic add 1 operation in parallel on the first element of GM</td></tr>
<tr><td>2</td><td>AtomicCas</td><td>[1, 256]</td><td>[1, 256]</td><td>uint32</td><td>Three cores perform atomic compare-and-swap in parallel on the first element of GM (if value is 1, replace with 2)</td></tr>
</table>

**Scenario 1: AtomicAdd Atomic Addition Operation**
- Input shape: src=[1, 256]
- Output shape: dst=[1, 256]
- Data type: int32
- Parameter: parallel block count = 3
- Description: Three cores are scheduled sequentially. Each core performs an atomic add 1 operation on the first element of GM (dst[0]), and the return value is the old value before the atomic operation. The final value of dst[0] is the initial value plus 3.


**Scenario 2: AtomicCas Atomic Compare-and-Swap Operation**
- Input shape: src=[1, 256]
- Output shape: dst=[1, 256]
- Data type: uint32
- Parameter: parallel block count = 3
- Description: Three cores are scheduled sequentially. Each core checks whether the first element of GM equals the expected value 1. If equal, it replaces it with the new value 2; otherwise, no modification is made. The return value is the old value before the atomic operation.

## Build and Run

Execute the following steps in the root directory of this sample to build and run it.
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

- Sample execution

  ```bash
  SCENARIO_NUM=1  # Set scenario number (values: 1, 2)
  mkdir -p build && cd build;      # Create and enter build directory
  cmake .. -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=$SCENARIO_NUM;make -j;    # Build project
  python3 ../scripts/gen_data.py -scenarioNum $SCENARIO_NUM   # Generate test input data
  ./demo                           # Execute the compiled executable program to run the sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin -scenarioNum $SCENARIO_NUM  # Verify if output results are correct
  ```

  For CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:

  ```bash
  cmake .. -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=$SCENARIO_NUM;make -j; # CPU debug mode
  cmake .. -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=$SCENARIO_NUM;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory and then re-run cmake.

- Build options description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` (default) | NPU architecture: AtomicAdd and AtomicCas only support dav-3510 (corresponding to Ascend 950PR/Ascend 950DT) |
  | `SCENARIO_NUM` | `1` (default), `2` | Scenario number: 1 (AtomicAdd atomic addition), 2 (AtomicCas atomic compare-and-swap) |

- Execution result

  The following execution result indicates successful precision comparison:

  ```bash
  test pass!
  ```