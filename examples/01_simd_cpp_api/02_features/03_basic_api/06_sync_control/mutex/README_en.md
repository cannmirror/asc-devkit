# Mutex Intra-Core Pipeline Synchronization Example

## Overview

This example demonstrates the usage of Mutex::Lock, Mutex::Unlock, AllocMutexID, and ReleaseMutexID intra-core pipeline synchronization interfaces. The example first obtains a MutexID from the framework via AllocMutexID, then uses Mutex::Lock and Mutex::Unlock to lock and release specified pipelines to implement synchronization dependencies between PIPE_MTE2, PIPE_V, and PIPE_MTE3 asynchronous pipelines. The example implements three tasks: data load-in, addition computation, and data load-out, using double buffering and the Mutex lock mechanism to implement pipeline synchronization control, and finally releases the MutexID using ReleaseMutexID.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── mutex
│   ├── scripts
│   │   ├── gen_data.py         // Script to generate input data and golden data
│   │   └── verify_result.py    // Script to verify output data matches golden data
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── mutex.asc               // Ascend C example implementation & invocation example
```

## Example Description

- Example Functionality:
  This example demonstrates the complete usage flow of Mutex-related interfaces:
  1. Use `AllocMutexID()` to obtain two MutexIDs (mutexId0 and mutexId1) from the framework
  2. The input data size is 1024 * 1024, which cannot fit entirely in UB, so it is processed by tile partitioning
  3. Use double buffering mechanism in the loop, alternating between two buffers (mutexId0 and mutexId1)
  4. For each tile, sequentially lock and unlock the load-in, computation, and load-out tasks
  5. After use, call `ReleaseMutexID()` to release the two MutexIDs

  Through the Mutex lock mechanism, synchronization between intra-core asynchronous pipelines is implemented, ensuring the correct execution order of the three tasks: data load-in, computation, and load-out.

- Example Specifications:
  <table border="2">
  <caption>Table 1: Example Specifications</caption>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="4" align="center">Mutex</td></tr>
  <tr><td rowspan="3" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[1024, 1024]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">[1024, 1024]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">z</td><td align="center">[1024, 1024]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">mutex_custom</td></tr>
  </table>

## Build and Run

Execute the following steps in the root directory of this example to build and run the example.
- Configure Environment Variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development toolkit on your current environment.
  - Default path, CANN package installed by root user

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, CANN package installed by non-root user

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Custom path install_path, CANN package installed

    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Example Execution

  ```bash
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;    # Build project
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                           # Execute the compiled executable program
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin  # Verify output correctness
  ```

  To use CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:

  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory and re-running cmake.

- Build Options Description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` (default) | NPU architecture: corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  The following output indicates successful precision comparison.

  ```bash
  test pass!
  ```