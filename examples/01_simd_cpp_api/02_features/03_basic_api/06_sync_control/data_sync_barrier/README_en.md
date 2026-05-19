# DataSyncBarrier Example

## Overview

This example demonstrates the invocation of DataSyncBarrier. This interface blocks subsequent instruction execution until all previous memory access instructions (the memory locations to wait for can be controlled via parameters) have completed.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── data_sync_barrier
│   ├── scripts
│   │   ├── gen_data.py            // Script to generate input data and golden data
│   │   └── verify_result.py       // Script to verify output data matches golden data
│   ├── CMakeLists.txt             // Build project file
│   ├── data_utils.h               // Data read/write functions
│   └── data_sync_barrier.asc      // Ascend C example implementation & invocation example
```

## Example Specifications

<table border="2">
<caption>Table 1: Example Specifications Reference</caption>
<tr><th>Type</th><th>Name</th><th>Shape</th><th>Data Type</th><th>Format</th></tr>
<tr><td>Input</td><td>srcGm</td><td>[1, 8]</td><td>int32_t</td><td>ND</td></tr>
<tr><td>Output</td><td>dstGm</td><td>[1, 8]</td><td>int32_t</td><td>ND</td></tr>
<tr><td>Kernel Function Name</td><td colspan="4" style="text-align:center;">kernel_data_sync_barrier</td></tr>
</table>

## Example Description

The following steps describe the usage scenario of `DataSyncBarrier` in this example:

1. The system has two AIV cores, labeled core 0 and core 1. The two variables `x` and `y` in GM both have initial values of 1.
2. Core 0 first writes `x=7` to `srcGm[1]` through the scalar pipeline interface WriteGmByPassDCache.
3. Core 0 then inserts `DataSyncBarrier<AscendC::MemDsbT::DDR>()` to wait for the previous GM write operation to complete.
4. Core 0 subsequently writes `y=6` to `srcGm[0]`.
5. Core 1 continuously polls `srcGm[0]` until it reads `y=6`, then reads `srcGm[1]` and writes `2 * x` to the output.

Expected behavior:

- When core 1 reads `y=6`, `x=7` must have already been written back to GM.
- Therefore, core 1 should read `x` as 7, and the final output should be 14.

Without synchronization, the scalar pipeline does not guarantee the order of two GM writes. It is possible that `y` has been updated while `x` has not yet completed writing. In this case, even if core 1 has already seen `y=6`, it may read an incorrect `x` value.


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
  cmake .. -DCMAKE_ASC_ARCHITECTURES=dav-2201;make -j;    # Build project
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                           # Execute the compiled executable program
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin  # Verify output correctness
  ```

  To use CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:
  ```bash
  cmake .. -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201;make -j; # CPU debug mode
  cmake .. -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201;make -j; # NPU simulation mode
  ```
  > **Note:** Before switching build modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory and re-running cmake.

- Build Options Description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2/A3, dav-3510 corresponds to Ascend 950 |

- Execution Result

  The following output indicates successful precision comparison.

  ```bash
  test pass!
  ```