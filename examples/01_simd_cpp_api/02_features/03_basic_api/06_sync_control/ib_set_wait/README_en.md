# IBSet and IBWait Cross-Core Synchronization Example

## Overview

This example demonstrates cross-core synchronization using IBSet and IBWait, applicable to the following scenario: when two cores cooperatively operate on the same global memory region with data dependencies, IBSet and IBWait are used to implement cross-core synchronization and avoid data read/write errors.

> **Note:** This example only applies to the programming model based on TPipe and TQue.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── ib_set_wait
│   ├── scripts
│   │   ├── gen_data.py         // Script to generate input data and golden data
│   │   └── verify_result.py    // Script to verify output data matches golden data
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── ib_set_wait.asc         // Ascend C example implementation & invocation example
```

## Example Functionality Description

This example uses two cores working cooperatively, with each core processing 256 half data elements, totaling 512 elements. Core 0 is responsible for reading input x[0:256] and y[0:256], and writing computation results to z[0:256]; Core 1 is responsible for reading Core 0's output z[0:256] and input y[256:512], and writing computation results to z[256:512]. It must be ensured that Core 1 can only read z[0:256] after Core 0 completes its write operation to avoid write-after-read data dependency issues.

In this example, IBSet and IBWait are used to implement cross-core synchronization:

- After Core 0 completes writing z[0:256], it sets a synchronization flag via IBSet
- Before reading z[0:256], Core 1 waits for Core 0 to complete its write operation via IBWait

### Example Specifications

<table>
<caption>Table 1: Example Input/Output Specifications</caption>
<tr><td rowspan="1" align="center">Example Type</td><td colspan="5" align="center">Cross-Core Synchronization</td></tr>
<tr><td rowspan="3" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center"></td></tr>
<tr><td align="center">x</td><td align="center">[512]</td><td align="center">half</td><td align="center">ND</td><td align="center"></td></tr>
<tr><td align="center">y</td><td align="center">[512]</td><td align="center">half</td><td align="center">ND</td><td align="center"></td></tr>
<tr><td rowspan="1" align="center">Example Output</td><td align="center">z</td><td align="center">[512]</td><td align="center">half</td><td align="center">ND</td><td align="center"></td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">kernel_ib_set_wait</td></tr>
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
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  The following output indicates successful precision comparison.

  ```bash
  test pass!
  ```