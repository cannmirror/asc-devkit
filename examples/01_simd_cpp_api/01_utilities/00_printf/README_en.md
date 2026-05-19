# Printf Interface Function Description

## Overview

This sample demonstrates the usage of the printf interface, which prints kernel function-related information.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── 00_printf
│   ├── scripts
│   │   ├── gen_data.py         // Script to generate input data and golden data
│   │   └── verify_result.py    // Script to verify output data against golden data
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── printf.asc              // Ascend C sample implementation & invocation sample
```

## Sample Description

- Sample Function:

  Implements Matmul computation using high-level APIs, with printf interface added for formatted output.

  The Matmul computation formula is:

  ```
  C = A * B
  ```

- Sample Specifications:

  Sample parameters: M = 512, N = 1024, K = 512. Shape information is shown in the table below:
  <table>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="4" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">matmul_custom</td></tr>
  </table>

- Invocation Implementation

  Use the kernel invocation operator `<<<>>>` to call the kernel function.

## Build and Run

- Configure Environment Variables
  Execute the following steps in the sample root directory to build and run the sample.
  Select the appropriate command to configure environment variables based on the [installation method](../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package in your current environment.
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
  mkdir -p build && cd build;                                               # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;                      # Build project
  python3 ../scripts/gen_data.py                                            # Generate test input data
  ./demo                                                                    # Execute the generated executable program
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output results
  ```

- Build Options Description

| Option | Available Values | Description |
|--------|------------------|-------------|
| `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU Architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products; dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result
  The final execution result is shown below, indicating successful accuracy comparison.
  ```bash
  test pass!
  ```