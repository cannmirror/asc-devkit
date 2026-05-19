# optimize_vf_loop Example

## Overview

Optimize VF loops through loop member variable access optimization, loop instruction distribution optimization, loop address management optimization, and other methods.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── optimize_vf_loop
│   ├── scripts
│   │   ├── gen_data.py              // Input data and golden data generation script
│   │   └── verify_result.py         // Verification script for comparing output data with golden data
│   ├── CMakeLists.txt               // Build project file
│   ├── data_utils.h                 // Data read and write functions
│   └── optimize_vf_loop.asc         // AscendC operator implementation & invocation example
```

## Operator Description

- Operator Function:
  Optimize VF loops through loop member variable access optimization, loop instruction distribution optimization, loop address management optimization, and other methods. Implements add-one operation for 1024 numbers.

- Operator Specification:
  <table>
  <tr><td rowspan="1" align="center">Operator Type(OpType)</td><td colspan="3" align="center">AIC Operator</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">Operator Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">1024</td><td align="center">float</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Operator Output</td><td align="center">y</td><td align="center">1024</td><td align="center">float</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">optimize_vf_loop</td></tr>
  </table>

- Operator Implementation:
  Optimize VF loops through loop member variable access optimization, loop instruction distribution optimization, loop address management optimization, and other methods.

  - Invocation Implementation
    Uses the kernel invocation operator <<<>>> to call the kernel function.

## Build and Run

Execute the following steps in the root directory of this example to build and run the operator.

- Configure Environment Variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on your current environment.

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

- Execute Example
  ```bash
  mkdir -p build && cd build;                                               # Create and enter build directory
  cmake ..;make -j;                                                         # Build project
  python3 ../scripts/gen_data.py                                            # Generate test input data
  ./demo                                                                    # Execute the compiled executable program to run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output result correctness and confirm algorithm logic
  ```
  The following output indicates successful precision comparison:
  ```bash
  test pass!
  ```