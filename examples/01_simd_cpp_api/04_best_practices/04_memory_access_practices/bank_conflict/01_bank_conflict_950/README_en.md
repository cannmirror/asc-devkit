# UB Bank Conflict Optimization Example

## Overview

UB bank conflict optimization example.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── 01_bank_conflict_950
│   ├── CMakeLists.txt          // Build project file
│   ├── bank_conflict.asc       // AscendC operator implementation & invocation example
│   └── scripts
│       ├── gen_data.py         // Input data and golden data generation script
│       └── verify_result.py    // Verification script for comparing output data with golden data
```

## Operator Description

- Operator Function:
  Provides UB bank conflict optimization reference.

- Operator Specification:
  <table>
  <tr><td rowspan="1" align="center">Operator Type(OpType)</td><td colspan="4" align="center">AIV Operator</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">Operator Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">1024 * 4</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">1024 * 4</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Operator Output</td><td align="center">z</td><td align="center">1024 * 4</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">bank_conflict</td></tr>
  </table>

- Operator Implementation:
  Avoids UB bank conflicts by optimizing UB address allocation.

  - Invocation Implementation
    Uses the kernel invocation operator <<<>>> to call the kernel function.

## Build and Run

Execute the following steps in the root directory of this example to build and run the operator.

- Configure Environment Variables
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