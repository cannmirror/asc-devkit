# Bank Conflict Operator Direct Invocation Example

## Overview

This example introduces the implementation of bank conflict optimization based on the Add operator and provides a kernel function direct invocation method.

## Supported Products

- Atlas A3 training series products/Atlas A3 inference series products
- Atlas A2 training series products/Atlas A2 inference series products

## Directory Structure

```
├── bank_conflict
│   ├── scripts
│   │   ├── gen_data.py                    // Input data and golden data generation script
│   │   └── verify_result.py               // Verification script for comparing output data with golden data
│   ├── CMakeLists.txt                     // Build project file
│   ├── data_utils.h                       // Data read and write functions
│   └── bank_conflict.asc                     // Ascend C operator implementation & invocation example
```

## Operator Description

- Operator Function:
  The operator implements an Add operator with a fixed shape of 1×4096.

  The Add calculation formula is:

  ```python
  z = x + y
  ```

  - x: input, shape [1, 4096], data type float
  - y: input, shape [1, 4096], data type float
  - z: output, shape [1, 4096], data type float

- Operator Specification:

  <table>
  <tr><td rowspan="1" align="center">Operator Type(OpType)</td><td colspan="4" align="center">Add</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">Operator Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">1 * 4096</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">1 * 4096</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Operator Output</td><td align="center">z</td><td align="center">1 * 4096</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">add_custom_v1 / add_custom_v2</td></tr>
  </table>


- Operator Implementation:
  This example implements an Add operator with a fixed shape of 1*4096.

  - Kernel Implementation
    The mathematical expression for the Add operator is:

    ```
    z = x + y
    ```

    The computation logic is: the vector computation interface provided by Ascend C operates on elements stored in LocalTensor. Input data must first be moved to on-chip memory, then computation interfaces are used to add the two input parameters to obtain the final result, which is then moved to external memory.

    The Add operator implementation flow is divided into 3 basic tasks: CopyIn, Compute, and CopyOut. The CopyIn task moves input tensors xGm and yGm from Global Memory to Local Memory, storing them in xLocal and yLocal respectively. The Compute task performs the addition operation on xLocal and yLocal, storing the result in zLocal. The CopyOut task moves the output data from zLocal to the output tensor zGm in Global Memory.

    Implementation 1: xLocal address is 0, yLocal address is 0x4000, zLocal address is 0x8000. There is a read-read conflict between xLocal and yLocal, and a read-write conflict between xLocal and zLocal.
    Implementation 2: To avoid Bank conflicts, the Tensor address is adjusted by configuring bufferSize during InitBuffer. xLocal address is 0, yLocal address is 0x4100, zLocal address is 0x10000.


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
  mkdir -p build && cd build;   # Create and enter build directory
  cmake ..;make -j;             # Build project
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                        # Execute the compiled executable program to run the example
  python3 ../scripts/verify_result.py output/output_z_v1.bin output/golden.bin   # Verify output result correctness and confirm algorithm logic
  python3 ../scripts/verify_result.py output/output_z_v2.bin output/golden.bin   # Verify output result correctness and confirm algorithm logic
  ```
  The following output indicates successful precision comparison:
  ```bash
  test pass!
  ```