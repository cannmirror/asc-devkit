# Implementing Add Operator Sample Using C_API (Asynchronous Scenario)

## Overview

This sample demonstrates the Add operator sample written using C_API interfaces, implemented based on asynchronous data movement and computation interfaces.

## Supported Products

- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── 01_async_add
│   ├── CMakeLists.txt      // Build project file
│   ├── c_api_add.asc       // Ascend C operator implementation and invocation sample
│   └── README.md
```

## Operator Description

- Operator Function:

  The Add operator implements the function of adding two data values and returning the result. The corresponding mathematical expression is:

  ```
  z = x + y
  ```

- Operator Specification:

  <table>
  <tr><td rowspan="1" align="center">Operator Type (OpType)</td><td colspan="4" align="center">Add</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">Operator Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Operator Output</td><td align="center">z</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">add_custom</td></tr>
  </table>

- Operator Implementation:

  - Kernel Implementation

    The computation logic is: C_API input data needs to be moved to on-chip storage first, then the computation interface is used to complete the addition of two input parameters to obtain the final result, and then move it to external storage.

    The implementation process of the Add operator consists of three steps:

    Step 1: Move the input x and y from Global Memory to Local Memory, storing them in xLocal and yLocal respectively.

    Step 2: Perform addition operation on xLocal and yLocal, storing the computation result in zLocal.

    Step 3: Move the output data from zLocal to the output z on Global Memory.

  - Invocation Implementation

    Use the kernel call operator <<<>>> to invoke the kernel function.

## Compilation and Execution

Execute the following steps in the root directory of this sample to compile and run the operator.

- Configure Environment Variables

  Please select the appropriate command to configure environment variables based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on your current environment.

  - Default path, root user installed CANN software package

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, non-root user installed CANN software package

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, installed CANN software package

    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Sample Execution

  ```bash
  mkdir -p build && cd build;   # Create and enter build directory
  cmake ..;make -j;             # Build the project
  ./c_api_add_example           # Run the sample
  ```

  The following execution result indicates that the accuracy verification passed successfully.

  ```bash
  [Success] Case accuracy is verification passed.
  ```