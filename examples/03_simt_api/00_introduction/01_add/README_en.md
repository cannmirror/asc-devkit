# Add Operator Example Based on SIMT Programming Mode

## Overview

This example implements the Add operator based on Ascend C SIMT programming, implementing the function of element-wise addition of two input tensors to get an output tensor, demonstrating the basic flow of SIMT programming.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Supported CANN Software Versions
- \>= CANN 9.0.0

## Directory Structure

```
├── 01_add
│   ├── add.asc             # SIMT implementation add invocation example
|   └── README.md
```

## Operator Description

- Operator function:
  This operator implements the addition of two tensors x and y with shape 48 * 256 to get operator output z. The calculation formula for the i-th element is:
  
  ```
  z[i] = x[i] + y[i]
  ```

- Operator specification:
  <table>
  <tr><td rowspan="1" align="center">Operator Type(OpType)</td><td colspan="4" align="center">add</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">Operator Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">48 * 256</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">48 * 256</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Operator Output</td><td align="center">z</td><td align="center">48 * 256</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">add_custom</td></tr>
  </table>

- Data tiling:
  * Number of cores: 48 cores
  * Threads per core: 256 threads
  * Elements per thread: 1 element
  * Total processing capacity: 48x256=12288

- Operator implementation:
  The implementation flow of the operator is to get data at the specified index from input x (pointer on Global Memory). Based on the above data tiling, first calculate the index of data that the thread should process, then calculate the output value through the addition operator.

- Invocation implementation:
  Use the kernel launch operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this example to build and run the operator.
- Configure environment variables
  Select the corresponding command to configure environment variables based on the [installation method](../../../../docs/en/quick_start.md#prepare&install) of the CANN development toolkit on the current environment.
  - Default path, CANN software package installed by root user
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, CANN software package installed by non-root user
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, CANN software package installation
    ```bash
    source ${install_path}/cann/set_env.sh
    ```
    
- Example execution
  ```bash
  mkdir -p build && cd build;   # Create and enter build directory
  cmake ..; make -j;            # Build the project
  ./demo                        # Run the example
  ```
  The following output indicates successful accuracy verification.
  ```
  [Success] Case accuracy is verification passed.
  ```