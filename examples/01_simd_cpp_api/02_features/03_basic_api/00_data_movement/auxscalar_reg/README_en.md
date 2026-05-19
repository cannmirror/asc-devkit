# auxscalar_reg Example

## Overview

This example implements reading multiple scalar data from UB (Unified Buffer) using AuxScalar method based on the Reg programming interface, combined with Adds for vector-scalar addition computation.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── auxscalar_reg
│   ├── scripts
│   │   ├── gen_data.py                // Input data and golden data generation script
│   ├── CMakeLists.txt                 // Build project file
│   ├── data_utils.h                   // Data read/write functions
│   ├── auxscalar_reg.asc              // AscendC example implementation & invocation example
│   └── README.md                      // Example introduction
```

## Example Description

- Example Functionality:
  Reads 4 scalar data from UB and performs Adds computation with vector x. Vector x has shape [1, 512], with each scalar computing with 128 consecutive vector elements controlled by a for loop.

  - AuxScalar read scalars can be used directly within VF functions; when used in mainScalar (outside VF functions), synchronization instructions are required

  **AuxScalar + Adds Mode**
  - Uses AuxScalar method (`__ubuf__` pointer subscript access, e.g., `scalarAddr[0]`) to read scalars from UB, combined with Adds for vector-scalar addition
  - Example Specifications:
    <table>
    <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="3" align="center">AIV Example</td></tr>
    <tr><td rowspan="3" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
    <tr><td align="center">x</td><td align="center">[1, 512]</td><td align="center">half</td></tr>
    <tr><td align="center">scalar</td><td align="center">[1, 4]</td><td align="center">half</td></tr>
    <tr><td rowspan="1" align="center">Example Output</td><td align="center">z</td><td align="center">[1, 512]</td><td align="center">half</td></tr>
    <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">auxscalar_reg</td></tr>
    </table>
  - Example Implementation:
    In the AuxScalarAddsVF function, reads the i-th scalar value from UB via `scalarAddr[i]`, and calls `Adds` to perform vector plus scalar computation.
    - Invocation Implementation
      Uses the kernel call operator `<<<>>>` to invoke the kernel function, starting 1 core.

## Compilation and Execution

Execute the following steps in the root directory of this example to compile and run the example.
- Configure Environment Variables
  Select the appropriate environment variable configuration command based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on the current environment.
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

- Example Execution
  ```bash
  mkdir -p build && cd build;                                                         # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;                                # Build project (default npu mode)
  python3 ../scripts/gen_data.py;                                                     # Generate test input data
  ./demo                                                                              # Execute compiled executable to run the example
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching compilation modes, clean the cmake cache by executing `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Compilation Options Description

  | Option | Possible Values | Description |
  |--------|-----------------|-------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  The following result indicates successful precision comparison:
  ```bash
  test pass!
  ```