# Fill Example

## Overview

This example initializes data in Global Memory to a specified value using the Fill high-level API, and performs vector addition using the Add interface. The Fill interface supports initializing output space before data transfer, commonly used for workspace address or output data zeroing scenarios.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```plain
├── fill
│   ├── scripts
│   │   └── gen_data.py         // Input data and ground truth data generation script
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read and write functions
│   └── fill.asc                // Ascend C example implementation & invocation example
```

## Example Description

- Example Function:
  This example initializes data in Global Memory to a specified value using the Fill high-level API in scenarios that require pre-initialization of Global Memory data, and performs vector addition using the Add interface.

- Example Specifications:
  <table>
  <caption>Table 1: Example Input/Output Specifications</caption>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="4" align="center">fill</td></tr>

  <tr><td rowspan="4" align="center">Example Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">input_x</td><td align="center">[1, 256]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">input_y</td><td align="center">[1, 256]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="2" align="center">Example Output</td></tr>
  <tr><td align="center">output_z</td><td align="center">[1, 256]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">fill_custom</td></tr>
  </table>

- Example Implementation:
  This example implements a fill example with fixed shapes: input input_x[1, 256], input_y[1, 256], output output_z[1, 256]. For detailed API information, refer to the [Fill API Documentation](../../../../../docs/api/context/Fill.md).

  - Kernel Implementation
    The computation logic includes the following steps:
    1. Use the Fill high-level API to initialize the output Global Memory to the current kernel's blockIdx value
    2. Use the Add interface to perform vector addition
    3. Use SetAtomicAdd mode to accumulate the computation results to the output Global Memory

  - Tiling Implementation
    In this example, no tiling implementation is required.

  - Invocation Implementation
    Use the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this example to build and run the example.

- Configure Environment Variables
  Select the appropriate environment variable configuration command based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on your current environment.
  - Default path, CANN software package installed by root user

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, CANN software package installed by non-root user

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, CANN software package installed

    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Example Execution

  ```bash
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                           # Execute the compiled executable program to run the example
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  For example:

  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clear the cmake cache by running `rm CMakeCache.txt` in the build directory and then re-run cmake.

- Build Options Description

  | Option | Available Values | Description |
  |--------|------------------|-------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2/A3 series, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  The execution result is as follows, indicating successful accuracy comparison.

  ```bash
  test pass!
  ```