# Arithmetic Class Example

## Overview

This example demonstrates the usage of basic arithmetic class interfaces based on LeakyRelu.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```plain
├── arithmetic
│   ├── scripts
│   │   ├── gen_data.py         // Input data and golden data generation script
│   │   └── verify_result.py    // Verification script for output data and golden data
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── arithmetic.asc          // Ascend C example implementation & calling example
```

## Example Description

- Example Function:

  This example implements element-wise Leaky Relu (Leaky Rectified Linear Unit) operation on the input tensor and returns the computation result.

  The corresponding mathematical expression is:
  $$
  out =
  \begin{cases}
  x, & \text{if } x \ge 0 \\
  \alpha x, & \text{if } x < 0
  \end{cases}
  $$

- Example Specification:
  <table>
  <tr><th align="center">Example Type (OpType)</th><th colspan="4" align="center">LeakyRelu</th></tr>
  <tr><td rowspan="2" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[4, 128]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">out</td><td align="center">[4, 128]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">arithmetic_kernel</td></tr>
  </table>

- Example Implementation:

  This example implements a LeakyRelu example with fixed shape [4, 128].

  - Kernel Implementation
    - Call DataCopy basic API to move data from GM (Global Memory) to UB (Unified Buffer)
    - Call LeakyRelu interface to perform Leaky Relu operation on the input tensor
    - Call DataCopy basic API to move computation results from UB (Unified Buffer) to GM (Global Memory)

  - Calling Implementation
    Use kernel call operator <<<>>> to call the kernel function.

## Build and Run

Execute the following steps in the root directory of this example to build and run the example.

- Configure environment variables

  Please select the corresponding environment variable configuration command based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on the current environment.

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

- Example execution

  ```bash
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;             # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                        # Execute the compiled executable program, run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output result correctness, confirm algorithm logic is correct
  ```

  When using CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:

  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clean cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then re-run cmake.

- Build option description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2/A3 series, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution result

  The execution result is shown below, indicating the precision comparison passed.

  ```bash
  test pass!
  ```