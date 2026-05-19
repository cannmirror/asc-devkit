# Xor Example

## Overview

This example implements element-wise XOR operation using the Xor high-level API.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```plain
├── xor
│   ├── scripts
│   │   └── gen_data.py         // Input data and ground truth data generation script
│   ├── CMakeLists.txt          // Build configuration file
│   ├── data_utils.h            // Data read and write functions
│   └── xor.asc                 // Ascend C example implementation & invocation example
```

## Example Description

- Example Function:
  Performs element-wise XOR operation. The concept and operation rules of XOR (exclusive OR) are as follows:
  Concept: For two data participating in the operation, perform "exclusive OR" operation on each binary bit.
  Operation rules: 0^0=0; 0^1=1; 1^0=1; 1^1=0; that is, for the two objects participating in the operation, if the two corresponding bits are "different" (different values), the result of that bit is 1, otherwise 0 (same is 0, different is 1).

  The calculation formula is as follows:
  $$
  dstTensor_i = Xor(src0Tensor_i, src1Tensor_i)
  $$
  $$
  Xor(x, y) = (x \mid y) \& (\sim(x \& y))
  $$

- Example Specification:
  <table>
  <tr><td rowspan="1" align="center">Example Type(OpType)</td><td colspan="4" align="center"> xor </td></tr>

  <tr><td rowspan="4" align="center">Example Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src0</td><td align="center">[1, 1024]</td><td align="center">int16_t</td><td align="center">ND</td></tr>
  <tr><td align="center">src1</td><td align="center">[1, 1024]</td><td align="center">int16_t</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">Example Output</td></tr>
  <tr><td align="center">dst</td><td align="center">[1, 1024]</td><td align="center">int16_t</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">xor_custom</td></tr>
  </table>

- Example Implementation:
  This example implements the xor_custom example with a fixed shape of inputs src0[1, 1024], src1[1, 1024], and output dst[1, 1024].

  - Kernel Implementation

    Uses the Xor high-level API to perform element-wise XOR operation. You can choose to use a temporary buffer and specify the number of elements to calculate.

  - Tiling Implementation

    On the host side, uses GetXorMaxMinTmpSize to get the maximum and minimum temporary space required for the Xor interface calculation.

  - Invocation Implementation
    Uses the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this example to build and run the example.

- Configure Environment Variables
  Select the corresponding command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development toolkit on your current environment.
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

- Example Execution

  ```bash
  mkdir -p build && cd build;      # Create and enter the build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build the project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                           # Execute the compiled executable to run the example
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  For example:

  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clear the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build Options Description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2/A3 series, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  The execution result is as follows, indicating successful precision comparison.

  ```bash
  test pass!
  ```