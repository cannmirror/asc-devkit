# Axpy Sample

## Overview

This sample implements the functionality of multiplying each element in the source operand src by a scalar and adding it to the corresponding element in the destination operand dst using the Axpy high-level API. The data type combinations for the source and destination operands of the Axpy interface can only be: (half, half), (float, float), or (half, float). In this sample, the input tensor and scalar data types are half, and the output tensor data type is float.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```plain
├── axpy_half_float
│   ├── scripts
│   │   ├── gen_data.py         // Input data and golden data generation script
│   │   └── verify_result.py    // Verification script for checking if output data matches golden data
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read and write functions
│   └── axpy_half_float.asc     // Ascend C operator implementation & call sample
```

## Sample Description

- Sample Function:
  The Axpy sample implements the functionality of multiplying each element in the source operand src by a scalar and adding it to the corresponding element in the destination operand dst, returning the calculation result.

  The corresponding mathematical expression is:

  $$
  out = x * scalar + out
  $$

- Sample Specifications:
  <table>
  <caption>Table 1: Sample Specifications</caption>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="5" align="center"> Axpy </td></tr>
  <tr><td rowspan="2" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">default</td></tr>
  <tr><td align="center">x</td><td align="center">[4, 128]</td><td align="center">half</td><td align="center">ND</td><td align="center">\</td></tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">out</td><td align="center">[4, 128]</td><td align="center">float32</td><td align="center">ND</td><td align="center">\</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">kernel_vec_ternary_scalar_Axpy_half_2_float</td></tr>
  </table>

- Sample Implementation:
  This sample implements an Axpy sample with a fixed shape of input x[4, 128] and output out[4, 128].
  - Kernel Implementation

    First, use the Duplicate interface to initialize the output tensor to 0, then use the Axpy interface to complete multiplying x by the scalar scalar and adding the original value in out to get the final result, which is then moved to external storage.

  - Tiling Implementation

    The host side uses GetAxpyMaxMinTmpSize to obtain the maximum and minimum temporary space required for the Axpy interface calculation.

  - Call Implementation
    Uses the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this sample to build and run the sample.

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

- Sample Execution

  ```bash
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                           # Execute the generated executable program to run the sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output correctness, confirm algorithm logic is correct
  ```

  For CPU debugging or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  Examples:

  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debugging mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clear the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build Option Description

  | Option | Available Values | Description |
  |--------|------------------|-------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debugging, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2/A3 series, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  The execution result is as follows, indicating that the accuracy comparison passed.

  ```bash
  test pass!
  ```