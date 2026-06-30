# Axpy Operator Direct Call Sample (Tensor API)

## Overview

This sample implements the Axpy function, where each element in the source operand src is multiplied by a scalar and then added to the corresponding element in the destination operand dst. The Axpy interface supports three data type combinations for source and destination operands: (half, half), (float, float), and (half, float). In this sample, the input tensor, scalar, and output tensor are all of type half.

This sample implements the Axpy operator using Ascend C programming language, and uses the <<<>>> kernel call operator to complete the basic flow of running the operator kernel function on the NPU side, providing an end-to-end implementation. Unlike the basic API sample, this sample uses the Tensor API `Transform<Inst::Axpy>` interface to perform the computation.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── axpy_half_half_tensor_api
│   ├── scripts
│   │   ├── gen_data.py         // Script to generate input data and golden data
│   │   └── verify_result.py    // Script to verify whether output data matches golden data
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── axpy_half_half_tensor_api.asc  // Ascend C operator implementation & call sample
```

## Operator Description

- **Operator Function:**
  The Axpy operator implements the function of multiplying each element in the source operand src by a scalar, adding it to the corresponding element in the destination operand dst, and returning the computation result.

  The corresponding mathematical expression is:
  ```
  out = x * scalar + out
  ```

- **Operator Specification:**
  <table>
  <tr><td rowspan="1" align="center">OpType</td><td colspan="5" align="center"> Axpy </td></tr>
  <tr><td rowspan="2" align="center">Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">default</td></tr> <tr><td align="center">x</td><td align="center">4 * 128</td><td align="center">float16</td><td align="center">ND</td><td align="center">\</td></tr> <tr><td rowspan="1" align="center">Output</td><td align="center">out</td><td align="center">4 * 128</td><td align="center">float16</td><td align="center">ND</td><td align="center">\</td></tr> <tr><td rowspan="1" align="center">Kernel Name</td><td colspan="5" align="center">kernel_axpy_half_half_tensor_api</td></tr>
  </table>

- **Operator Implementation:**
  This sample implements the Axpy operator with a fixed shape of 4*128.

  - **Kernel Implementation**

    The mathematical expression of the Axpy operator is:
    ```
    out = x * scalar + out
    ```

    The computation logic is: using the vector computation interface provided by Tensor API, the input data needs to be first moved to on-chip storage, then the `Transform<Inst::Axpy>` interface is used to multiply x by the scalar and add the original value in out to obtain the final result, which is then moved out to external storage.

    The implementation flow of the Axpy operator is divided into 5 steps: Create GM/UB Tensor, CopyIn, Compute, CopyOut. The CopyIn task is responsible for moving the input Tensor from Global Memory to UB Local Memory. The Compute task is responsible for executing `Transform<Inst::Axpy>` on srcUbTensor, and the computation result is stored in dstUbTensor. The CopyOut task is responsible for moving the output data from dstUbTensor to the output Tensor in Global Memory.

  - **Call Implementation**

    Use the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this sample to compile and run the operator.

- **Configure Environment Variables**

  Please refer to the [installation guide](../../../../../docs/quick_start.md#prepare&install) of the CANN development toolkit package on the current environment to select the corresponding command for configuring environment variables.

  - Default path, root user installs CANN software package
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, non-root user installs CANN software package
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, install CANN software package
    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- **Sample Execution**
  ```bash
  mkdir -p build && cd build;   # Create and enter the build directory
  cmake ..; make -j;            # Build the project
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                        # Execute the compiled executable program to run the sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify whether the output result is correct
  ```

  The following execution result indicates that the accuracy comparison is successful.
  ```bash
  test pass!
  ```