# GroupMatmul Operator Direct Invocation Sample

## Overview

This sample introduces the high-performance implementation of the QuantGroupMatmul operator, supporting grouped quantization matrix multiplication and Gelu activation computation, using the <<<>>> kernel call operator to complete the basic flow of running and verifying the operator kernel function on the NPU side.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── grouped_matmul
│   ├── scripts
│   │   ├── gen_data.py                    // Input data and golden data generation script
│   │   └── verify_result.py               // Verification script for comparing output data with golden data
│   ├── CMakeLists.txt                     // Build project file
│   ├── data_utils.h                       // Data read and write functions
│   └── quant_group_matmul_custom.asc      // Ascend C operator implementation & invocation sample
```

## Operator Description

- Operator Functionality:  
  The operator implements grouped pertoken quantization matmul computation, with the grouping axis as the m axis, and performs Gelu activation function computation on the results.

  The QuantGroupMatmul calculation formula is:

  ```python
  offset = 0
  for i in range(g):
      mmOut = x[offset:offset + group[i]] * weight[i] + bias[i]  # Cube computation
      y[offset:offset + group[i]] = Gelu(mmOut * scale[i] * pertokenScale[offset:offset + group[i]])  # vector computation
      offset += group[i]
  ```

  - x: left matrix, shape [m, k], data type int8;
  - weight: right matrix, shape [g, k, n], data type int8;
  - bias: matrix multiplication bias, shape [g, n], data type int32, bias[i] is applied to each row of the i-th matrix multiplication result;
  - group: records the size of each group m, data type int64;
  - scale: quantization parameter for right matrix, shape [g, n], data type float, used for dequantization of matrix multiplication results, scale[i] is used for dequantization of the i-th matrix multiplication result;
  - pertokenScale: quantization parameter for left matrix, shape [m], data type float, used for dequantization of matrix multiplication results, using the same index range as x rows for dequantization;
  - y: output, matrix storing matrix multiplication results, shape [m, n], data type float16;

- Operator Specification:

<table>
<tr><td rowspan="1" align="center">Operator Type(OpType)</td><td colspan="4" align="center">QuantGroupMatmul</td></tr>
<tr><td rowspan="7" align="center">Operator Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">1024 * 1024</td><td align="center">int8</td><td align="center">ND</td></tr>
<tr><td align="center">weight</td><td align="center">8 * 1024 * 8192</td><td align="center">int8</td><td align="center">NZ</td></tr>
<tr><td align="center">bias</td><td align="center">8 * 8192</td><td align="center">int32</td><td align="center">ND</td></tr>
<tr><td align="center">group</td><td align="center">8</td><td align="center">int64</td><td align="center">ND</td></tr>
<tr><td align="center">scale</td><td align="center">8 * 8192</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">pretokenScale</td><td align="center">1024</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Operator Output</td><td align="center">y</td><td align="center">1024 * 8192</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">quant_group_matmul_custom</td></tr>
</table>

- Operator Implementation:  
  This sample implements the pertoken quantized QuantGroupMatmul operator.

  - Kernel Implementation
  
    The QuantGroupMatmul operator computation is:

    ```python
    offset = 0
    for i in range(g):
        mmOut = x[offset:offset + group[i]] * weight[i] + bias[i]  # Cube computation
        y[offset:offset + group[i]] = Gelu(mmOut * scale[i] * pertokenScale[offset:offset + group[i]])  # vector computation
        offset += group[i]
    ```
  - Invocation Implementation  
    Use the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this sample to build and run the operator.
- Configure environment variables  
  Select the appropriate environment variable configuration command based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package in the current environment.
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
    
- Sample execution
  ```bash
  mkdir -p build && cd build;   # Create and enter build directory
  cmake ..;make -j;             # Build project
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                        # Execute compiled executable program, run sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output correctness, confirm algorithm logic is correct
  ```
  Execution result shown below indicates accuracy comparison passed.
  ```bash
  test pass!
  ```