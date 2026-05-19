# Power Sample

## Overview

This sample implements element-wise power operations using the Power high-level API. It supports three modes: tensor-to-tensor, tensor-to-scalar, and scalar-to-tensor power operations for exponent and base.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```plain
├── power
│   ├── scripts
│   │   └── gen_data.py         // Script for generating input data and ground truth data
│   ├── CMakeLists.txt          // Build configuration file
│   ├── data_utils.h            // Data read and write functions
│   └── power.asc               // Ascend C sample implementation and invocation sample
```

## Sample Description

- Sample Function:  
  Implements element-wise power operations, supporting three modes: tensor-to-tensor, tensor-to-scalar, and scalar-to-tensor power operations for exponent and base.

  The calculation formula is as follows:
  $$Power(x, y) = x^y$$

  Tensor-to-tensor, mode = 0: Two tensors of the same length, performing element-wise power operation
  $$dstTensor_i = Power(srcbaseTensor_i, srcexpTensor_i)$$

  Tensor-to-scalar, mode = 1: Using a scalar as the exponent, all tensor elements use the same exponent for power operation
  $$dstTensor_i = Power(srcbaseTensor_i, srcexpScalar)$$

  Scalar-to-tensor, mode = 2: Using a scalar as the fixed base, all tensor elements use the same base for power operation
  $$dstTensor_i = Power(srcbaseScalar, srcexpTensor_i)$$

- Sample Specifications:  
  <table>
  <caption>Table 1: Sample Input/Output Specifications</caption>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="4" align="center"> power </td></tr>
  <tr><td rowspan="4" align="center">Sample Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">srcbase</td><td align="center">[1, 16]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">srcexp</td><td align="center">[1, 16]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">Sample Output</td></tr>
  <tr><td align="center">dst</td><td align="center">[1, 16]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">power_custom</td></tr>
  </table>

- Scenario Description:
  <table>
  <caption>Table 2: SCENARIO Parameter Description</caption>
  <tr><td align="center">SCENARIO</td><td align="center">Base</td><td align="center">Exponent</td><td align="center">Description</td></tr>
  <tr><td align="center">0</td><td align="center">Tensor</td><td align="center">Tensor</td><td align="center">Both base and exponent are tensors</td></tr>
  <tr><td align="center">1</td><td align="center">Tensor</td><td align="center">Scalar</td><td align="center">Base is a tensor, exponent is a scalar</td></tr>
  <tr><td align="center">2</td><td align="center">Scalar</td><td align="center">Tensor</td><td align="center">Base is a scalar, exponent is a tensor</td></tr>
  </table>

- Sample Implementation:  
  This sample implements the power_custom sample with fixed shapes: input srcbase[1, 16], srcexp[1, 16], output dst[1, 16]. The mode parameter defaults to 0, meaning both exponent and base are tensors.

  - Kernel Implementation

    Uses the Power high-level API for power operations, supporting three modes: tensor-to-tensor, tensor-to-scalar, and scalar-to-tensor.

  - Tiling Implementation

    On the host side, GetPowerMaxMinTmpSize is used to obtain the maximum and minimum temporary space required for the Power API calculation.

  - Invocation Implementation  
    Uses the kernel caller <<<>>> to invoke the kernel function.

## Compilation and Execution

Execute the following steps in the sample root directory to compile and run the sample.

- Configure Environment Variables  
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on your current environment.
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

- Sample Execution

  ```bash
  SCENARIO=0
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO=$SCENARIO ..;make -j;    # Build project, default NPU mode
  python3 ../scripts/gen_data.py --scenario $SCENARIO  # Generate test input data
  ./demo                           # Execute the compiled executable program to run the sample
  ```

  For CPU debug or NPU simulation modes, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  Examples:

  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** You must clean the cmake cache before switching compilation modes. Execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Compilation Options Description

  | Option | Available Values | Description |
  |--------|------------------|-------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2/A3 series, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  | `SCENARIO` | `0` (default), `1`, `2` | Scenario: 0-tensor-to-tensor, 1-tensor-to-scalar, 2-scalar-to-tensor |

- Execution Result

  The execution result appears as follows, indicating successful accuracy verification.

  ```bash
  test pass!
  ```