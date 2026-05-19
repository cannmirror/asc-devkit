# Softmax Example

## Overview

This example performs SoftMax computation on an input Tensor row by row based on the SoftMax high-level API in an activation function scenario, and uses AdjustSoftMaxRes to post-process the computation results. SoftMax performs reducemax, sub, exp, reducesum, and div steps on the input tensor[m, n] row by row to obtain a normalized probability distribution, commonly used in attention mechanisms and output layers of classification tasks.

> **API Note:** In addition to the `SoftMax` interface used in this example, Ascend C also provides the `SimpleSoftMax` interface, which uses pre-computed sum and max data to perform SoftMax computation on the input Tensor without internal reduce processes. Simply replace `AscendC::SoftMax` with `AscendC::SimpleSoftMax` in `softmax.asc` to switch.

AdjustSoftMaxRes is used to post-process SoftMax computation results. When the input max contains a specified value, it adjusts the corresponding softmaxres result row by row to a custom value. This example uses the float data type with input Tensor shape [32, 32], completing SoftMax computation and AdjustSoftMaxRes post-processing.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── softmax
│   ├── scripts
│   │   └── gen_data.py         // Input data and ground truth data generation script
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   ├── README.md               // Example documentation
│   └── softmax.asc             // Ascend C example implementation & calling example
```

## Example Description

- Example Function:

  This example performs SoftMax computation on the input Tensor row by row and uses AdjustSoftMaxRes to post-process the computation results. When the input max contains a specified value (0xFF7FFFFF, the maximum finite value for float type), it adjusts the corresponding position data in the output to a custom value (0.0, floating-point zero). This mechanism is commonly used in attention mask scenarios to set softmax outputs of invalid positions to zero.

- Example Specifications:

<div align="left">
<table>
<caption>Table 1: Example Specification Table</caption>
<tr><td align="center" rowspan="1">Example Type(OpType)</td><td align="center" colspan="4">softmax</td></tr>

<tr><td align="center" rowspan="3">Example Input</td></tr>
<tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">[32, 32]</td><td align="center">float</td><td align="center">ND</td></tr>

<tr><td align="center" rowspan="4">Example Output</td></tr>
<tr><td align="center">y</td><td align="center">[32, 32]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">max</td><td align="center">[32, 8]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">sum</td><td align="center">[32, 8]</td><td align="center">float</td><td align="center">ND</td></tr>

<tr><td align="center" rowspan="1">Kernel Function Name</td><td align="center" colspan="4">softmax_custom</td></tr>
</table>
</div>

- Example Implementation:  
  This example implements the softmax example with fixed shapes: input x[32, 32], output y[32, 32], max[32, 8], sum[32, 8].

  - Kernel Implementation  
    Core computation steps: After loading input data, call `AscendC::SoftMax` to complete SoftMax computation, then call `AscendC::AdjustSoftMaxRes` to post-process the results, and finally store the results.

  - Tiling Implementation  
    The tiling implementation process for the softmax example is as follows: First, partition the shape by rows using the average allocation method to align with the number of cores, determine the number of rows for the main cores, then determine the number of rows for the tail core. For the shape computed by the main core, call GetSoftMaxMinTmpSize to obtain the temporary buffer size required by the API, then call the tiling function of the SoftMax high-level API to obtain the tiling parameters required by the API. The tiling for the high-level API required by the tail core computation is calculated by the kernel side.

  - Calling Implementation  
    Use the kernel launch operator `<<<>>>` to call the kernel function.

## Build and Run

Execute the following steps in the root directory of this example to build and run the example.
- Configure Environment Variables
  Select the corresponding command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development toolkit package on your current environment.
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

- Execute Example
  ```bash
  mkdir -p build && cd build;
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # Default npu mode
  python3 ../scripts/gen_data.py
  ./demo
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching compilation modes, clean the cmake cache by executing `rm CMakeCache.txt` in the build directory and re-run cmake.

- Build Options Description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2/A3 series, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  The execution result is as follows, indicating successful accuracy comparison.
  ```bash
  test pass!
  ```