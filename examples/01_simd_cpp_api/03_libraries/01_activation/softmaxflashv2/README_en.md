# SoftmaxFlashV2 Example

## Overview

This example implements a single softmaxflashv2 operator based on the SoftmaxFlashV2 high-level API in a large language model attention mechanism scenario. This API is an enhanced version of SoftmaxFlash corresponding to the FlashAttention-2 algorithm, supporting update mode (incremental computation), commonly used in attention mechanism computation and long sequence block processing scenarios in large language models. This example uses the float data type with input Tensor shape [960, 960], completing SoftmaxFlashV2 attention computation.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── softmaxflashv2
│   ├── scripts
│   │   ├── gen_data.py         // Input data and ground truth data generation script
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── softmaxflashv2.asc      // Ascend C operator implementation & calling example
```

## Example Description

- Example Function:
  Single softmaxflashv2 operator. For an input tensor[m0, m1, ...mt, n] (where t is greater than or equal to 0), the product of non-last axis lengths is treated as m, so the input tensor shape is viewed as [m, n]. The following computation is performed on the input tensor[m, n] row by row. Different update values correspond to different computation formulas, where x, inmax, and insum are inputs, and M, S, and E are outputs.
  update is false:

  $$M = rowmax(x_i)$$

  $$SoftmaxFlashV2(z_i) = exp(x_i - M)$$

  $$S=\sum_{i}^{n} \exp(x_i - M)$$

  update is true:

  $$M = max(rowmax(x_i), inmax)$$

  $$SoftmaxFlashV2(z_i) = exp(x_i - M)$$

  $$E = exp(inmax_i - M)$$

  $$S = sum_{i}^{n} exp(x_i - M) + E \cdot insum$$

- Example Specifications:

<div align="left">
<table>
<caption>Table 1: Example Specification Table</caption>
<tr><td align="center" rowspan="1">Example Type(OpType)</td><td align="center" colspan="4">Softmaxflash</td></tr>

<tr><td align="center" rowspan="3">Example Input</td></tr>
<tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center"> [960, 960] </td><td align="center">float</td><td align="center">ND</td></tr>

<tr><td align="center" rowspan="3">Example Output</td></tr>
<tr><td align="center">max</td><td align="center"> [960, 8] </td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">sum</td><td align="center"> [960, 8] </td><td align="center">float</td><td align="center">ND</td></tr>

<tr><td align="center" rowspan="1">Kernel Function Name</td><td align="center" colspan="4">softmaxflashv2_custom</td></tr>
</table>
</div>

- Example Implementation:  
  This example implements the softmaxflashv2 example with fixed shapes: input x [960, 960], output max[960, 8], sum[960, 8].

  - Kernel Implementation  
    Core computation steps: After loading input data, call `AscendC::SoftmaxFlashV2(xLocal, sumLocal, maxLocal, xLocal, expmaxLocal, sharedTmpBuffer, softmaxTiling)` to complete SoftmaxFlashV2 computation, then store the results.

  - Tiling Implementation  
    The tiling implementation process for the softmaxflashv2 example is as follows: First, partition the shape by rows using the average allocation method to align with the number of cores, determine the number of rows for the main cores, then determine the number of rows for the tail core. For the shape computed by the main core, call the tiling function of the SoftmaxFlashV2 high-level API to obtain the tiling parameters required by the API. The tiling for the high-level API required by the tail core computation is calculated by the kernel side.

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