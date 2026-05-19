# SoftmaxFlashV3 Example

## Overview

This example implements a single softmaxflashv3 operator based on the SoftmaxFlashV3 high-level API in a large model training attention optimization scenario. This API is an enhanced version of SoftmaxFlash corresponding to the Softmax PASA algorithm. It adds mean calculation on top of V2 and performs numerical stability optimization through the alpha parameter, suitable for softmax computation scenarios that require higher numerical precision. This example uses half (input/output) and float (statistics) data types with input Tensor shape [8, 2048], completing SoftmaxFlashV3 attention computation.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── softmaxflashv3
│   ├── scripts
│   │   ├── gen_data.py         // Input data and ground truth data generation script
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── softmaxflashv3.asc      // Ascend C operator implementation & calling example
```

## Example Description

- Example Function:
  For an input tensor[m0, m1, ..., mt, n] (where t is greater than or equal to 0), the product of non-last axis lengths m0, m1, ..., mt is treated as m, so the input tensor shape is viewed as [m, n]. Split the last axis of the input tensor x, with the number of chunks being splitMeanCnt, and the split tensor is x_cnti. Compute according to the following formula, where x, inmax, insum, and inmean are inputs, and M, S, E, and A are outputs.
  update is false:

  $$
  A_1 = \text{rowmean}(x_{cnt})_i, i \in [0, \text{splitMeanCnt}]\\
  A_2 = \text{rowmean}(x_i), i \in [0, n]\\
  x_i = x_i - (A_2 - A_1) * (\alpha / (1 - \alpha))\\
  A = A_2\\
  M_1 = \text{rowmax}(x_i), i \in [0, n]\\
  M = M_1\\
  M_2 = M\\
  \text{SoftmaxFlashV3}(z_i) = \exp(x_i - M_2), i \in [0, n]\\
  S = \sum_{i}^{n} \exp(x_i - M_2)\\
  $$

  update is true:

  $$
  A_1 = \text{rowmean}(x_{cnt})_i, i \in [0, \text{splitMeanCnt}]\\
  A_2 = \text{rowmean}(x_i), i \in [0, n]\\
  x_i = x_i - (A_2 - A_1) * (\alpha / (1 - \alpha))\\
  A = (A_2 + \text{inmean} * (\text{loopCnt} - 1)) / \text{loopCnt}\\
  M_1 = \text{rowmax}(x_i), i \in [0, n]\\
  C = (A_2 - A) * (\alpha / (1 - \alpha))\\
  P = (\text{inmean} - A) * (\alpha / (1 - \alpha))\\
  M = \max(C + M_1, P + \text{inmax})\\
  M_2 = M - C\\
  \text{SoftmaxFlashV3}(z_i) = \exp(x_i - M_2), i \in [0, n]\\
  E = \exp(\text{inmax}_i - M_2 + P)\\
  S = \sum_{i}^{n} \exp(x_i - M_2) + E * \text{insum}\\
  $$

- Example Specifications:

<div align="left">
<table>
<caption>Table 1: Example Specification Table</caption>
<tr><td align="center" rowspan="1">Example Type(OpType)</td><td align="center" colspan="4"> softmaxflashv3 </td></tr>

<tr><td align="center" rowspan="6">Example Input</td></tr>
<tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">src</td><td align="center"> [8, 2048] </td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td align="center">inMax</td><td align="center"> [8, 8] </td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">inSum</td><td align="center"> [8, 8] </td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">inMean</td><td align="center"> [8, 8] </td><td align="center">float</td><td align="center">ND</td></tr>

<tr><td align="center" rowspan="2">Example Output</td></tr>
<tr><td align="center">dst</td><td align="center"> [8, 2048] </td><td align="center">half</td><td align="center">ND</td></tr>

<tr><td align="center" rowspan="1">Kernel Function Name</td><td align="center" colspan="4">softmaxflashv3_custom</td></tr>
</table>
</div>

- Example Implementation:  
  This example implements the softmaxflashv3 example with fixed shapes: input src[8, 2048], inMax[8, 8], inSum[8, 8], inMean[8, 8], output dst[8, 2048].

  - Kernel Implementation  
    Core computation steps: After loading input data, call `AscendC::SoftmaxFlashV3` to complete SoftmaxFlashV3 computation, then store the results.

  - Tiling Implementation  
    The tiling implementation process for the softmaxflashv3 example is as follows: First, partition the shape by rows using the average allocation method to align with the number of cores, determine the number of rows for the main cores, then determine the number of rows for the tail core. The Kernel side calculates the SoftMaxTiling parameters based on baseM.

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