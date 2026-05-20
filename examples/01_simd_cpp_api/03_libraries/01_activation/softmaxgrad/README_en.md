# SoftmaxGrad Example

## Overview

This example implements softmax gradient computation based on the `SoftmaxGrad` or `SoftmaxGradFront` high-level API in a neural network backpropagation scenario.

- **SoftmaxGrad**: Performs complete softmax gradient backpropagation computation on the input tensor `[m, n]` row by row, with the computation formula `(grad - sum(grad * src)) * src`.
- **SoftmaxGradFront**: Only computes the first half of the softmax gradient, with the computation formula `sum(grad * src)`, outputting the sum result of each row. This interface is commonly used in scenarios such as FlashAttention that require intermediate gradient results.

Relationship between the two interfaces: `SoftmaxGradFront` is a subset of `SoftmaxGrad`. When `SoftmaxGrad(isFront=true)`, it is recommended to use `SoftmaxGradFront`. This example controls two modes through the compilation macro `USE_FRONT_MODE`, using float data type with both input x and y shapes as `[960, 960]`.

> **Note:** The last axis length of the input tensor must satisfy 32-byte alignment (for float type, this means a multiple of 8). In `SoftmaxGradFront` mode, the last axis of the output tensor is fixed to 1 datablock (for float type, 8 elements), with all elements having the same value.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── softmaxgrad
│   ├── scripts
│   │   ├── gen_data.py         // Input data and ground truth data generation script (supports two modes)
│   ├── CMakeLists.txt          // Build project file (supports -DUSE_FRONT_MODE)
│   ├── data_utils.h            // Data read/write functions
│   └── softmaxgrad.asc         // Ascend C example implementation & calling example (two modes combined)
```

## Example Specifications

<div align="left">
<table>
<caption>Table 1: SoftmaxGrad Mode Example Specifications</caption>
<tr><td align="center">Name</td><td align="center">Shape</td><td align="center">Data Type</td><td align="center">Layout Format</td></tr>
<tr><td align="center">Input x</td><td align="center">[960, 960]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">Input y</td><td align="center">[960, 960]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">Output z</td><td align="center">[960, 960]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">Kernel Function Name</td><td align="center" colspan="3">softmaxgrad_custom</td></tr>
</table>
</div>

- Example Implementation:  
  This example implements fixed shape examples with input x `[960, 960]`, y `[960, 960]`, computed with core partitioning.

  - Tiling Implementation  
    Partition by rows using the average allocation method to align with the number of cores, determining the number of rows for main and tail cores. Query SLICE_TABLE based on the reduce axis length to determine the number of rows processed per loop, call `GetSoftMaxGradMinTmpSize` and `SoftMaxGradTilingFunc` to obtain the Tiling parameters required by the API.

  - Kernel Implementation  
    Core computation steps: After loading input data, call the corresponding SoftmaxGrad/SoftmaxGradFront API to complete gradient computation, then store the results.

    SoftmaxGrad mode calling method:
    ```cpp
    AscendC::SoftmaxGrad<float, true>(yLocal, xLocal, yLocal, tmpBuffer, softmaxTiling, false, srcShape);
    ```

    SoftmaxGradFront mode calling method:
    ```cpp
    AscendC::SoftmaxGradFront<float>(zLocal, xLocal, yLocal, tmpBuffer, softmaxTiling, srcShape);
    ```

- Calling Implementation  
  Use the kernel launch operator `<<<>>>` to call the kernel function.

## Build and Run

Execute the following steps in the root directory of this example to build and run the example.

- Configure Environment Variables

  Select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on your current environment.
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

- Execute Example

   **SoftmaxGrad Mode (default)**:
   ```bash
   mkdir -p build && cd build;
   cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..; make -j;
   python3 ../scripts/gen_data.py
   ./demo
   ```

   **SoftmaxGradFront Mode**:
   ```bash
   mkdir -p build && cd build;
   cmake -DUSE_FRONT_MODE=ON -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..; make -j;
   python3 ../scripts/gen_data.py --front-mode
   ./demo
   ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..; make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..; make -j; # NPU simulation mode
  ```

  > **Note:** Before switching compilation modes, clean the cmake cache by executing `rm CMakeCache.txt` in the build directory and re-run cmake.

- Build Options Description

   <div align="left">
   <table>
   <caption>Table 3: Build Options Description</caption>
   <tr><td align="center">Option</td><td align="center">Available Values</td><td align="center">Description</td></tr>
   <tr><td align="center">CMAKE_ASC_RUN_MODE</td><td align="center">npu (default), cpu, sim</td><td align="center">Run mode: NPU run, CPU debug, NPU simulation</td></tr>
   <tr><td align="center">CMAKE_ASC_ARCHITECTURES</td><td align="center">dav-2201 (default), dav-3510</td><td align="center">NPU architecture: dav-2201 corresponds to Atlas A2/A3 series, dav-3510 corresponds to Ascend 950PR/Ascend 950DT</td></tr>
   <tr><td align="center">USE_FRONT_MODE</td><td align="center">OFF (default), ON</td><td align="center">Example mode: OFF is SoftmaxGrad, ON is SoftmaxGradFront</td></tr>
   </table>
   </div>

  The execution result is as follows, indicating successful accuracy comparison.
  ```bash
  test pass!
  ```