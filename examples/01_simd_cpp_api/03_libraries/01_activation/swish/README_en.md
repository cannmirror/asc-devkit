# Swish Example

## Overview

This example implements activation function computation based on the Swish/Silu high-level API in a deep learning model activation function scenario. The two APIs have a close mathematical relationship:

- **Swish (default)**: `y = x / (1 + exp(-beta * x))`, where beta is an adjustable parameter. In this example, beta=1.702 (GELU approximation)
- **Silu**: A special case of Swish when beta=1, `y = x / (1 + exp(-x))`, smoother than ReLU with gradients that never completely vanish

This example controls two modes through the compilation macro `USE_SILU_MODE`, using float data type with 32 input Tensor elements.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── swish
│   ├── scripts
│   │   ├── gen_data.py         // Input data and ground truth data generation script (supports Swish/Silu modes)
│   ├── CMakeLists.txt          // Build project file (supports -DUSE_SILU_MODE)
│   ├── data_utils.h            // Data read/write functions
│   └── swish.asc               // Ascend C operator implementation & calling example (two modes combined)
```

## Example Description

- Example Function:  
  This example performs Swish/Silu activation computation on the input Tensor element-wise and writes the computation results to the output Tensor. Swish and Silu have a close mathematical relationship; Silu is a special case of Swish when beta=1.

  The computation formula is as follows:
  $$dstLocal_i = Swish(srcLocal_i) = \frac{srcLocal_i}{1 + e^{-\beta \cdot srcLocal_i}}$$
  $$dstLocal_i = Silu(srcLocal_i) = \frac{srcLocal_i}{1 + e^{-srcLocal_i}}$$

  In this example, Swish mode uses beta=1.702 (GELU approximation value), and two modes are switched through the compilation macro `USE_SILU_MODE`.

- Example Specifications:

<div align="left">
<table>
<caption>Table 1: Example Specification Table</caption>
<tr><td align="center" rowspan="1">Example Type(OpType)</td><td align="center" colspan="4"> swish / silu </td></tr>

<tr><td align="center" rowspan="3">Example Input</td></tr>
<tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">src</td><td align="center">[1, 32]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center" rowspan="2">Example Output</td></tr>
<tr><td align="center">dst</td><td align="center">[1, 32]</td><td align="center">float</td><td align="center">ND</td></tr>

<tr><td align="center" rowspan="1">Kernel Function Name</td><td align="center" colspan="4">swish_custom</td></tr>
</table>
</div>

- Example Implementation:  
  This example implements the swish_custom example with fixed shapes: input src[1, 32], output dst[1, 32].

  - Kernel Implementation  
    Core computation steps: After loading input data, call the corresponding high-level API to complete computation, then store the results.

    Swish mode calling method:
    ```cpp
    AscendC::Swish(dstLocal, srcLocal, dataSize, scalarValue);
    ```

    Silu mode calling method:
    ```cpp
    AscendC::Silu<T, false>(dstLocal, srcLocal, dataSize);
    ```

  - Tiling Implementation  
    This example is a single-core element-wise computation scenario with no complex core partitioning logic. The Host side obtains the temporary buffer size required by the API through `AscendC::GetSwishTmpSize` (Swish mode) or `AscendC::GetSiluTmpSize` (Silu mode) and directly passes it to the Kernel for use.

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

  **Swish mode (default)**:
  ```bash
  mkdir -p build && cd build;
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..; make -j;
  python3 ../scripts/gen_data.py
  ./demo
  ```

  **Silu mode**:
  ```bash
  mkdir -p build && cd build;
  cmake -DUSE_SILU_MODE=ON -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..; make -j;
  python3 ../scripts/gen_data.py --silu-mode
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
  <tr><td align="center">USE_SILU_MODE</td><td align="center">OFF (default), ON</td><td align="center">Example mode: OFF is Swish, ON is Silu</td></tr>
  </table>
  </div>

- Execution Result

  The execution result is as follows, indicating successful accuracy comparison.
  ```bash
  test pass!
  ```