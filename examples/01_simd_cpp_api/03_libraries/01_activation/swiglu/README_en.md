# SwiGLU Example

## Overview

This example performs SwiGLU computation on two input Tensors element-wise based on the SwiGLU high-level API in large language model and Mixture of Experts (MoE) scenarios. SwiGLU is a GLU variant that uses Swish as the activation function, with the computation formula dst_i = src0_i ⊗ Swish(src1_i), where Swish(x) = x/(1+e^(-βx)). This API is commonly used in gated feed-forward networks (FFN) in LLMs and supports data types such as float/half/bfloat16_t. This example uses the float data type with 32 input Tensor elements and a beta value of 1, completing SwiGLU activation computation.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── swiglu
│   ├── scripts
│   │   ├── gen_data.py         // Input data and ground truth data generation script
│   ├── CMakeLists.txt          // Build project file (supports -DCMAKE_ASC_RUN_MODE, -DCMAKE_ASC_ARCHITECTURES)
│   ├── data_utils.h            // Data read/write functions
│   └── swiglu.asc              // Ascend C example implementation & calling (with Tiling mechanism)
```

## Example Description

- Example Function:  
  SwiGLU is a GLU variant that uses Swish as the activation function.

  The computation formula is as follows:
  $$dstTensor_i=(srcTensor0_i)\bigotimes Swish(srcTensor1_i)$$
  The Swish activation function computation formula is as follows (β is a constant):
  $$Swish(x)=x/(1 + e^{(-\beta x)})$$

- Example Specifications:  
  <table border="2" align="left">
  <caption>Table 1: Example Specification Table</caption>
  <tr><td align="center" rowspan="1">Example Type</td><td align="center" colspan="4"> swiglu </td></tr>

  <tr><td align="center" rowspan="4">Example Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src0</td><td align="center">[1, 32]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">src1</td><td align="center">[1, 32]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center" rowspan="2">Example Output</td></tr>
  <tr><td align="center">dst</td><td align="center">[1, 32]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td align="center" rowspan="1">Kernel Function Name</td><td align="center" colspan="4">swiglu_custom</td></tr>
  </table>

- Example Implementation:  
  This example implements a fixed shape example with 32 input elements. Through the Tiling mechanism, computation parameters (dataLength, sharedTmpBufferSize) are passed from the Host side to the Device side, supporting flexible configuration of computation scale.

  - Kernel Implementation  
    Core computation steps: After moving input data from GM to UB, call `AscendC::SwiGLU` to complete SwiGLU computation, then move the results back to Global Memory.

  - Tiling Implementation  
    Temporary space handling: Obtain the required temporary space size through `AscendC::GetSwiGLUMaxMinTmpSize` and pass it to the Kernel side through Tiling. When the temporary space is greater than 0, use the buffer provided by the developer; otherwise, it is automatically allocated by the framework.

  - Calling Implementation  
    Use the kernel launch operator `<<<>>>` to call the kernel function, passing src0, src1, dst, workspace, and tiling parameters.

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

  **Default mode (dav-2201 architecture)**:
  ```bash
  mkdir -p build && cd build;
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..; make -j;
  python3 ../scripts/gen_data.py
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
  <caption>Table 2: Build Options Description</caption>
  <tr><td align="center">Option</td><td align="center">Available Values</td><td align="center">Description</td></tr>
  <tr><td align="center">CMAKE_ASC_RUN_MODE</td><td align="center">npu (default), cpu, sim</td><td align="center">Run mode: NPU run, CPU debug, NPU simulation</td></tr>
  <tr><td align="center">CMAKE_ASC_ARCHITECTURES</td><td align="center">dav-2201 (default), dav-3510</td><td align="center">NPU architecture: dav-2201 corresponds to Atlas A2/A3 series, dav-3510 corresponds to Ascend 950PR/Ascend 950DT</td></tr>
  </table>
  </div>

- Execution Result

  The execution result is as follows, indicating successful accuracy comparison.
  ```bash
  test pass!
  ```