# GeGLU Example

## Overview

This example implements GLU variant activation computation based on the GeGLU high-level API in the GLU (Gated Linear Unit) gated activation function scenario. GeGLU uses GELU as the activation function, with the computation mode `dst = src0 * GELU(src1)`. This example uses the float data type with 1024 input Tensor elements, completing the Geglu example computation.

> **API Note:** In addition to the `GeGLU` interface used in this example, Ascend C also provides the following GLU variant interfaces. The calling method is consistent with `GeGLU` - simply replace the function name to switch:
> - **ReGlu**: A GLU variant that uses Relu as the activation function. Replace `AscendC::GeGLU` with `AscendC::ReGlu` in `geglu.asc`.


## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── geglu
│   ├── scripts
│   │   └── gen_data.py         // Input data and ground truth data generation script
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   ├── README.md               // Example documentation
│   └── geglu.asc               // Ascend C example implementation & calling example
```

## Example Description

- Example Function:
  This example performs GeGLU gated activation computation on two input Tensors and writes the computation results to the output Tensor.

  GeGLU is a GLU variant that uses GELU as the activation function.

  The computation formula is as follows:
  $$dstTensor_i = srcTensor0_i \otimes GELU(srcTensor1_i)$$
  The GELU activation function computation formula is as follows:
  $$GELU(x)=0.5 * x * (1 + tanh(\sqrt{\frac{2}{\pi}} * (x + 0.044715 * x^3)))$$
  After simplification:
  $$GELU(x)=\frac{x}{1 + e^{-1.59576912 * (x + 0.044715 * x^3)}}$$

- Example Specifications:

<div align="left">
<table>
<tr><td align="center" rowspan="1">Example Type(OpType)</td><td align="center" colspan="4"> geglu </td></tr>

<tr><td align="center" rowspan="4">Example Input</td></tr>
<tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">src0</td><td align="center">[1,1024]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">src1</td><td align="center">[1,1024]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center" rowspan="2">Example Output</td></tr>
<tr><td align="center">dst</td><td align="center">[1,1024]</td><td align="center">float</td><td align="center">ND</td></tr>

<tr><td align="center" rowspan="1">Kernel Function Name</td><td align="center" colspan="4">geglu_custom</td></tr>
</table>
</div>

- Example Implementation:  
  This example implements the geglu_custom example with fixed shapes: input src[1,1024], src1[1,1024], output dst[1,1024].

  - Kernel Implementation  
    Core computation steps: After loading input data, call `AscendC::GeGLU` to complete GeGLU gated activation computation, then store the results.

  - Tiling Implementation  
    This example is a single-core element-wise computation scenario with no complex core partitioning logic. The Host side obtains the temporary buffer size required by the API through `AscendC::GetGeGLUMaxMinTmpSize` and directly passes it to the Kernel for use.

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