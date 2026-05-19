# Gelu Example

## Overview

This example performs GELU (Gaussian Error Linear Unit) computation on input Tensor elements based on the Gelu high-level API in the activation function scenario. This example uses the float data type with an input Tensor element count of 32 to complete the Gelu example computation.

> **Advanced Interface Note:** In addition to the `Gelu` interface used in this example, Ascend C also provides the following advanced GELU interfaces. The calling method is consistent with `Gelu`. You can switch by simply replacing the function name:
> - **FasterGelu**: An accelerated version of GELU, suitable for scenarios with higher performance requirements. Replace `AscendC::Gelu` in `gelu.asc` with `AscendC::FasterGelu`.
> - **FasterGeluV2**: A further optimized version of GELU that reduces computational requirements. Replace `AscendC::Gelu` in `gelu.asc` with `AscendC::FasterGeluV2`.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── gelu
│   ├── scripts
│   │   └── gen_data.py         // Input data and ground truth data generation script
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read and write functions
│   ├── README.md               // Example documentation
│   └── gelu.asc                // Ascend C operator implementation & call example
```

## Example Description

- Example Function:  
  This example performs GELU activation computation on input Tensor elements element-wise and writes the computation results to the output Tensor.

  The computation formula is as follows:
  $$dstLocal_i = GELU(srcLocal_i)$$
  $$GELU(x)=0.5 * x * (1 + tanh(\sqrt{\frac{2}{\pi}} * (x + 0.044715 * x^3)))$$
  $$GELU(x)=\frac{x}{1 + e^{-1.59576912 * (x + 0.044715 * x^3)}}$$

- Example Specifications:

<div align="left">
<table>
<tr><td align="center" rowspan="1">Example Type(OpType)</td><td align="center" colspan="4"> gelu </td></tr>

<tr><td align="center" rowspan="3">Example Input</td></tr>
<tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">src</td><td align="center">[1, 32]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center" rowspan="2">Example Output</td></tr>
<tr><td align="center">dst</td><td align="center">[1, 32]</td><td align="center">float</td><td align="center">ND</td></tr>

<tr><td align="center" rowspan="1">Kernel Function Name</td><td align="center" colspan="4">gelu_custom</td></tr>
</table>
</div>

- Example Implementation:  
  This example implements the gelu_custom example with a fixed shape of input src[1, 32] and output dst[1, 32].

  - Kernel Implementation  
    Core computation steps: After moving input data in, call `AscendC::Gelu` to complete GELU computation, then move the results out.

  - Tiling Implementation  
    This example is a single-core element-wise computation scenario with no complex multi-core logic. The Host side obtains the temporary buffer size required by the API through `AscendC::GetGeluMinTmpSize` and passes it directly to the Kernel for use.

  - Call Implementation  
    Use the kernel call operator <<<>>> to call the kernel function.

## Build and Run

Execute the following steps in the root directory of this example to build and run the example.
- Configure Environment Variables  
  Select the appropriate environment variable configuration command based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on your current environment.
  - Default path, root user installed CANN software package
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, non-root user installed CANN software package
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, CANN software package installed
    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Example Execution
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

  > **Note:** Before switching build modes, clean the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build Option Description
  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result  
  The execution result is shown below, indicating the accuracy comparison passed.
  ```bash
  test pass!
  ```