# Quant Example

## Overview

This example implements quantization computation using the [AscendQuant](../../../../../docs/api/context/AscendQuant.md) high-level API, which converts high-precision data to low-precision data to reduce storage and computational overhead. The example demonstrates the process of quantizing float type input data through scale scaling and offset to int8_t type output. On 950 devices, while maintaining compatibility with the AscendQuant interface, it is recommended to use the [Quantize](../../../../../docs/api/context/Quantize.md) interface first. This interface can adapt to various quantization scenarios through a unified structure configuration.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── quant
│   ├── scripts
│   │   ├── gen_data.py         // Script to generate input data and ground truth data
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read and write functions
│   └── quant.asc               // Ascend C operator implementation & invocation example
```

## Example Description

- Example Function:
  QuantCustom single example, performs quantization computation element-wise on the input tensor, converting half/float data type to int8_t data type.

- Example Specifications:
  <table border="2" align="left">
  <caption>Table 1: Example Input/Output Specifications</caption>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="4" align="center"> quant </td></tr>

  <tr><td rowspan="3" align="center">Example Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">inputGm</td><td align="center">[1, 1024]</td><td align="center">float</td><td align="center">ND</td></tr>


  <tr><td rowspan="2" align="center">Example Output</td></tr>
  <tr><td align="center">outputGm</td><td align="center">[1, 1024]</td><td align="center">int8_t</td><td align="center">ND</td></tr>


  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">quant_custom</td></tr>
  </table>
  <br clear="left" />
<br />

- Example Implementation:
  This example implements a fixed shape with input inputGm[1, 1024] and quantization parameters scale=2.0, offset=0.9. This example uses the PER_TENSOR scenario (per-tensor quantization), converting float data type to int8_t data type.

  - Kernel Implementation
    The computation logic is: the operation elements of the vector computation interface provided by Ascend C are all LocalTensor. Input data must first be moved to on-chip storage, then the AscendQuant (A2A3) or Quantize (950 series) high-level API interface is used to complete the quantization computation and obtain the final result, which is then moved to external storage.

  - Tiling Implementation
    The tiling implementation process for the QuantCustom example is as follows: first obtain the maximum/minimum temporary space size required for the AscendQuant or Quantize interface to complete the computation, use the minimum temporary space, and then determine the required tiling parameters based on the input length dataLength.

  - Invocation Implementation
    Use the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this example to build and run the operator.
- Configure Environment Variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on your current environment.
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
    
- Run the Example
  ```bash
  mkdir -p build && cd build;
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # Default npu mode
  python3 ../scripts/gen_data.py -DCMAKE_ASC_ARCHITECTURES=dav-2201   # Generate test input data
  ./demo
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  For example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clear the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build Options
  | Option | Available Values | Description |
  |--------|------------------|-------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  
- Execution Result
  The following execution result indicates that the precision comparison passed.
  ```bash
  test pass!
  ```