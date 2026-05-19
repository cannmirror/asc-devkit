# Dequant Example

## Overview

This example implements dequantization computation using the [AscendDequant](../../../../../docs/api/context/AscendDequant.md) high-level API, which restores quantized low-precision data to high-precision data. The example demonstrates the PER_CHANNEL scenario (per-channel quantization), where int32_t type input data is multiplied by a scale factor to convert to float type output. On 950 series devices, while maintaining compatibility with the AscendDequant interface, it is recommended to use the [Dequantize](../../../../../docs/api/context/Dequantize.md) interface first. This interface can adapt to various quantization scenarios through a unified structure configuration.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── dequant
│   ├── scripts
│   │   ├── gen_data.py         // Script to generate input data and ground truth data
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read and write functions
│   └── dequant.asc             // Ascend C example implementation & invocation example
```

## Example Description

- Example Function:
  This example performs dequantization computation element-wise on the input tensor, converting int32_t data type to float and other data types.

- Example Specifications:
  <table border="2" align="left">
  <caption>Table 1: Example Input/Output Specifications</caption>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="4" align="center"> dequant </td></tr>

  <tr><td rowspan="4" align="center">Example Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
   <tr><td align="center">inputGm</td><td align="center">[128, 32]</td><td align="center">int32_t</td><td align="center">ND</td></tr>
   <tr><td align="center">deqScaleGm</td><td align="center">[1, 32]</td><td align="center">float</td><td align="center">ND</td></tr>

   <tr><td rowspan="2" align="center">Example Output</td></tr>
   <tr><td align="center">outputGm</td><td align="center">[128, 32]</td><td align="center">float</td><td align="center">ND</td></tr>


  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">dequant_custom</td></tr>
  </table>
  <br clear="left" />
<br />

- Example Implementation:
   This example implements a fixed shape with input inputGm[128, 32], scaleGm[1, 32], and output outputGm[128, 32]. It performs dequantization computation element-wise, converting int32_t data type to float and other data types.

  - Kernel Implementation
    The computation logic is: this example moves the input data to on-chip storage, then uses the AscendDequant (A2/A3) or Dequantize (950 series) high-level API interface to complete the dequantization computation and obtain the final result, which is then moved to external storage.

  - Tiling Implementation
    The tiling implementation process for the DequantCustom example is as follows: first obtain the maximum/minimum temporary space size required for the AscendDequant or Dequantize interface to complete the computation, use the minimum temporary space, and then determine the required tiling parameters based on the input length dataLength.

  - Invocation Implementation
    Use the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this example to build and run the example.
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
  python3 ../scripts/gen_data.py   # Generate test input data
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