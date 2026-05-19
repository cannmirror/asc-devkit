# Brcb Example

## Overview

This sample demonstrates broadcast copy functionality using the Brcb API in data filling scenarios, copying data from an input tensor to multiple DataBlocks in an output tensor. The Brcb API extracts 8 elements at a time from the input tensor and fills them into 8 DataBlocks (each DataBlock is 32 bytes) of the output tensor, with each element corresponding to one DataBlock, enabling efficient data broadcast operations.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── brcb
│   ├── scripts
│   │   ├── gen_data.py         // Script to generate input data and golden data
│   │   └── verify_result.py    // Script to verify output data against golden data
│   ├── CMakeLists.txt          // Build configuration file
│   ├── data_utils.h            // Data read/write functions
│   └── brcb.asc                // Ascend C sample implementation & invocation example
```

## Sample Description

- Sample Function:  
  This sample demonstrates the broadcast copy functionality using the Brcb API. It extracts 8 elements at a time from the input tensor and fills them into 8 DataBlocks (each DataBlock is 32 bytes) of the output tensor, with each element corresponding to one DataBlock. The Brcb API is suitable for scenarios requiring broadcasting small amounts of data to many locations, such as constant filling and mask generation. The repeatTime parameter controls the number of iterations, while dstBlkStride and dstRepStride parameters control the address stride between DataBlocks. For detailed API documentation, refer to [Brcb API Documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0089.html).

- Sample Specifications:  
  <table border="2" align="center">
  <caption>Table 1: Sample Input/Output Specifications</caption>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="4" align="center">Brcb</td></tr>
  <tr><td rowspan="3" align="center">Sample Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[1,16]</td><td align="center">uint16_t</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">Sample Output</td></tr>
  <tr><td align="center">y</td><td align="center">[1,256]</td><td align="center">uint16_t</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">Kernel Name</td><td colspan="4" align="center">brcb_custom</td></tr>
  </table>

- Sample Implementation:  
  This sample implements a Brcb broadcast copy example with fixed shape: input x[1,16], output y[1,256].

  - Kernel Implementation  
    - Call DataCopy basic API to transfer data from GM (Global Memory) to UB (Unified Buffer)
    - Call Brcb interface to execute broadcast copy operation, copying data from input tensor to multiple DataBlocks in output tensor
    - Call DataCopy basic API to transfer data from UB (Unified Buffer) to GM (Global Memory)

  - Invocation Implementation  
    Use the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the sample root directory to build and run the sample.

- Configure Environment Variables  
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development toolkit on your current environment.
  - Default path, root user installed CANN package
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, non-root user installed CANN package
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Custom path install_path, installed CANN package
    ```bash
    source ${install_path}/cann/set_env.sh
    ```
    
- Sample Execution
  ```bash
  mkdir -p build && cd build;      # Create and enter build directory
  cmake .. -DCMAKE_ASC_ARCHITECTURES=dav-2201;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                           # Execute the compiled executable to run the sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output results
  ```
  The following output indicates successful accuracy comparison.
  ```bash
  test pass!
  ```

  For CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  Example:
  ```bash
  cmake .. -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201;make -j; # CPU debug mode
  cmake .. -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201;make -j; # NPU simulation mode
  ```
  > **Note:** Before switching build modes, you need to clear the cmake cache. Execute `rm CMakeCache.txt` in the build directory and run cmake again.

- Build Options

  | Option | Available Values | Description |
  |--------|------------------|-------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 for Atlas A2 Training Series/Atlas A2 Inference Series and Atlas A3 Training Series/Atlas A3 Inference Series, dav-3510 for Ascend 950PR/Ascend 950DT |