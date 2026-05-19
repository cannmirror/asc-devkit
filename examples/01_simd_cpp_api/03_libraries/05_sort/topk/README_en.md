# TopK Sample

## Overview

In the sorting scenario, this sample uses the TopK high-level API to find the top K maximum or minimum values from input Tensors by value while preserving the corresponding index information. It supports joint sorting of float type values and int32_t type indices.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── topk
│   ├── scripts
│   │   └── gen_data.py         // Input data and ground truth data generation script
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read and write functions
│   └── topk.asc                // Ascend C sample implementation and call sample
```

## Sample Description

- Sample Function:  
  This sample function performs TopK computation on input tensors in Normal mode, used to obtain the top k maximum or minimum values and their corresponding indices in the last dimension.
- Sample Specification:  
  <table>
  <caption>Table 1: Sample Input/Output Specification</caption>
  <tr><td rowspan="1" align="center">Sample Type(OpType)</td><td colspan="4" align="center"> topk </td></tr>

  <tr><td rowspan="5" align="center">Sample Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">srcLocalValue</td><td align="center">[2, 32]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">srcLocalIndex</td><td align="center">[1, 32]</td><td align="center">int32_t</td><td align="center">ND</td></tr>
  <tr><td align="center">srcLocalFinish</td><td align="center">[1, 32]</td><td align="center">int32_t</td><td align="center">ND</td></tr>


  <tr><td rowspan="3" align="center">Sample Output</td></tr>
  <tr><td align="center">dstLocalValue</td><td align="center">[2, 8]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">dstLocalIndex</td><td align="center">[2, 8]</td><td align="center">int32_t</td><td align="center">ND</td></tr>


  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">topk_custom</td></tr>
  </table>

- Sample Implementation:  
    This sample implements a topk sample with fixed shape: input value tensor srcLocalValue[2, 32], index tensor srcLocalIndex[1, 32], completion flag srcLocalFinish[1, 32], output result value dstLocalValue[2, 8], result index dstLocalIndex[2, 8].

  - Kernel Implementation:  
    The computation logic is: input data needs to be moved to on-chip storage first, then the TopK high-level API is used to complete the topk computation, and finally the results are moved out.

  - Tiling Implementation:  
    The tiling implementation process of this sample is as follows: use the GetTopKMaxMinTmpSize interface to calculate the required maximum/minimum temporary space size, use the minimum temporary space, and then determine the required tiling parameters based on the input length.

  - Call Implementation  
    Use the kernel call operator <<<>>> to call the kernel function.

## Build and Run

Execute the following steps in the root directory of this sample to build and run the sample.
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

- Sample Execution
  ```bash
  mkdir -p build && cd build;
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # Default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo
  ```

  When using CPU debugging or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debugging mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching compilation modes, you need to clear the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build Options Description
  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debugging, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2/A3 series, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result
  The execution result is as follows, indicating successful precision comparison.
  ```bash
  test pass!
  ```