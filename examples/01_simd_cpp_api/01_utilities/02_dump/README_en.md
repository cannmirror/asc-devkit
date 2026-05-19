# MmadCustomDump Sample

## Overview

This sample uses matrix multiplication as an example to demonstrate the usage of kernel debugging interfaces `asc_dump` and `printf`, implementing dump of tensor data on the NPU side and output of computation parameters.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── 02_dump
│   ├── CMakeLists.txt         // Build project file
│   ├── half.hpp               // Data type dependency file
│   └── mmad_custom_dump.asc   // Ascend C sample implementation & invocation sample
```

## Sample Description

- Sample Function:
  This sample is based on matrix multiplication computation and demonstrates the usage of the `asc_dump` series interfaces (including `asc_dump_gm`, `asc_dump_cbuf`, and `asc_dump_l1buf`) in NPU-side kernel functions. By calling these interfaces, tensor data at different physical locations can be visualized.

  Additionally, this series of interfaces is compatible with the `AscendC::DumpTensor` interface. However, in future development, it is recommended to prioritize using the `asc_dump` series interfaces. If you need to dump data at a specified offset position, since the `asc_dump` series does not currently support this capability, you can continue using the `DumpAccChkPoint` interface.

- Sample Specifications:
  - Mmad Sample:
    Matrix multiplication specifications: M = 16, N = 16, K = 16. Detailed information is shown in the table:
    <table>
    <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="4" align="center">Mmad</td></tr>
    </tr>
    <tr><td rowspan="3" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
    <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">ND</td></tr>
    <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">ND</td></tr>
    </tr>
    </tr>
    <tr><td rowspan="1" align="center">Sample Output</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td></tr>
    </tr>
    <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">mmad_custom</td></tr>
    </table>

- Sample Implementation:
  1) The `asc_dump` series interfaces have identical function signatures, differing only in function names. These names indicate the specific location to dump:
  ```cpp
  __aicore__ inline void asc_dump_gm(__gm__ T* input, uint32_t desc, uint32_t dumpSize);
  __aicore__ inline void asc_dump_ubuf(__ubuf__ T* input, uint32_t desc, uint32_t dumpSize);
  __aicore__ inline void asc_dump_cbuf(__cc__ T* input, uint32_t desc, uint32_t dumpSize);
  __aicore__ inline void asc_dump_l1buf(__cbuf__ T* input, uint32_t desc, uint32_t dumpSize);
  ```
  2) During the data loading phase, the `asc_dump_gm` interface is used to dump the original data of input matrix A, matrix B, and bias matrix Bias located in Global Memory (GM). After data is loaded into L0C and matrix multiplication is completed, the `asc_dump_cbuf` interface is used to dump and output the final computation results in L0C, presenting the complete process status from GM input to L0C computation completion.

## Build and Run

Execute the following steps in the sample root directory to build and run the sample.
- Configure Environment Variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package in your current environment.
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

- Sample Execution
  ```bash
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;                # Build project
  ./demo                           # Execute the generated executable program
  ```

- Build Options Description

| Option | Available Values | Description |
|--------|------------------|-------------|
| `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU Architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products; dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result
  The execution result is shown below, indicating successful accuracy comparison.
  ```bash
  [Success] Case accuracy is verification passed.
  ```