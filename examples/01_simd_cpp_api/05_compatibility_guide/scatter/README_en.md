# Scatter Compatibility Sample

## Overview

This sample demonstrates the data scatter function, which scatters an input tensor to a result tensor based on the input tensor and destination address offset tensor.

Different hardware implementations are isolated through compile-time macros:
- Atlas A2/A3: Does not support the Scatter instruction, uses scalar move-out method (GetValue/SetValue loop) for implementation.
- Ascend 950PR/Ascend 950DT: Directly calls the Scatter instruction for implementation.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 training series products/Atlas A3 inference series products
- Atlas A2 training series products/Atlas A2 inference series products

## Directory Structure

```
├── scatter
│   ├── scripts
│   │   ├── gen_data.py         // Script for generating input data and golden data
│   │   └── verify_result.py    // Script for verifying output data matches golden data
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── scatter_custom.asc      // Ascend C sample implementation & call sample
```

## Sample Specifications

<table>
<caption>Sample Specification Table</caption>
<tr><td rowspan="1" align="center">Category</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td rowspan="2" align="center">Sample Input</td>
<td align="center">src</td><td align="center">[1, 128]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td align="center">dst_offset</td><td align="center">[1, 128]</td><td align="center">uint32</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Sample Output</td><td align="center">dst</td><td align="center">[1, 128]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">scatter_custom</td></tr>
</table>

## Sample Implementation

- Atlas A2/A3: Does not support the Scatter instruction. Uses scalar GetValue/SetValue loop to read offset addresses and source data element by element, and writes source data to destination locations to implement data scatter.
- Ascend 950PR/Ascend 950DT: Calls the Scatter instruction to scatter source data to destination tensor according to offset addresses.

## Build and Run

Execute the following steps in the root directory of this sample to build and run the sample.

- Configure environment variables

  Please select the corresponding command to configure environment variables based on the [installation method](../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on the current environment.
  - Default path, root user installed CANN package

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, non-root user installed CANN package

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, installed CANN package

    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Sample execution

  ```bash
  mkdir -p build && cd build;
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;
  python3 ../scripts/gen_data.py
  ./demo
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:

  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching compilation modes, you need to clean the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Compilation option description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2/A3 series, dav-3510 corresponds to Ascend 950PR/950DT |

- Execution result

  The following execution result indicates successful precision comparison:

  ```bash
  test pass!
  ```