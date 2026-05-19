# SetLoadDataBoundary Compatibility Sample

## Overview

  This sample implements setting boundary values for L1 Buffer, isolating different hardware implementations through compile-time macros.

  Atlas A2/A3 series products set the boundary value of L1 Buffer through the `SetLoadDataBoundary` interface, while the new architecture of Ascend 950PR/950DT does not support this interface. To implement the same data circular read function, you need to manually split the Load3D instruction to achieve data circular read wrap-around through adjusting the destination operand address offset.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 training series products/Atlas A3 inference series products
- Atlas A2 training series products/Atlas A2 inference series products

## Directory Structure

```
├── set_loaddata_boundary
│   ├── scripts
│   │   ├── gen_data.py         // Script for generating input data and golden data
│   │   └── verify_result.py    // Script for verifying output data matches golden data
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── set_loaddata_boundary.asc    // AscendC sample implementation & call sample
```

## Sample Specifications

<table>
<caption>Sample Specification Table</caption>
<tr><td rowspan="1" align="center">Category</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td rowspan="3" align="center">Sample Input</td></tr>
<tr><td align="center">x</td><td align="center">[32, 32]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td align="center">y</td><td align="center">[32, 32]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Sample Output</td><td align="center">z</td><td align="center">[32, 32]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">set_loaddata_boundary</td></tr>
</table>

## Sample Implementation
   
- **Atlas A2/A3 Training/Inference Series Products**: Set the boundary value of L1 Buffer to 1024 bytes through the SetLoadDataBoundary interface. When the Load3D instruction processes the source operand, if the address of the source operand in L1 Buffer exceeds the set boundary, data will be automatically read from the starting address of L1 Buffer, implementing the data circular read function.

- **Ascend 950PR/950DT**: The new architecture hardware has removed the registers related to L1 Buffer boundary value setting and no longer supports the SetLoadDataBoundary interface. To implement the same data circular read function, you need to manually split the Load3D interface into multiple instructions and achieve manual wrap-around by adjusting the address offset of the destination operand.

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
  mkdir -p build && cd build;                                               # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;                      # Build project
  python3 ../scripts/gen_data.py                                            # Generate test input data
  ./demo                                                                    # Execute the compiled executable program
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output results are correct
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:

  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;  # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;  # NPU simulation mode
  ```

  > **Note**: Before switching compilation modes, you need to clean the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Compilation option description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2/A3 series, dav-3510 corresponds to Ascend 950PR/950DT |

- Execution result

  The following execution result indicates successful precision comparison.

  ```bash
  test pass!
  ```