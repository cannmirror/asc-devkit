# L1 to GM Data Copy Compatibility Sample

## Overview

This sample demonstrates the end-to-end process of copying data from L1 to GM, isolating different hardware implementations through compile-time macros.

- Atlas A2/A3 training/inference series products directly use the DataCopy interface for data transfer.
- Ascend 950PR/950DT new architecture does not support direct transfer. Data is output to L0C Buffer through Mmad matrix multiplication computation, then transferred to GM from L0C Buffer through Fixpipe.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 training series products/Atlas A3 inference series products
- Atlas A2 training series products/Atlas A2 inference series products

## Directory Structure

```
├── data_copy_l1togm
│   ├── scripts
│   │   ├── gen_data.py         // Script for generating input data and golden data
│   │   └── verify_result.py    // Script for verifying output data matches golden data
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── data_copy_l1togm.asc    // AscendC sample implementation & call sample
```

## Sample Specifications

This sample has different implementation logic based on different architectures:

### Atlas A2/A3 Training/Inference Series Products

- Sample specifications:
  <table>
  <caption>Atlas A2/A3 Training/Inference Series Products Sample Specification Table</caption>
  <tr><td rowspan="1" align="center">Category</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td rowspan="1" align="center">Sample Input</td>
  <td align="center">x</td><td align="center">[64, 128]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">z</td><td align="center">[64, 128]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">data_copy_l1togm</td></tr>
  </table>
- Sample implementation: Calls the DataCopy instruction to implement data transfer from GM to L1 and then to GM.

### Ascend 950PR/950DT

- Sample specifications:
  <table>
  <caption>Ascend 950PR/950DT Product Sample Specification Table</caption>
  <tr><td rowspan="1" align="center">Category</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td rowspan="3" align="center">Sample Input</td></tr>
  <tr><td align="center">x</td><td align="center">[64, 128]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">[128, 128]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">z</td><td align="center">[64, 128]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">data_copy_l1togm</td></tr>
  </table>
- Sample implementation: The new architecture of Ascend 950PR/950DT does not support direct transfer from L1 to GM. In cube-only scenarios, matrix multiplication can be used to achieve the transfer effect. Allocate an additional identity matrix in GM (original matrix * identity matrix = original matrix), output to L0C Buffer through Mmad matrix multiplication computation, then transfer to GM from L0C Buffer through Fixpipe. The data flow is: GM -> A1/B1 -> L0A/L0B -> L0C -> GM.

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
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # Default NPU mode
  python3 ../scripts/gen_data.py                                            # Generate test input data
  ./demo                                                                    # Execute the compiled executable program
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output results are correct
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
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2/A3 series, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution result

  The following execution result indicates successful precision comparison:

  ```bash
  test pass!
  ```