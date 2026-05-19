# Fill Compatibility Sample

## Overview

This sample demonstrates how to use the Fill interface to initialize L0A Buffer and L0B Buffer, isolating different hardware implementations through compile-time macros.

- In Atlas A2/A3 series products, the Fill interface can be used directly to initialize L0A/L0B Buffer.
- However, in the Ascend 950PR/Ascend 950DT platform, since the hardware instructions related to L0A Buffer/L0B Buffer initialization have been removed, the Fill interface cannot be used directly to initialize L0A/L0B Buffer. Instead, you need to first initialize L1 Buffer, then transfer the initialized data to L0A/L0B to indirectly complete the initialization of L0A/L0B Buffer.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 training series products/Atlas A3 inference series products
- Atlas A2 training series products/Atlas A2 inference series products

## Directory Structure

```
├── fill
│   ├── scripts
│   │   ├── gen_data.py         // Script for generating input data and golden data
│   │   └── verify_result.py    // Script for verifying output data matches golden data
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── fill.asc                // AscendC sample implementation & call sample
```

## Sample Specifications

<table>
<caption>Sample Specification Table</caption>
<tr><td rowspan="3" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">[128, 128]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td align="center">y</td><td align="center">[128, 64]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Sample Output</td><td align="center">z</td><td align="center">[128, 64]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">fill</td></tr>
</table>

### Sample Implementation

  1. First initialize L0A Buffer and L0B Buffer. The method varies by hardware architecture:
     - Atlas A2/A3 training/inference series products: Call the `Fill` interface to directly initialize L0A Buffer and L0B Buffer to a specified value (initialized to 1 in this sample).
     - Ascend 950PR/950DT: Use the `Fill` interface to initialize L1 Buffer to a specified value (initialized to 1 in this sample), then transfer to L0A Buffer and L0B Buffer through the `LoadData` interface.
  2. Call the `Mmad` interface to perform matrix multiplication computation.
  3. Transfer the result to Global Memory through the `Fixpipe` interface.

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
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;                      # Build project (Atlas A2/A3 series products)
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

  For Ascend 950PR/950DT compilation:

  ```bash
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;                      # Build project (Ascend 950PR/950DT)
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