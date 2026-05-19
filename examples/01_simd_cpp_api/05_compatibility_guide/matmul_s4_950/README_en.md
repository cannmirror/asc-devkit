# int4b_t Data Type Matmul Compatibility Sample

## Overview

End-to-end compatibility sample for int4b_t matrix multiplication computation.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── matmul_s4_950
│   ├── scripts
│   │   ├── gen_data.py         // Script for generating input data and golden data
│   │   └── verify_result.py    // Script for verifying output data matches golden data
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_s4.asc    // AscendC operator implementation & call sample
```

## Operator Description

- Operator function:
  End-to-end sample for implementing L0A Buffer and L0B Buffer initialization values
- Operator specifications:
  <table>
  <tr><td rowspan="1" align="center">Operator Type (OpType)</td><td colspan="4" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="7" align="center">Operator Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">256 * 256</td><td align="center">int8_t</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">256 * 256</td><td align="center">int8_t</td><td align="center">ND</td></tr>
  </tr>
  <tr><td align="center">x4</td><td align="center">256 * 128</td><td align="center">int4b_t</td><td align="center">ND</td></tr>
  </tr>
    <tr><td align="center">y4</td><td align="center">128 * 64</td><td align="center">int4b_t</td><td align="center">ND</td></tr>
  </tr>
    <tr><td align="center">tiling</td><td align="center">64</td><td align="center">int32_t</td><td align="center">ND</td></tr>
  </tr>
    </tr>
    <tr><td align="center">workspace</td><td align="center">131072</td><td align="center">uint64_t</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Operator Output</td><td align="center">z</td><td align="center">256 * 256</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">matmul_s4</td></tr>
  </table>
- Operator implementation:
    The Cube computation unit has removed the int4b_t data type. Users can perform Cast conversion from int4b_t to int8_t on the Vector Core in MIX mode on the operator side, then transfer to L1 via UB for Mmad computation.

  - Call implementation
    Uses the kernel call operator `<<<>>>` to call the kernel function.

## Build and Run

Execute the following steps in the root directory of this sample to build and run the operator.

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
  cmake ..;make -j;                                                         # Build project
  python3 ../scripts/gen_data.py                                            # Generate test input data
  ./demo                                                                    # Execute the compiled executable program
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output results are correct
  ```

  The following execution result indicates successful precision comparison.

  ```bash
  test pass!
  ```