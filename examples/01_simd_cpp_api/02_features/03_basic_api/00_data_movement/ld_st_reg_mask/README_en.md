# ld_st_reg_mask Example

## Overview
This example implements load/store operations from UB (Unified Buffer) to MaskReg (mask register) based on Reg programming interfaces, as well as masked store operations. The example uses LoadAlign, StoreAlign, CreateMask, and Duplicate interfaces.

## Supported Products
- Ascend 950PR/Ascend 950DT

## Directory Structure
```
├── ld_st_reg_mask
│   ├── scripts
│   │   └── gen_data.py                // Input data and ground truth data generation script
│   ├── figures
│   │   └── ld_st_reg_mask.png         // Example schematic
│   ├── CMakeLists.txt                 // Build project file
│   ├── data_utils.h                   // Data read/write functions
│   ├── README.md                      // Example introduction
│   └── ld_st_reg_mask.asc             // AscendC example implementation & invocation example
```

## Example Description
- Example Functionality:
  The example takes a uint8_t vector with 1024 elements as input, uses the first 256 bits as a mask for Duplicate computation, then sets all bits in the mask register to 1, and saves the 256-bit value from the mask register to UB.
- Example Specifications:
  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="3" align="center">AIV Example</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 1024]</td><td align="center">uint8_t</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">y</td><td align="center">[1, 1024]</td><td align="center">uint8_t</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">ld_st_reg_mask</td></tr>
  </table>
- Example Implementation:
<p align="center">
  <img src="figures/ld_st_reg_mask.png" width="1000">
</p>

  1. In the CopyVF function, call LoadAlign interface to transfer 256 bits (32*uint8_t) of data from UB to MaskReg to enable dynamic mask setting. In this example, 32 uint8_t values are set to 1,0,...,1, where the first and last values are 1 (b'00000001). Since the chip reads numbers from the LSB, these 32 values are filled into MaskReg as b'1000...1...000.
  2. Call Duplicate interface for data filling. MaskReg can indicate which elements participate in computation. From step 1, bits 1 and 249 in MaskReg are 1. Using this mask, only elements 1 and 249 in RegTensor are filled with value 2.
  3. Use StoreAlign interface to save the results from RegTensor to UB.
  4. Set all bits in MaskReg to 1, then save the data from MaskReg to UB (address = save address from step 3 + 256B) via StoreAlign interface, implementing the functionality of storing MaskReg data in UB. This corresponds to 32 uint8_t values, where each bit of each value is 1, so each value is 255 (0xFFFF..FF).
  5. Store the MaskReg from step 4 to UB 30 consecutive times, with UB starting address at save address from step 3 + 2*256B, saving 32 elements each time. This step demonstrates using the POST_MODE_UPDATE mode of the Store interface.
  
  - Invocation Implementation
    Use the kernel call operator <<<>>> to invoke the kernel function.
    
## Build and Run
Execute the following steps in the root directory of this example to build and run the example.
- Configure Environment Variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on your current environment.
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
    
- Example Execution
  ```bash
  mkdir -p build && cd build;                                               # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;                      # Build project (default npu mode)
  python3 ../scripts/gen_data.py                                            # Generate test input data
  ./demo                                                                    # Execute the compiled executable to run the example
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Build Option Description

| Option | Available Values | Description |
| | ---------------------------| -----------------------------| ---------------------------------------------------|
| `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
| `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result
  When the execution result shows the following, it indicates successful precision comparison.
  ```bash
  test pass!
  ```