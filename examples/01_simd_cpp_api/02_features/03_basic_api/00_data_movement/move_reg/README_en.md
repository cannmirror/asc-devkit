# move_reg Example

## Overview
This example demonstrates and validates moving data from RegTensor (the basic unit for Reg vector computation) to MaskReg (mask register) using the MaskGenWithRegTensor interface, based on the Reg programming interface.

## Supported Products
- Ascend 950PR/Ascend 950DT

## Directory Structure
```
├── move_reg
│   ├── scripts
│   │   └── gen_data.py                // Input data and golden data generation script
│   ├── CMakeLists.txt                 // Build configuration file
│   ├── data_utils.h                   // Data read/write functions
│   ├── move_reg.asc                   // AscendC example implementation & invocation example
│   └── README.md                      // Example introduction
```

## Example Description
- Example Function:
  Demonstrates and validates moving data from RegTensor to MaskReg. Input 64 int32-type cond data, use MaskGenWithRegTensor&lt;int32, 0&gt; to extract 64-bit data from the first 8 bytes of condReg, and fill it into a 256-bit MaskReg in downsampling mode (every 4 mask bits correspond to 1 src bit), outputting 32 bytes (256 bits) of mask data to UB (Unified Buffer).

  - Example Specifications:
    <table>
    <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="3" align="center">AIV Example</td></tr>
    <tr><td rowspan="2" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
    <tr><td align="center">cond</td><td align="center">[1, 64]</td><td align="center">int32</td></tr>
    <tr><td rowspan="1" align="center">Example Output</td><td align="center">maskOut</td><td align="center">[32]</td><td align="center">uint8</td></tr>
    <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="3" align="center">move_reg</td></tr>
    </table>
  - Example Implementation:
    Inside the MaskGenVF function:
    1. Use LoadAlign to load cond data into RegTensor with alignment
    2. Extract 64-bit data from the first 8 bytes of condReg through MaskGenWithRegTensor&lt;int32, 0&gt;
    3. Fill into 256-bit MaskReg in downsampling mode: every 4 mask bits correspond to 1 src bit
    4. Use StoreAlign to write mask data back to UB
    - Downsampling Result:
      cond[0]=0x00000001, cond[1]=0xFFFFFFFF
      src 64 bits: bit0=1, bit32..63=1..1
      mask 256 bits: byte[0]=0x0f, byte[16..31]=0xff..0xff
    - Invocation Implementation
      Use the kernel call operator `<<<>>>` to invoke the kernel function, starting 1 core.

## Build and Run
Execute the following steps in the root directory of this example to build and run the example.
- Configure Environment Variables
  Please select the corresponding command to configure environment variables according to the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on the current environment.
  - Default path, root user installed CANN software package
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, non-root user installed CANN software package
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, installed CANN software package
    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Example Execution
  ```bash
  mkdir -p build && cd build;                                                    # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;                           # Build project (default npu mode)
  python3 ../scripts/gen_data.py                                                 # Generate test input data
  ./demo                                                                        # Execute the compiled executable program to run the example
  ```

  When using CPU debugging or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debugging mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clean the cmake cache. Execute `rm CMakeCache.txt` in the build directory and then re-run cmake.

- Build Option Description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debugging, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result
  The execution result is as follows, indicating precision comparison succeeded.
  ```bash
  test pass!
  ```