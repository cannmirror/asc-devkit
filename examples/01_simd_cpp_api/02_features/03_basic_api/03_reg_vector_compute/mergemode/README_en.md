# MergeMode Sample

## Overview
This sample verifies the behavior of MaskMergeMode::MERGING mode using the Reg programming interface, demonstrating the mechanism where inactive mask bits retain the original dstReg value, using the Max interface for verification.

## Supported Products
- Ascend 950PR/Ascend 950DT

## Directory Structure
```
├── mergemode
│   ├── scripts
│   │   ├── gen_data.py                // Input data and ground truth generation script
│   ├── CMakeLists.txt                 // Build configuration file
│   ├── data_utils.h                   // Data read/write functions
│   ├── mergemode.asc                  // AscendC sample implementation & invocation
│   └── README.md                      // Sample introduction
```

## Sample Description
- Sample Function:
  Verifies `MERGING` mode: when mask is inactive, the corresponding bits of dstReg retain their original values instead of participating in computation. Input is 200 negative numbers (not a multiple of VL), output is 2 (original dstReg value retained for inactive bits).

  **Verification Principle**
  - Input: 200 float negative numbers (-100 to -1), VL=256 Byte, total 4 repeats
  - Repeat 0-3: Before each repeat, Duplicate initializes dstReg=2
  - Max MERGING: Active bits = max(negative, negative) = negative, inactive bits = retain dstReg original value = 2
  - Repeat 3 (200 elements, last repeat has only 8 active bits):
    - yAddr[192:200] = negative numbers (8 active bits)
    - yAddr[200:256] = 2 (56 inactive bits, MERGING retains dstReg original value)
  - ReduceMax(yAddr[0:256]) = 2, verifying that MERGING mode indeed retains dstReg original value for inactive bits

  - Sample Specifications:
    <table>
    <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="3" align="center">AIV Sample</td></tr>
    <tr><td rowspan="2" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
    <tr><td align="center">x</td><td align="center">[1, 200]</td><td align="center">float (negative)</td></tr>
    <tr><td rowspan="1" align="center">Sample Output</td><td align="center">y</td><td align="center">[1, 8]</td><td align="center">float</td></tr>
    <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">mergemode</td></tr>
    </table>

  - Sample Implementation:
    1. **MaxMergeModeVF**:
       - Each loop iteration first calls `Duplicate(dstReg, 2)` to initialize dstReg
       - Use `UpdateMask` to handle non-VL-multiple data
       - Use `Max`: active bits compute max value, inactive bits retain dstReg original value = 2
       - Use `StoreAlign(allMask)` to write entire VL, verifying dstReg inactive bits are indeed 2
    2. **ReduceMaxVF**: Perform ReduceMax reduction on yAddr[0:256], result should be 2 (value retained for inactive bits)
    3. Output is 32B-aligned 8 floats, ReduceMax result is in the first element, expected value is 2

    - Invocation Implementation
      Use the kernel caller `<<<>>>` to invoke the kernel function, launching 1 core.

## Build and Run
Execute the following steps in the sample root directory to build and run the sample.
- Configure Environment Variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on your system.
  - Default path, CANN package installed by root user
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, CANN package installed by non-root user
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Custom path (install_path), CANN package installed
    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Sample Execution
  ```bash
  mkdir -p build && cd build;                                                    # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;                           # Build project (default npu mode)
  python3 ../scripts/gen_data.py                                                 # Generate test input data
  ./demo                                                                         # Execute the compiled program
  ```

  For CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Build Options

| Option | Values | Description |
|--------|--------|-------------|
| `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
| `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  The following output indicates successful accuracy verification:
  ```bash
  test pass!
  ```