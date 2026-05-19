# DataCopy ub2l1 Example

## Overview

This example implements data transfer from UB (Unified Buffer) to L1 (L1 Buffer) based on DataCopy in the Mmad matrix multiplication scenario.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── data_copy_ub2l1
│   ├── scripts
│   │   ├── gen_data.py                   // Input data and ground truth data generation script
│   ├── CMakeLists.txt                    // Build project file
│   ├── data_utils.h                      // Data read/write functions
│   └── data_copy_ub2l1.asc               // AscendC example implementation & invocation example
```

## Example Description

- Example Functionality:
  Transfers data from UB (Unified Buffer) to L1 (L1 Buffer), then performs Mmad matrix multiplication calculation, and finally transfers data out to GM (Global Memory) through Fixpipe. For interface documentation, refer to [On-the-fly Basic Data Transfer](../../../../../../docs/api/context/基础数据搬运.md).
- Example Specifications:
  <table>
  <tr><td rowspan="3" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[32, 32]</td><td align="center">half</td><td align="center">NZ</td></tr>
  <tr><td align="center">y</td><td align="center">[32, 32]</td><td align="center">half</td><td align="center">NZ</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">z</td><td align="center">[32, 32]</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">data_copy_ub2l1</td></tr>
  </table>
- Example Implementation:
  - Transfer data from GM (Global Memory) to UB (Unified Buffer).
  - Transfer data from UB (Unified Buffer) to L1 (L1 Buffer).
  - Call basic API LoadData to transfer data from L1 (L1 Buffer) to A2 (L0A Buffer) and B2 (L0B Buffer).
  - Call basic API Mmad to perform matrix multiplication calculation.
  - Call basic API Fixpipe to transfer data from CO1 (L0C Buffer) to GM (Global Memory).
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
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;                      # Build project, default npu mode
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
  | ----------------| -----------------------------| --------------------------------------------------------------------------------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU architecture: dav-3510 corresponds to Ascend 950PR/950DT |

- Execution Result

  When the execution result shows the following, it indicates successful precision comparison.
  ```bash
  test pass!
  ```