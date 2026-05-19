# DataCopy Data Slice Transfer Example

## Overview
This example implements data slice transfer based on DataCopy, extracting subsets of multi-dimensional Tensor data for transfer between GM (Global Memory) and UB (Unified Buffer).

## Supported Products
- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products


## Directory Structure
```
├── data_copy_gm2ub_slice
│   ├── scripts
│   │   ├── gen_data.py         // Input data and ground truth data generation script
│   │   └── verify_result.py    // Verification script for comparing output data with ground truth
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── slice.asc               // Ascend C example implementation & invocation example
```

## Example Description
- Example Functionality:
  Implements data slice transfer, supporting slice-based data movement. Extracts subsets from a 2D source operand Tensor[3, 87] (extracting 4 data segments: [0, 16:40], [0, 47:71], [2, 16:40], [2, 47:71], totaling 96 float32 data elements) and transfers them contiguously to a 2D destination operand Tensor[2, 48]. For interface documentation, refer to [Slice Data Transfer](../../../../../../docs/api/context/切片数据搬运.md).

- Example Specifications:

  <table>
  <tr><td rowspan="2" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">slice parameters</td></tr>
  <tr><td align="center">x</td><td align="center">[3, 87]</td><td align="center">float32</td><td align="center">ND</td><td align="center">[[0, 16:40], [0, 47:71]], [[2, 16:40], [2, 47:71]]</td></tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">y</td><td align="center">[2, 48]</td><td align="center">float32</td><td align="center">ND</td><td align="center">\</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">kernel_slice</td></tr>
  </table>

- Example Implementation:
  - Kernel Implementation
    The computation logic is: input data needs to be transferred from GM (Global Memory) to UB (Unified Buffer) according to slice parameters, then transferred back to external GM (Global Memory).
    
    For detailed interface description, refer to Ascend C API DataCopy Slice Data Transfer.

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
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;                      # Build project, default npu mode
  python3 ../scripts/gen_data.py                                            # Generate test input data
  ./demo                                                                    # Execute the compiled executable to run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output results, confirm algorithm correctness
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  Example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Build Option Description

  | Option | Available Values | Description |
  | ----------------| -----------------------------| --------------------------------------------------------------------------------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  When the execution result shows the following, it indicates successful precision comparison.
  ```bash
  test pass!
  ```