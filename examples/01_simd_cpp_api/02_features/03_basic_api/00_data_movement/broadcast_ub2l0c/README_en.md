# BroadCastVecToMM Example

## Overview

This example implements data broadcast transfer based on BroadCastVecToMM, broadcasting data located on UB (Unified Buffer) and transferring it to CO1 (L0C Buffer).

## Supported Products

- Atlas Inference Series Products AI Core

## Directory Structure

```
├── broadcast_ub2l0c
│   ├── scripts
│   │   ├── gen_data.py               // Input data and golden data generation script
│   │   └── verify_result.py          // Verification script for comparing output data with golden data
│   ├── CMakeLists.txt                // Build project file
│   ├── data_utils.h                  // Data read/write functions
│   └── broad_cast_vec_to_mm.asc      // Ascend C example implementation & invocation example
```

## Example Description

- Example Functionality:

  This example broadcasts data with shape [1, 16] from UB (Unified Buffer) to [16, 16] and transfers it to CO1 (L0C Buffer). Refer to BroadCastVecToMM for interface documentation.

- Example Specifications:

  <table border="2">
  <tr><td rowspan="3" align="center">Example Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 16]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">Example Output</td></tr>
  <tr><td align="center">z</td><td align="center">[16, 16]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">broad_cast_vec_to_mm_custom</td></tr>
  </table>

- Example Implementation:

  - Kernel Implementation
    - Calls DataCopy basic API to transfer data from GM (Global Memory) to UB (Unified Buffer), and transfers broadcast data out to GM (Global Memory).
    - Calls BroadCastVecToMM basic API to broadcast data on UB (Unified Buffer) from [1, 16] to [16, 16] and transfer to CO1 (L0C Buffer).
    - Calls DataCopy enhanced data transfer interface to transfer broadcast data from CO1 (L0C Buffer) to UB (Unified Buffer).

  - Invocation Implementation

    Uses the kernel call operator <<<>>> to invoke the kernel function.

## Compilation and Execution

Execute the following steps in the root directory of this example to compile and run the example.
- Configure Environment Variables
  Select the appropriate environment variable configuration command based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on the current environment.
  - Default path, CANN software package installed by root user
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, CANN software package installed by non-root user
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, CANN software package installation
    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Example Execution
  ```bash
  mkdir -p build && cd build;                                               # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2002 ..;make -j;                      # Build project, default npu mode
  python3 ../scripts/gen_data.py                                            # Generate test input data
  ./demo                                                                    # Execute compiled executable to run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output correctness, confirm algorithm logic
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2002 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2002 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching compilation modes, clean the cmake cache by executing `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Compilation Options Description

  | Option | Possible Values | Description |
  |--------|-----------------|-------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2002` | NPU architecture: dav-2002 corresponds to Atlas Inference Series Products AI Core |

- Execution Result

  The following result indicates successful precision comparison:
  ```bash
  test pass!
  ```