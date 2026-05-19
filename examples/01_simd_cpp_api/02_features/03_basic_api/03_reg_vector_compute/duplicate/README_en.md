# Duplicate Example

## Overview
This example demonstrates Duplicate operation (scalar fill mode) using the Reg programming interface, primarily calling the Duplicate interface.
- `Duplicate(dstReg, scalarValue, mask)` interface copies a scalar value multiple times and fills it into a vector
- `Duplicate(dstReg, srcReg, mask)` interface copies the first element of the source RegTensor multiple times and fills it into a vector

## Supported Products
- Ascend 950PR/Ascend 950DT

## Directory Structure
```
├── duplicate
│   ├── scripts
│   │   │   ├── gen_data.py            // Golden data generation script
│   ├── CMakeLists.txt                 // Build project file
│   ├── data_utils.h                   // Data read/write functions
│   ├── duplicate.asc                  // Ascend C implementation & invocation example
│   └── README.md                      // Example introduction
```

## Example Description
- Example Function:
  Fills the scalar value 3.14 to each position of the output vector with 256 elements of type float.
- Example Specifications:
  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="3" align="center">AIV Example</td></tr>
  <tr><td rowspan="2" align="center">Example Output</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">y</td><td align="center">[1, 256]</td><td align="center">float</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">duplicate</td></tr>
  </table>
- Example Implementation:
  The DuplicateScalarVF function calls the Duplicate interface for scalar filling:
  - Uses the Duplicate interface to fill the scalar value to each element of dstReg
  - Uses StoreAlign to write results back to UB
  - Invocation Implementation
    Use the kernel launch operator <<<>>> to invoke the kernel function.

## Build and Run
Execute the following steps in the example root directory to build and run the example.
- Configure Environment Variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development toolkit on your current environment.
  - Default path, CANN package installed by root user
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, CANN package installed by non-root user
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Custom path install_path, CANN package installed
    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Example Execution
  ```bash
  mkdir -p build && cd build;                                               # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;                      # Build project (default npu mode)
  python3 ../scripts/gen_data.py                                            # Generate test golden data
  ./demo                                                                    # Execute the compiled program
  ```

  For CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching compilation modes, clear the cmake cache by running `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Compilation Options Description

| Option                      | Available Values             | Description                                                   |
| ---------------------------| -----------------------------| --------------------------------------------------------------|
| `CMAKE_ASC_RUN_MODE`       | `npu` (default), `cpu`, `sim`| Run mode: NPU execution, CPU debug, NPU simulation            |
| `CMAKE_ASC_ARCHITECTURES`  | `dav-3510`                   | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  The following output indicates successful precision comparison:
  ```bash
  test pass!
  ```