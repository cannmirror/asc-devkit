# Arange Example

## Overview
This example demonstrates Arange operation using the Reg programming interface, primarily calling the Arange interface.
- The Arange interface generates an incrementing/decrementing index sequence starting from the provided scalar value
- This example demonstrates incrementing mode with a starting value of 0

## Supported Products
- Ascend 950PR/Ascend 950DT

## Directory Structure
```
├── arange
│   ├── scripts
│   │   │   ├── gen_data.py            // Golden data generation script
│   ├── CMakeLists.txt                 // Build project file
│   ├── data_utils.h                   // Data read/write functions
│   ├── arange.asc                     // Ascend C implementation & invocation example
│   └── README.md                      // Example introduction
```

## Example Description
- Example Function:
  Generates an incrementing index sequence starting from 0, with vector shape [1, 256] and data type float.
  Arange generates incrementing indices starting from 0: {0, 1, 2, 3, ..., 255}.
- Example Specifications:
  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="3" align="center">AIV Example</td></tr>
  <tr><td rowspan="2" align="center">Example Output</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">y</td><td align="center">[1, 256]</td><td align="center">float</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">arange</td></tr>
  </table>
- Example Implementation:
  The ArangeVF function calls the Arange interface to generate indices:
  - Uses the Arange interface to generate incrementing indices, each repeat generates oneRepeatSize indices
  - Updates the starting value after each repeat
  - Invocation Implementation
    Use the kernel launch operator <<<>>> to invoke the kernel function.

## Notes
- The Arange interface does not require input data, only needs to specify the starting scalar value
- Each repeat generates oneRepeatSize incrementing indices; for multiple repeats, the starting value must be manually updated
- Default template parameter is incrementing mode (IndexOrder::INCREASE_ORDER)

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