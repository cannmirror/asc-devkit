# Muls Sample

## Overview
This sample implements Muls operation using the Reg programming interface, primarily calling the Muls interface.
- Adds/Maxs/Mins/LeakyRelu interfaces can also reference this sample

## Supported Products
- Ascend 950PR/Ascend 950DT

## Directory Structure
```
├── muls
│   ├── scripts
│   │   ├── gen_data.py                // Input data and ground truth generation script
│   ├── CMakeLists.txt                 // Build configuration file
│   ├── data_utils.h                   // Data read/write functions
│   ├── muls.asc                       // AscendC sample implementation & invocation
│   └── README.md                      // Sample introduction
```

## Sample Description
- Sample Function:
  Perform scalar multiplication on a vector. The vector has 256 elements, data type is float, scalar value is 2.0.
- Sample Specifications:
  <table>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="3" align="center">AIV Sample</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 256]</td><td align="center">float</td></tr>
  <tr><td align="center">scalar</td><td align="center">[1]</td><td align="center">float</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">z</td><td align="center">[1, 256]</td><td align="center">float</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">muls</td></tr>
  </table>
- Sample Implementation:
   The MulsVF function calls the Muls interface for computation and writes results back to UB
  - Invocation Implementation
    Use the kernel caller <<<>>> to invoke the kernel function.

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
  mkdir -p build && cd build;                                               # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;                                     # Build project (default npu mode)
  python3 ../scripts/gen_data.py                                            # Generate test input data
  ./demo                                                                    # Execute the compiled program
  ```

  For CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Build Options

| Option                       | Values                       | Description                                                   |
| ---------------------------- | ---------------------------- | ------------------------------------------------------------- |
| `CMAKE_ASC_RUN_MODE`         | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation           |
| `CMAKE_ASC_ARCHITECTURES`    | `dav-3510` (default)         | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result
  The following output indicates successful accuracy verification:
  ```bash
  test pass!
  ```