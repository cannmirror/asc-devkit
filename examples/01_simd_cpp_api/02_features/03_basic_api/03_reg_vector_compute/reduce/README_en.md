# Reduce Example

## Overview

This example implements the Reduce operation using the Reg programming interface, primarily calling the Reduce interface (SUM mode).

- The Reduce interface supports SUM/MAX/MIN reduction modes. This example uses SUM mode as a demonstration.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── reduce
│   ├── scripts
│   │   ├── gen_data.py                // Input data and ground truth generation script
│   ├── CMakeLists.txt                 // Build configuration file
│   ├── data_utils.h                   // Data read/write functions
│   ├── reduce.asc                     // AscendC example implementation & invocation example
│   └── README.md                      // Example introduction
```

## Example Description

- Example functionality:

  Performs reduction sum operation on the input vector. The vector shape is [1, 256] with float data type. The output is the reduction sum result.

- Example specifications:

  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="3" align="center">AIV Example</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 256]</td><td align="center">float</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">y</td><td align="center">[1]</td><td align="center">float</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Name</td><td colspan="4" align="center">reduce</td></tr>
  </table>

- Example implementation:

   The ReduceSumVF function calls the Reduce interface for reduction calculation:

   - Uses LoadAlign to load data into registers
   - Uses Reduce(ReduceType::SUM) to sum all elements within a single repeat
   - Uses Add to accumulate partial sums from multiple repeats
   - Uses StoreAlign to write the final result back to UB

  - Invocation implementation

    Uses the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this example to build and run the example.

- Configure environment variables

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

- Execute the example

  ```bash
  mkdir -p build && cd build;                                               # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;                      # Build project (default npu mode)
  python3 ../scripts/gen_data.py                                            # Generate test input data
  ./demo                                                                    # Execute the compiled executable to run the example
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:

  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clear the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build options description

| Option | Possible Values | Description |
| ------------| -----------------------------| ---------------------------------------------------|
| `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
| `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution result

  The following execution result indicates successful precision comparison.

  ```bash
  test pass!
  ```