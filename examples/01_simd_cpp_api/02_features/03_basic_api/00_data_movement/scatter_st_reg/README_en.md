# scatter_st_reg Example

## Overview
This example implements discrete data store functionality (dispersing elements to UB (Unified Buffer)) based on the Reg programming interface, using the Scatter interface.

## Supported Products
- Ascend 950PR/Ascend 950DT

## Directory Structure
```
├── scatter_st_reg
│   ├── scripts
│   │   ├── gen_data.py                // Input data and golden data generation script
│   ├── CMakeLists.txt                 // Build configuration file
│   ├── data_utils.h                   // Data read/write functions
│   ├── scatter_st_reg.asc             // AscendC example implementation & invocation example
│   └── README.md                      // Example introduction
```

## Example Description
- Example Function:
  Disperse elements from source data to destination addresses according to indices.

  **Scatter Mode**
  - Source data count is 128 elements, index data count is 128 elements, output data count is 128 elements. The generated index here is a reverse order index of 128 elements, so the stored data count is also 128.
  - Example Specifications:
    <table>
    <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="3" align="center">AIV Example</td></tr>
    <tr><td rowspan="2" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
    <tr><td align="center">x</td><td align="center">[1, 128]</td><td align="center">half</td></tr>
    <tr><td></td><td align="center">index</td><td align="center">[1, 128]</td><td align="center">uint16_t</td></tr>
    <tr><td rowspan="1" align="center">Example Output</td><td align="center">y</td><td align="center">[1, 128]</td><td align="center">half</td></tr>
    <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">scatter_st_reg</td></tr>
    </table>
  - Example Implementation:
    Inside the ScatterVF function, call the Scatter interface to scatter write data to UB by element index.
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
  mkdir -p build && cd build;                                                         # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;                                # Build project (default npu mode)
  python3 ../scripts/gen_data.py;                                                     # Generate test input data
  ./demo                                                                              # Execute the compiled executable program to run the example
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