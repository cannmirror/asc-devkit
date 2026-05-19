# GmByPassDCache Class Sample

## Overview

This sample demonstrates reading data from and writing data to GM addresses without going through DCache by combining calls to the `ReadGmByPassDCache` and `WriteGmByPassDCache` interfaces. This sample reads a value and writes it to the output address after adding 100.

When multiple cores operate on GM addresses, if the data cannot be aligned to the Cache Line, the DCache approach reads and writes in Cache Line size, which causes random data overwriting by multiple cores. In this case, you can use the method of directly reading and writing GM addresses without going through DCache to avoid the random overwriting issue described above.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── gm_by_pass_dcache
│   ├── scripts
│   │   ├── gen_data.py         // Input data and golden data generation script
│   │   └── verify_result.py    // Verification script for comparing output data with golden data
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── gm_by_pass_dcache.asc   // Ascend C operator implementation & invocation sample
```

## Sample Description

- Sample functionality:
  Combine calls to `ReadGmByPassDCache` and `WriteGmByPassDCache` to:
  1. Use `ReadGmByPassDCache` to read an int32_t scalar value from a GM address
  2. Use `WriteGmByPassDCache` to write the read value plus 100 to the output GM address

- Sample specifications:
  <table>
  <caption>Table 1: Sample Specifications Description</caption>
  <tr><td rowspan="2" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 1024]</td><td align="center">int32_t</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">z</td><td align="center">[1, 1024]</td><td align="center">int32_t</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">kernel_gm_by_pass_dcache</td></tr>  
  </table>

## Build and Run

Execute the following steps in the root directory of this sample to build and run it.

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

- Sample execution

  ```bash
  mkdir -p build && cd build;                                        # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;              # Build project, default npu mode
  python3 ../scripts/gen_data.py                                     # Generate test input data
  ./demo                                                             # Execute the compiled executable program to run the sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin  # Verify if output results are correct
  ```

  For CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;  # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;  # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory and then re-run cmake.

- Build options description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  
- Execution result

  The following execution result indicates successful precision comparison:

  ```bash
  test pass!
  ```