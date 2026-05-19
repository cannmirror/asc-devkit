# optimize_datacopy_loop_mode Sample

## Overview
Use loop mode to reduce the number of DataCopyPad instructions when using the DataCopyPad API.

## Supported Products
- Ascend 950PR/Ascend 950DT

## Directory Structure
```
├── optimize_datacopy_loop_mode
│   ├── scripts
│   │   ├── gen_data.py                         // Script for generating input data and golden data
│   │   └── verify_result.py                    // Script for verifying output data against golden data
│   ├── CMakeLists.txt                          // Build configuration file
│   ├── data_utils.h                            // Data read and write functions
│   └── optimize_datacopy_loop_mode.asc         // AscendC operator implementation and sample invocation
```

## Operator Description
- Operator Function:  
  Use loop mode to reduce the number of DataCopyPad instructions when using the DataCopyPad API.
- Operator Specifications:
  <table>
  <tr><td rowspan="1" align="center">Operator Type(OpType)</td><td colspan="3" align="center">AIC Operator</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">Operator Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">1024</td><td align="center">float</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Operator Output</td><td align="center">y</td><td align="center">1024</td><td align="center">float</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">optimize_datacopy_loop_mode</td></tr>
  </table>
- Operator Implementation:  
  Use loop mode to reduce the number of DataCopyPad instructions when using the DataCopyPad API.
  
  - Invocation Implementation  
    Use the kernel call operator `<<<>>>` to invoke the kernel function.
    
## Build and Run
Execute the following steps in the root directory of this sample to build and run the operator.
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
    
- Sample Execution
  ```bash
  mkdir -p build && cd build;                                               # Create and enter build directory
  cmake ..;make -j;                                                         # Build the project
  python3 ../scripts/gen_data.py                                            # Generate test input data
  ./demo                                                                    # Execute the compiled executable to run the sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output results against golden data to confirm algorithm correctness
  ```
  The following output indicates successful accuracy comparison:
  ```bash
  test pass!
  ```