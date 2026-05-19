# optimize_datacopy_nddma Sample

## Overview
When performing non-aligned data copy, use nddma copy to reduce the number of copy instructions.

## Supported Products
- Ascend 950PR/Ascend 950DT

## Directory Structure
```
├── optimize_datacopy_nddma
│   ├── scripts
│   │   ├── gen_data.py                     // Input data and golden data generation script
│   │   └── verify_result.py                // Verification script for output data and golden data
│   ├── CMakeLists.txt                      // Build project file
│   ├── data_utils.h                        // Data read/write functions
│   └── optimize_datacopy_nddma.asc         // AscendC operator implementation & call sample
```

## Operator Description
- Operator Function:
  When performing non-aligned data copy, use nddma copy to reduce the number of copy instructions
- Operator Specification:
  <table>
  <tr><td rowspan="1" align="center">Operator Type(OpType)</td><td colspan="3" align="center">AIC Operator</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">Operator Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">1024</td><td align="center">float</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Operator Output</td><td align="center">y</td><td align="center">1024</td><td align="center">float</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">optimize_datacopy_nddma</td></tr>
  </table>
- Operator Implementation:
  When performing non-aligned data copy, use nddma copy to reduce the number of copy instructions
  
  - Call Implementation
    Use the kernel call operator <<<>>> to call the kernel function.

## Build and Run
Execute the following steps in the sample root directory to build and run the operator.
- Configure Environment Variables
  Please select the corresponding environment variable configuration command based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on the current environment.
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
    
- Sample Execution
  ```bash
  mkdir -p build && cd build;                                               # Create and enter build directory
  cmake ..;make -j;                                                         # Build project
  python3 ../scripts/gen_data.py                                            # Generate test input data
  ./demo                                                                    # Execute the compiled executable program, run the sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output result correctness, confirm algorithm logic is correct
  ```
  The execution result is as follows, indicating precision comparison succeeded.
  ```bash
  test pass!
  ```