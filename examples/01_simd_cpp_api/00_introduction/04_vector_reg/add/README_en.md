# Add Sample Based on RegBase Programming

## Overview

This sample implements Add computation based on Reg programming interfaces, using 4-core parallel mode to process data.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── add
│   ├── scripts
│   │   ├── gen_data.py                // Input data and golden data generation script
│   │   └── verify_result.py           // Golden data comparison file
│   ├── CMakeLists.txt                 // Compilation project file
│   ├── data_utils.h                   // Data read/write functions
│   └── vector_add.asc                 // AscendC sample implementation & invocation sample
```

## Sample Description

- Sample Function:
  This sample implements vector self-addition operation based on RegBase programming paradigm, calling interfaces such as Reg::LoadAlign, Reg::Add, Reg::StoreAlign to complete register-level vector computation.

- Sample Specifications:
  <table>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="3" align="center">AIV Sample</td></tr>
  <tr><td rowspan="2" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">[512, 512]</td><td align="center">float</td></tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">y</td><td align="center">[512, 512]</td><td align="center">float</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">vector_add</td></tr>
  </table>

- Sample Implementation:
  - Kernel Implementation
    - Call DataCopy basic API to move data from GM (Global Memory) to UB (Unified Buffer)
    - Call AddVF function through asc_vf_call to implement vector self-addition computation
    - Call DataCopy basic API to move results from UB (Unified Buffer) to GM (Global Memory)

  - Invocation Implementation
    Use the kernel invocation operator <<<>>> to call the kernel function.

## Compilation and Execution

Execute the following steps in the root directory of this sample to compile and run the sample.

- Configure Environment Variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on the current environment.
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
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;                      # Compile project (default npu mode)
  python3 ../scripts/gen_data.py                                            # Generate test input data
  ./demo                                                                    # Execute compiled program, run sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output results are correct
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note**: Before switching compilation modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then run cmake again.

- Compilation Options Description

| Option | Available Values | Description |
|--------|------------------|-------------|
| `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
| `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result
  The execution result is as follows, indicating that the accuracy comparison passed.
  ```bash
  test pass!
  ```