# ElementWiseCompoundCompute Example

## Overview

This sample demonstrates the usage of compound compute interfaces. Compound compute interfaces fuse multiple compute operations into a single instruction, effectively reducing instruction count, intermediate storage overhead, and improving computational efficiency compared to calling multiple basic interfaces separately. Refer to AddRelu/Axpy for interface documentation.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── element_wise_compound_compute
│   ├── scripts
│   │   ├── gen_data.py         // Script to generate input data and golden data
│   │   └── verify_result.py    // Script to verify output data against golden data
│   ├── CMakeLists.txt          // Build configuration file
│   ├── data_utils.h            // Data read/write functions
│   └── element_wise_compound_compute.asc    // Ascend C sample implementation & invocation example
```

## Sample Description

- Sample Specifications:
  <table border="2">
  <caption>Table 1: Sample Specifications</caption>
  <tr>
    <th align="left">Scenario Number (SCENARIO_NUM)</th>
    <th align="left">Interface Name</th>
    <th align="left">Description</th>
    <th align="left">Formula</th>
    <th align="left">Input Type</th>
    <th align="left">Output Type</th>
  </tr>
  <tr>
    <td align="left">1</td>
    <td align="left">AddRelu</td>
    <td align="left">Vector addition with ReLU activation fusion</td>
    <td align="left">dst = max(src0 + src1, 0)</td>
    <td align="left">half</td>
    <td align="left">half</td>
  </tr>
  <tr>
    <td align="left">2</td>
    <td align="left">Axpy</td>
    <td align="left">Scalar multiplication with vector addition fusion</td>
    <td align="left">dst = dst + src * scalar</td>
    <td align="left">half</td>
    <td align="left">half</td>
  </tr>
  </table>

  Input and output shapes are both [1, 512], format is ND, kernel function name is `element_wise_compound_compute_custom`.

- Sample Implementation:
  - Kernel Implementation
    - Call DataCopy basic API to transfer data from GM (Global Memory) to UB (Unified Buffer)
    - Call different compound compute interfaces based on scenario: Scenario 1 calls AddRelu for vector addition with ReLU activation fusion, Scenario 2 calls Axpy for scalar multiplication with vector addition fusion
    - Call DataCopy basic API to transfer results from UB (Unified Buffer) to GM (Global Memory)

- Invocation Implementation  
  Use the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the sample root directory to build and run the sample.

- Configure Environment Variables  
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development toolkit on your current environment.
  - Default path, root user installed CANN package
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, non-root user installed CANN package
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Custom path install_path, installed CANN package
    ```bash
    source ${install_path}/cann/set_env.sh
    ```
    
- Sample Execution
  ```bash
  SCENARIO_NUM=1
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO_NUM  # Generate test input data
  ./demo                        # Execute the compiled executable to run the sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin  # Verify output results
  ```
  The following output indicates successful accuracy comparison.
  ```bash
  test pass!
  ```

  For CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clear the cmake cache. Execute `rm CMakeCache.txt` in the build directory and run cmake again.

- Build Options
  | Option | Available Values | Description |
  |--------|------------------|-------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 for Atlas A2 Training Series/Atlas A2 Inference Series and Atlas A3 Training Series/Atlas A3 Inference Series, dav-3510 for Ascend 950PR/Ascend 950DT |
  | `SCENARIO_NUM` | `1` (default), `2` | Scenario number: 1 (AddRelu), 2 (Axpy) |