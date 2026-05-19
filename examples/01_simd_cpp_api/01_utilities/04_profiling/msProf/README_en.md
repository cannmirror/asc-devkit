# Profiling Sample Based on MatmulLeakyRelu Operator

## Overview

This sample is based on the MatmulLeakyRelu operator and demonstrates how to collect on-board performance data using msProf. Users can quickly identify software and hardware performance bottlenecks of operators based on the output performance data, improving operator performance analysis efficiency.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── matmul_leakyrelu
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   ├── matmul_leakyrelu.asc    // Ascend C operator implementation & invocation sample
│   └── scripts
│       ├── gen_data.py         // Script to generate input data and golden data
│       └── verify_result.py    // Golden data comparison file
```

## Operator Description
  The MatmulLeakyRelu operator implements matrix multiplication addition (Matmul) combined with LeakyRelu activation function computation. For detailed functionality description, refer to the [MatmulLeakyRelu Operator Details](../../../00_introduction/03_fusion_operation/matmul_leakyrelu/README.md) section.

## msProf Tool Introduction
msProf is a single-operator performance analysis tool. It includes two usage modes: msprof op and msprof op simulator. This tool helps users identify operator memory, operator code, and operator instruction anomalies, enabling comprehensive operator optimization. Currently, it supports performance data collection and automatic parsing based on different run modes (on-board or simulation) and different file formats (executable files or operator binary .o files).

- On-Board Performance Collection

    Through on-board performance collection, the operator's runtime on the Ascend AI processor can be directly measured. This method is suitable for quickly identifying operator performance issues in a board environment.

    Execute operator optimization using msprof op based on executable file demo:
    ```
    msprof op ./demo
    ```

    - Performance Data Description
      After the command completes, a folder named "OPPROF_{timestamp}_XXX" will be generated in the default directory. The performance data folder structure is shown below:

      ```bash
      ├──dump                       # Raw performance data, users do not need to focus on this
      ├──ArithmeticUtilization.csv  # cube/vector instruction cycle ratio
      ├──L2Cache.csv                # L2 Cache hit rate, affects MTE2, recommend planning data transfer logic reasonably to increase hit rate
      ├──Memory.csv                 # UB, L1 and main memory read/write bandwidth rate
      ├──MemoryL0.csv               # L0A, L0B, and L0C read/write bandwidth rate
      ├──MemoryUB.csv               # Vector and Scalar to UB read/write bandwidth rate
      ├──OpBasicInfo.csv            # Operator basic information
      ├──PipeUtilization.csv        # Collection of compute unit and transfer unit time and ratio
      ├──ResourceConflictRatio.csv  # UB bank group, bank conflict and resource conflict ratio among all instructions
      └──visualize_data.bin         # MindStudio Insight presentation file
      ```

For more information on using the msProf tool, please refer to the Operator Optimization (msProf) section in [MindStudio Tools](https://www.hiascend.com/document/redirect/CannCommercialToolOpDev).


## Build and Run

Execute the following steps in the sample root directory to build and run the operator.
- Configure Environment Variables

  Select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package in your current environment.
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
  mkdir -p build && cd build;           # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;                     # Build project
  python3 ../scripts/gen_data.py        # Generate test input data
  msprof op ./demo                      # Execute operator optimization using msprof op based on executable file demo
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output results
  ```

- Build Options Description

| Option | Available Values | Description |
|--------|------------------|-------------|
| `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU Architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products; dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result
  The execution result is shown below, indicating successful accuracy comparison.
  ```bash
  test pass!
  ```