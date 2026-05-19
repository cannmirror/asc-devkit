# Profiling Example Based on Gather Operator

## Overview

This example demonstrates how to collect on-board performance data using msProf based on the Gather operator. Users can quickly identify software and hardware performance bottlenecks of the operator based on the output performance data, improving operator performance analysis efficiency.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Supported CANN Software Versions
- \>= CANN 9.0.0

## Directory Structure Introduction

```
├── msProf
│   ├── CMakeLists.txt         # cmake build file
│   ├── gather.asc             # SIMT implementation gather example
|   └── README.md
```

## Operator Description
  The gather operator implements the function of retrieving 12288 rows of data at specified indices from a two-dimensional vector with shape 100000 * 128. For detailed function description, please refer to the [Gather Operator Details](../../../02_features/00_resource_management/basic_gather/README.md) chapter.

## msProf Tool Introduction
msProf is a single-operator performance analysis tool. It includes two usage modes: msprof op and msprof op simulator. This tool helps users identify operator memory, operator code, and operator instruction anomalies, achieving comprehensive operator tuning. Currently, it supports performance data collection and automatic parsing based on different run modes (on-board or simulation) and different file formats (executable files or operator binary .o files).

- On-board performance collection

    Through on-board performance collection, you can directly measure the operator execution time on Ascend AI processors. This method is suitable for quickly identifying operator performance issues in on-board environments.

    Execute operator tuning through msprof op based on executable file demo:
    ```
    msprof op ./demo
    ```

    - Performance data description


      After the command completes, a folder named "OPPROF_{timestamp}_XXX" is generated in the default directory. The performance data folder structure is as follows:

      ```bash
      ├──dump                       # Raw performance data, users do not need to focus on this
      ├──ArithmeticUtilization.csv  # cube/vector instruction cycle ratio
      ├──L2Cache.csv                # L2 Cache hit rate
      ├──Memory.csv                 # UB, L1 and main memory read/write bandwidth rate
      ├──MemoryL0.csv               # L0A, L0B, and L0C read/write bandwidth rate
      ├──MemoryUB.csv               # Vector and Scalar to UB read/write bandwidth rate
      ├──OpBasicInfo.csv            # Operator basic information
      ├──PipeUtilization.csv        # Collect compute unit and transfer unit time and ratio
      ├──ResourceConflictRatio.csv  # UB bank group, bank conflict and resource conflict ratio among all instructions
      └──visualize_data.bin         # MindStudio Insight presentation file
      ```

Users can use MindStudio Insight to open the `visualize_data.bin` file to visually view operator information, including operator basic information, inter-core load analysis, compute workload analysis, memory load analysis, and so on. For more msProf tool usage, please refer to the "Operator Tuning msOpProf" section in [Operator Development Tools](https://www.hiascend.com/document/redirect/CannCommercialToolOpDev).


## Build and Run

Execute the following steps in the root directory of this example to build and run the operator.
- Configure environment variables

  Please select the appropriate environment variable configuration command based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on your current environment.
  - Default path, root user installation
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, non-root user installation
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, custom installation
    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Execute the example
  ```bash
  mkdir -p build && cd build;           # Create and enter build directory
  cmake ..;make -j;                     # Build the project
  msprof op ./demo                      # Execute operator tuning through msprof op based on executable file demo
  ```
  The following execution result indicates that accuracy verification passed.
  ```
  [Success] Case accuracy is verification passed.
  ```