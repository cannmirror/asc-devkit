# SIMT Timestamp Implementation Sample Based on Gather Example

## Overview

This sample demonstrates how to use the `clock()` interface in the `__simt_vf__` kernel function to implement timestamp marking based on gather computation. It records timestamps before and after sample execution and calculates execution time. This interface is suitable for SIMD & SIMT hybrid programming scenarios or SIMT programming scenarios.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Supported CANN Software Versions

- \>= CANN 9.0.0-beta.2

## Directory Structure

```
├── 06_clock
│   ├── CMakeLists.txt         # cmake build file
│   ├── gather.asc             # Ascend C sample implementing gather call
|   └── README.md
```

## Sample Description

- Sample function:

  Implementing gather computation based on SIMT.

## Build and Run

Execute the following steps in the sample root directory to build and run the sample.

- Configure environment variables

  Select the appropriate command to configure environment variables based on the [installation method](../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on the current environment.

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
  mkdir -p build && cd build;   # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..; make -j;            # Build project
  ./demo                        # Run sample
  ```

- Build option description

  | Option | Available Values | Description |
  | ------ | ---------------- | ----------- |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution result

  The following output indicates successful timestamp marking and accuracy verification:

  ```
  simt_vf execute cycle : 22289
  ...
  [Success] Case accuracy is verification passed.
  ```