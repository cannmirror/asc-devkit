# AI CPU Operator Tiling Sink Sample Introduction

## Overview

This sample introduces how to use AI CPU operators for tiling sink computation.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── 02_aicpu_device_tiling
│   ├── CMakeLists.txt                     // Build project file
│   ├── aicore_kernel.asc                  // AI Core operator implementation
│   ├── kernel_args.h                      // Tiling structure header file
│   ├── main.asc                           // AI CPU operator and AI Core operator invocation
│   └── aicpu_tiling.aicpu                 // AI CPU operator implementation
```

## Sample Description

- In main.asc, both AI CPU operators and AI Core operators are invoked using the kernel call operator `<<<...>>>`. The AI CPU operator passes the tiling computation results to the AI Core operator.
- The AI CPU operator and AI Core operator are launched on different streams. In the sample, they are aicpu_stream and aicore_stream respectively. Events are used to record tasks dispatched on streams. Use aclrtRecordEvent to record an event in a specified stream, and use aclrtStreamWaitEvent to block the specified stream until the specified event completes.

## Build and Run

Execute the following steps in the root directory of this sample.
- Configure environment variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on your current environment.
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
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project
  ./demo                           # Execute the compiled executable program to run the sample
  ```

- Build options description

  | Option | Available Values | Description |
  | ----------------| -----------------------------| --------------------------------------------------------------------------------------|
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution result

  The following execution result indicates successful execution:
  ```bash
  MyAicpuKernel inited
  MyAicpuKernel inited type 1 mode 2 len 4 end!
  Hello World: int mode 2 len 4 m 10.
  Hello World: int mode 2 len 4 m 10.
  Hello World: int mode 2 len 4 m 10.
  ```