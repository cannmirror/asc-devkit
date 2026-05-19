# CPU Debug Direct Invocation Sample

## Overview

This sample demonstrates CPU Debug testing for the Add operator implemented in Ascend C programming language.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products

## Directory Structure

```
├── 03_cpudebug
│   ├── CMakeLists.txt          // Build project file
│   └── add.asc                 // Ascend C operator implementation & invocation sample
```

## Sample Description

- CPU Debug Introduction:
  The CPU Debug feature supports debugging the runtime state during CPU execution, primarily through the GDB tool. GDB debugging supports common debugging operations such as setting breakpoints, viewing register and memory states, single-stepping, and viewing call stacks.

- Operator Introduction:
  For detailed description of the Add operator functionality, refer to the [Add Operator Details](../../00_introduction/01_vector/basic_api_tque_add/README.md) section.

## Build and Run

Execute the following steps in the sample root directory to build and run the sample.
- Configure Environment Variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package in your current environment.
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
  mkdir -p build && cd build;
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;
  ./add
  ```
  Select the appropriate `CMAKE_ASC_ARCHITECTURES` parameter based on the actual NPU hardware architecture being tested.
  - Build Options Description
    | Option | Description |
    |--------|-------------|
    | `CMAKE_ASC_RUN_MODE` | Specify as `cpu` to enable CPU domain compilation |
    | `CMAKE_ASC_ARCHITECTURES` | Specify the NPU architecture version. CMake will configure the corresponding CPU debugging dependency libraries based on this value. `dav-2201` corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products; `dav-3510` corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result
  The execution result is shown below, indicating successful execution.
  ```bash
  [Success] Case accuracy is verification passed.
  ```
- Enter GDB Debugging Mode
  The generated CPU domain executable supports debugging through GDB. GDB supports common debugging operations such as setting breakpoints, viewing register and memory states, single-stepping, and viewing call stacks. Add `gdb --args` before `./add` in the above command to enter GDB mode.
  ```bash
  gdb --args ./add
  ```
  CPU Debug simulates NPU execution logic by launching a separate subprocess for each kernel function. Therefore, when using GDB for debugging, you need to set `follow-fork-mode` to let GDB follow the child process in order to set breakpoints inside the kernel function. After entering GDB, first set the child process following mode:

  ```text
  (gdb) set follow-fork-mode child
  ```

  Then debug as needed. Common operations:

  ```text
  # Set breakpoint at kernel function entry
  (gdb) break Compute

  # Run program
  (gdb) run

  # Single step execution
  (gdb) next

  # Print variable value
  (gdb) print xLocal.GetValue(0)

  # Continue execution to next breakpoint
  (gdb) continue
  ```