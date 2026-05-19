# SIMT Kernel Logging Example

## Overview

This example demonstrates the complete usage of Ascend logging functionality in SIMT programming mode, including log output to screen, log file storage, and log level control. By configuring environment variables, developers can flexibly control log output behavior to assist with operator development and issue diagnosis.

For more details about logging functionality, please refer to: [Ascend Logging Reference](https://hiascend.com/document/redirect/CannCommunitylogref)

## Supported Products

- Ascend 950PR/Ascend 950DT

## Supported CANN Software Versions

- \>= CANN 9.1.0

## Directory Structure Introduction

```
├── 09_log
│   ├── CMakeLists.txt         // cmake build file
│   ├── log.asc                // SIMT Kernel example code
|   └── README.md
```

## Example Description  

- Logging functionality explanation:
  - **Log output to screen**: Control whether logs output to standard output (screen) via environment variables for real-time viewing
  - **Log file storage**: Specify log file storage path via environment variables for subsequent analysis
  - **Log level**: Control log output level via environment variables to filter log content as needed

  Log level description:

  | Value | Level | Description |
  |----|------|------|
  | 0 | DEBUG | Output all logs (DEBUG/INFO/WARNING/ERROR), most detailed information |
  | 1 | INFO | Output INFO and above level logs (INFO/WARNING/ERROR) |
  | 2 | WARNING | Output WARNING and above level logs (WARNING/ERROR) |
  | 3 | ERROR | Output only ERROR level logs |
  | 4 | NULL | No log output |

- Example implementation:  
  This example demonstrates the complete flow of Kernel-side exception throwing and Host-side error capture by constructing an abnormal scenario:

  - Kernel implementation  
    Use standard C `assert` assertion in the Kernel function. When the condition is not met, a device-side exception triggers. To avoid repeated printing by all threads, typically only `thread 0` triggers the assertion.

  - Host implementation  
    After Kernel execution, the Host side captures and prints error information through the `ASCENDC_CHECK` macro. `ASCENDC_CHECK` is a tool macro for capturing and printing ACL errors, defined as follows:

    ```c
    #define ASCENDC_CHECK(expr) do { \
        aclError ret = (expr); \
        if (ret != ACL_SUCCESS) { \
            fprintf(stderr, "Ascend Error: %s:%d code=%d %s\n", \
                __FILE__, __LINE__, ret, aclGetRecentErrMsg()); \
        } \
    } while(0)
    ```

    Pass ACL interface calls as parameters, and the macro automatically checks return values and prints error information on failure.

    ```c
    // Capture Kernel execution errors (including errors triggered by assert)
    ASCENDC_CHECK(aclrtGetLastError(ACL_RT_THREAD_LEVEL));
    ASCENDC_CHECK(aclrtSynchronizeDevice());
    ```

## Build and Run

Execute the following steps in the root directory of this example to build and run the operator.

- Configure environment variables  
  Please select the appropriate environment variable configuration command based on the [installation method](../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on your current environment.
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

- Configure log environment variables (NPU mode only)  
  This example supports controlling log output behavior via environment variables (the following environment variables only take effect in NPU mode):
  ```bash
  export ASCEND_PROCESS_LOG_PATH=./log        // Log file storage path
  export ASCEND_SLOG_PRINT_TO_STDOUT=1        // Control whether logs print to screen (1: enable, 0: disable)
  export ASCEND_GLOBAL_LOG_LEVEL=1            // Control log level
  ```

  > Note:
  > - Log environment variables control log output from Ascend internal libraries (RUNTIME/ASCENDCL and so on). `printf` output inside the Kernel is not affected by these environment variables.
  > - Log output to screen and file storage are mutually exclusive. When screen output is enabled (`ASCEND_SLOG_PRINT_TO_STDOUT=1`), logs will not be stored even if `ASCEND_PROCESS_LOG_PATH` is configured.
  >   - In screen output mode, you can save logs through shell redirection: `./demo > err.log 2>&1`
  >   - In file storage mode, disable screen output and configure `ASCEND_PROCESS_LOG_PATH`, logs will be written to the specified directory

- Execute the example
  ```bash
  mkdir -p build && cd build;
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;                      # Build project (default npu mode)
  ./demo
  ```

  For NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  For example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean cmake cache. Execute `rm CMakeCache.txt` in the build directory and re-run cmake.

- Build options description
  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `sim` | Run mode: NPU run, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU architecture: This example only supports dav-3510 (Ascend 950PR/Ascend 950DT) |

- Execution result (for example, in non-screen output mode, file storage details can be found in [Ascend Logging Reference](https://hiascend.com/document/redirect/CannCommunitylogref))  
  After execution, similar information will be printed:
  ```bash
  [INFO] Input shape: 12288
  [INFO] Launching kernel with assert(total_length < 100)...
  Ascend Error: <your_path>/09_log/log.asc:88 code=507035 EZ9999: Inner Error!
  ...
  rtDeviceSynchronize execution failed, reason=vector core exception
  ...

  [ASSERT] <your_path>/09_log/log.asc:47: void add_custom(float *, float *, float *, uint64_t): Assertion `total_length < 100 && "Total length exceeds expected limit!"' failed.
  [INFO] Execution completed. Check for error messages above.
  ```

  > **Explanation:**
  > - `<your_path>` represents the absolute path of the example code directory, replaced with the actual path in real output
  > - Key error information interpretation:
  >   - `Ascend Error: ... code=507035 EZ9999: Inner Error!` —底层错误码 returned by ACL interface
  >   - `vector core exception` — AI Core execution exception, Kernel task failed
  >   - `Assertion 'total_length < 100 ...' failed` — Specific location and condition of assertion failure, helping to locate problematic code

  Log files will also be generated in the `./log` directory, containing detailed runtime logs.