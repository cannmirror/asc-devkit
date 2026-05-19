# SIMT Assert Assertion Implementation Example Based on Gather Operator

## Overview

This example demonstrates how to use the `assert()` interface in SIMT programming to implement on-board function debugging.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Supported CANN Software Versions
- \> CANN 9.0.0

## Directory Structure

```
├── 01_assert
│   ├── CMakeLists.txt         # cmake build file
│   ├── assert.asc             # Ascend C operator implementation with assert assertion example
|   └── README.md
```

## Operator Description

- Operator function:

  This example demonstrates the practical usage of the `assert()` interface in SIMT implementation functions, enabling assertion debugging during operator execution.


- Operator implementation:
  ```cpp
  __global__ void  simt_assert(float* input, uint32_t in_shape)
  {
      // Calculate global thread ID
      int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (threadIdx.x < 1) {
          printf("[SIMT] %s\n", "trap check start 1!");
          printf("[SIMT] %s\n", "trap check start 2!");
          printf("[SIMT] %s\n", "trap check start 3!");
          assert(in_shape < 1);
          printf("[SIMT] %s\n", "trap check 1!");
      } else if(threadIdx.x < 5) {
          printf("[SIMT] %s\n", "trap check 2!");
          assert(in_shape > 1);
          printf("[SIMT] %s\n", "trap check 3!");
      }
  }
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
    
- Execute the example
  ```bash
  mkdir -p build && cd build;   # Create and enter build directory
  cmake ..; make -j;            # Build the project
  ./demo                        # Run the example
  ```
  After execution, the following output indicates that the function works correctly.
  ```
  [SIMT] trap check 2!
  [SIMT] trap check 2!
  [SIMT] trap check 2!
  [SIMT] trap check 2!
  [SIMT] trap check 2!
  [SIMT] trap check 2!
  [SIMT] trap check 2!
  [SIMT] trap check 2!
  [SIMT] trap check start 1!
  [SIMT] trap check start 1!
  [SIMT] trap check start 2!
  [SIMT] trap check start 2!
  [SIMT] trap check start 3!
  [SIMT] trap check start 3!
  [ASSERT] xxx/assert.asc:32: void simt_assert(float *, uint32_t): Assertion `in_shape < 1' failed.
  [ASSERT] xxx/assert.asc:32: void simt_assert(float *, uint32_t): Assertion `in_shape < 1' failed.
  ```