# HelloWorld Example

## Overview

This example is a SIMT programming introductory example, demonstrating the basic flow of running and verifying the example kernel function on the NPU side by using the <<<>>> kernel launch operator, with printf printing output results inside the kernel function.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Supported CANN Software Versions
- \> CANN 9.0.0

## Directory Structure

```
├── hello_world_simt
│   ├── CMakeLists.txt      // Build project file
│   └── hello_world.asc     // Ascend C SIMT programming example implementation & invocation example
```

## Build and Run

Execute the following steps in the root directory of this example to build and run the example.
- Configure environment variables
  Select the corresponding command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development toolkit on the current environment.
  - Default path, CANN software package installed by root user
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, CANN software package installed by non-root user
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, CANN software package installation
    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Example execution
  ```bash
  mkdir -p build && cd build;   # Create and enter build directory
  cmake ..;make -j;             # Build the project
  ./demo                        # Run the example
  ```
  The following output after execution indicates successful execution.
  ```bash
  [blockIdx (0/2)][threadIdx (2/32)]: Hello World!
  [blockIdx (0/2)][threadIdx (1/32)]: Hello World!
  [blockIdx (0/2)][threadIdx (0/32)]: Hello World!
  [blockIdx (1/2)][threadIdx (2/32)]: Hello World!
  [blockIdx (1/2)][threadIdx (1/32)]: Hello World!
  [blockIdx (1/2)][threadIdx (0/32)]: Hello World!
  ```