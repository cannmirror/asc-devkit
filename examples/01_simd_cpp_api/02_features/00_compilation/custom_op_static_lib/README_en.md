# Custom Operator Static Library

## Overview

This sample uses `AddCustom` as an example to demonstrate how to build, package, and link a custom operator static library, and execute the operator through aclnn.

## Supported Products

This sample supports the following product models:

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── custom_op_static_lib
│   ├── app                       // Link static library and execute operator
│   │   ├── CMakeLists.txt
│   │   └── main.cpp
│   ├── op                        // Build and package to generate custom operator static library
│   │   ├── add_custom_host.cpp
│   │   ├── add_custom_kernel.cpp
│   │   ├── add_custom_tiling.h
│   │   └── CMakeLists.txt
│   └── CMakeLists.txt
```

## Code Implementation Description

In this sample, the `op` directory is responsible for building and packaging the custom operator static library. The generated static library path is `./build/customize-install/lib/lib${package_name}.a`. The `app` directory imports the static library through `find_package(${package_name})`, compiles `main.cpp` to generate `execute_add_op`, calls `aclnnAddCustom`, and verifies the result.

For AddCustom operator introduction and other content, refer to [Operator Description](../custom_op/README.md).

## Build and Run

- Configure environment variables

  Select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on the current environment. Ensure `ASCEND_HOME_PATH` points to the CANN installation root path for header files and library paths to take effect.

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

  Execute the following steps in the sample root directory to run this sample.

  ```bash
  mkdir -p build; cd build
  cmake .. && make -j
  ./execute_add_op
  ```

  The following output indicates successful execution:

  ```log
  test pass
  ```