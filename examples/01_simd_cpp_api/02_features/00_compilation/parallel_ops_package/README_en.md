# Multi-Vendor Parallel Build, Package, and Deployment Sample for Custom Operator Projects

## Overview

This sample demonstrates how to use `ExternalProject_Add` in a top-level CMake project to parallelly build two independent custom operator projects:

- `add_custom` (AddCustom)
- `leaky_relu_custom` (LeakyReluCustom)

Where `add_custom` uses a flat directory organization (host/kernel/tiling source code at the same level), while `leaky_relu_custom` maintains a hierarchical directory structure.

Each sub-project will separately complete the compilation and packaging of custom operators, generating independent `custom_opp_*.run` installation packages.

## Supported Products

This sample supports the following product models:

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products
- Atlas 200I/500 A2 Inference Products
- Atlas Inference Series Products

> Note: This sample involves multiple operator examples. Please refer to the actual supported product models for each operator example.

## Directory Structure

```text
parallel_ops_package
├── CMakeLists.txt
├── README.md
├── add_custom
│   ├── CMakeLists.txt
│   ├── add_custom_host.cpp
│   ├── add_custom_kernel.cpp
│   └── add_custom_tiling.h
└── leaky_relu_custom
    ├── CMakeLists.txt
    ├── framework
    │   ├── CMakeLists.txt
    │   └── onnx_plugin
    │       ├── CMakeLists.txt
    │       └── leaky_relu_custom_plugin.cc
    ├── op_host
    │   ├── CMakeLists.txt
    │   └── leaky_relu_custom
    │       └── leaky_relu_custom_host.cpp
    └── op_kernel
        ├── CMakeLists.txt
        └── leaky_relu_custom
            ├── leaky_relu_custom_kernel.cpp
            └── leaky_relu_custom_tiling.h
```

## Sample Description

`parallel_ops_package` uses the same Add/LeakyRelu sample description as `custom_op`. Please refer to:

- [custom_op/README.md "Sample Description" section](../custom_op/README.md#样例描述)

## Sample Specification Description

`parallel_ops_package` uses the same Add/LeakyRelu specification description as `custom_op`. Please refer to:

- [custom_op/README.md "Sample Specification Description" section](../custom_op/README.md#样例规格描述)

## Code Implementation Description

The Add/LeakyRelu code implementation in `parallel_ops_package` can refer to the `custom_op` documentation.

- [custom_op/README.md "Code Implementation Description" section](../custom_op/README.md#代码实现介绍)

## Build and Run

Execute the following steps in the sample root directory to build, package, and deploy the custom sample packages.

- Configure environment variables

  Select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on the current environment.

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

- Build, package, and deploy two operator packages

  ```bash
  cmake -S . -B build
  cmake --build build -j
  # add_custom package
  ./build/add_custom/custom_opp_*.run

  # leaky_relu_custom package
  ./build/leaky_relu_custom/custom_opp_*.run
  ```

  The following output indicates successful execution:

  ```log
  SUCCESS
  ```

## Build Result Description

The top-level project will generate two subdirectories under `build/`:

- `build/add_custom/`: AddCustom intermediate artifacts and installation package
- `build/leaky_relu_custom/`: LeakyReluCustom intermediate artifacts and installation package

These two directories are independent of each other, facilitating parallel development and release in multi-vendor scenarios.