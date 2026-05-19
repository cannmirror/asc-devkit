# Custom Operator Project + aclnn Single Operator API Invocation Sample

## Overview

This sample demonstrates how to execute fixed-shape operators using the aclnn `OpType` single operator API, based on the example custom operator project.

## Supported Products

This sample supports the following product models:
- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products
- Atlas 200I/500 A2 Inference Products
- Atlas Inference Series Products

## Directory Structure

```
├── aclnn_invocation
│   ├── CMakeLists.txt
│   └── main.cpp
```

## Code Implementation

After completing the development and deployment of the custom operator, you can verify the single operator functionality through single operator API invocation. For details, refer to the [Single Operator API Invocation](https://hiascend.com/document/redirect/CannCommunityAscendCInVorkSingleOp) section and the "Single Operator API Execution" section in [Single Operator Invocation](https://hiascend.com/document/redirect/CannCommunityCppOpcall).

Single operator API execution is a C-language API-based operator execution method that does not require a single operator description file for offline model conversion. You can directly call the single operator API interface.

This sample executes operator computation using the two-stage interface with `aclnnAddCustomGetWorkspaceSize` and `aclnnAddCustom`. The core workflow is as follows:
1. Create input/output `aclTensor` and prepare device-side data.
2. Call `aclnnAddCustomGetWorkspaceSize` to obtain the workspace size required for this computation, and allocate corresponding device memory.
3. Call `aclnnAddCustom` to execute the computation, synchronize the stream with `aclrtSynchronizeStream`, and copy the results back to the host side for verification.

## Build and Run

- Compile, package, and deploy the custom operator project

  Before running this sample, navigate to the [Custom Operator Project Sample](../../00_compilation/custom_op/) directory to complete compilation, packaging, and deployment.

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

- Execute the sample

  Execute the following steps in the root directory of this sample to run it.

  ```bash
  mkdir -p build; cd build
  cmake .. && make -j
  ./execute_add_op
  ```

  The following result indicates successful execution:

  ```log
  test pass
  ```