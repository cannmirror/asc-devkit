# Custom Operator Project + aclop Single Operator Model Invocation Sample

## Overview

This sample demonstrates how to execute fixed-shape operators using the `aclopExecuteV2` single operator model approach, based on the example custom operator project.

## Supported Products

This sample supports the following product models:
- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products
- Atlas 200I/500 A2 Inference Products
- Atlas Inference Series Products

## Directory Structure

```
├── aclop_invocation
│   ├── add_custom.json
│   ├── CMakeLists.txt
│   └── main.cpp
```

## Code Implementation

After completing the development and deployment of the custom operator, you can verify the single operator functionality through single operator model invocation. For details, refer to the "Single Operator Model Execution" section in [Single Operator Invocation](https://hiascend.com/document/redirect/CannCommunityCppOpcall).

For single operator model invocation in offline mode, you need to generate the single operator offline model in advance using `atc --singleop` and configure the model directory in the application.

This sample executes the single operator model using the `aclopExecuteV2` interface. The core workflow is as follows:
1. Create input/output Tensor descriptions using `aclCreateTensorDesc`, and prepare device-side data buffers using `aclCreateDataBuffer`.
2. After specifying the model directory with `aclopSetModelDir`, use `aclopExecuteV2` to execute the operator computation.
3. Synchronize the stream with `aclrtSynchronizeStream`, and copy the device-side results back to the host side for result verification.

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
  # Generate single operator offline model using atc model conversion tool
  atc --singleop=../add_custom.json --output=. --soc_version=${soc_version}
  cmake .. && make -j
  ./execute_add_op
  ```

  > To obtain the AI processor model <soc_version>:
  > - For the following product models: Run the `npu-smi info` command on the server with the Ascend AI processor installed to query and obtain the **Name** information. The actual configuration value is AscendName. For example, if **Name** is xxxyy, the actual configuration value is Ascendxxxyy.
  >   - Atlas A2 Training Series Products / Atlas A2 Inference Series Products
  >   - Atlas 200I/500 A2 Inference Products
  >   - Atlas Inference Series Products
  >   - Atlas Training Series Products
  >
  > - For the following product models: Run the `npu-smi info -t board -i <id> -c <chip_id>` command on the server with the Ascend AI processor installed to query and obtain the **Chip Name** and **NPU Name** information. The actual configuration value is Chip Name_NPU Name. For example, if **Chip Name** is Ascendxxx and **NPU Name** is 1234, the actual configuration value is Ascendxxx_1234. Where:
  >
  >   id: Device ID. The NPU ID obtained from the `npu-smi info -l` command is the device ID.
  >
  >   chip_id: Chip ID. The Chip ID obtained from the `npu-smi info -m` command is the chip ID.
  >   - Ascend 950PR/Ascend 950DT
  >   - Atlas A3 Training Series Products / Atlas A3 Inference Series Products
  >
  >   Operator projects created based on AI processor models of the same series have common basic functionality (operator development, compilation, and deployment based on the project).

  The following result indicates successful execution:

  ```log
  test pass
  ```