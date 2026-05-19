# Custom Example Project + ONNX Model Invocation Example

## Overview

This example uses LeakyRelu computation to demonstrate how to invoke custom operators through ONNX network calls.

## Supported Products

This example supports the following product models:
- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── leaky_relu_onnx_invocation
│   ├── CMakeLists.txt
│   ├── leaky_relu.py
│   └── main.cpp
```

## Example Implementation

After completing the development and deployment of the custom operator, you can verify the example functionality through ONNX model invocation. After converting the LeakyRelu ONNX model to an OM model, you can load the model in the application and perform inference.

This example uses the `aclmdlExecute` interface to execute model inference. The core process is as follows:
1. Load the OM model via `aclmdlLoadFromFile` and obtain the model ID, then call `aclmdlGetDesc` to get model description information.
2. Call `aclmdlCreateDataset` to create input/output datasets, and allocate Device memory and create DataBuffers based on `aclmdlGetInputSizeByIndex`/`aclmdlGetOutputSizeByIndex`.
3. Construct Host-side input data, copy to Device side via `aclrtMemcpy`, call `aclmdlExecute` to perform inference, and copy results back to Host side for result verification.

## Compilation and Execution

Execute the following steps in the root directory of this example to compile and run the example.
- Compile, package, and deploy the custom example project

  Before running this example, navigate to the [custom example project](../../00_compilation/custom_op/) directory to complete compilation, packaging, and deployment.

- Install ONNX

  ```bash
  pip3 install onnx==1.12.0
  ```

- Configure Environment Variables

  Select the appropriate environment variable configuration command based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on the current environment.
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

- Example Execution

  ```bash
  mkdir -p build; cd build
  python3 ../leaky_relu.py
  # Use the atc model conversion tool to convert *.onnx format model to *.om format model
  atc --model=./leaky_relu.onnx --framework=5 --soc_version=${soc_version} --output=./leaky_relu_custom --input_shape="X:8,16,1024" --input_format=ND
  cmake .. && make -j
  ./execute_leaky_relu_op
  ```

  > Obtain the AI processor model <soc_version> as follows:
  > - For the following product models: Execute the `npu-smi info` command on the server with the Ascend AI processor installed to query and obtain the **Name** information. The actual configuration value is AscendName. For example, if **Name** is xxxyy, the actual configuration value is Ascendxxxyy.
  >   - Atlas A2 Training Series Products / Atlas A2 Inference Series Products
  >   - Atlas 200I/500 A2 Inference Products
  >   - Atlas Inference Series Products
  >   - Atlas Training Series Products
  >
  > - For the following product models, execute the `npu-smi info -t board -i <id> -c <chip_id>` command on the server with the Ascend AI processor installed to query and obtain **Chip Name** and **NPU Name** information. The actual configuration value is Chip Name_NPU Name. For example, if **Chip Name** is Ascendxxx and **NPU Name** is 1234, the actual configuration value is Ascendxxx_1234. Where:
  >
  >   id: Device ID. The NPU ID obtained from the `npu-smi info -l` command is the device ID.
  >
  >   chip_id: Chip ID. The Chip ID obtained from the `npu-smi info -m` command is the chip ID.
  >   - Ascend 950PR/Ascend 950DT
  >   - Atlas A3 Training Series Products / Atlas A3 Inference Series Products
  >
  >   Example projects created based on AI processor models of the same series have common basic functionality (example development, compilation, and deployment based on the project).

  The execution is successful when the following result appears:

  ```log
  test pass
  ```