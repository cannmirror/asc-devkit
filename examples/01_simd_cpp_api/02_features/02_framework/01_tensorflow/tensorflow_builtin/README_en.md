# Custom Sample Project + TensorFlow Built-in Operator Sample

## Overview

This sample demonstrates how to map Ascend C custom operators to TensorFlow built-in operators and invoke them through TensorFlow, based on the Add computation.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── tensorflow_builtin
│   └── run_add_custom.py
```

## Code Implementation

After creating the sample project, a `framework/tf_plugin` directory is generated under the sample project directory to store TensorFlow framework adaptation plugin implementation files.

This sample demonstrates mapping the Add computation to the TensorFlow built-in operator Add. The core workflow is as follows:
1. Use `np.random.uniform` to generate random input data, and define input nodes using `tf.compat.v1.placeholder`.
2. Build the computation graph: Use `tf.math.add` to implement tensor addition operations.
3. Create CPU and NPU sessions separately, execute the computation graph and pass in input data via `session.run`.
4. Use `np.allclose` to compare NPU and CPU computation results to verify computation correctness.

## Build and Run

Execute the following steps in the root directory of this sample to compile and run it.

- Compile, package, and deploy the custom sample project

  Before running this sample, navigate to the [Custom Sample Project Sample](../../00_compilation/custom_op/) directory to complete compilation, packaging, and deployment.

  > [!NOTE] Note
  > You need to adapt the plugin code at path: `examples/01_simd_cpp_api/02_features/00_compilation/custom_op/framework/tf_plugin/tensorflow_add_custom_plugin.cc`. Modify the TensorFlow invocation sample name OriginOpType in the plugin code to "AddV2" as shown below:
  >
  > ```cc
  > REGISTER_CUSTOM_OP("AddCustom")
  >   .FrameworkType(TENSORFLOW)      // type: TENSORFLOW
  >   .OriginOpType("AddV2")      // name in tf module
  >   .ParseParamsByOperatorFn(AutoMappingByOpFn);
  > ```

- Install TensorFlow plugin package

  Refer to the "Install Framework Plugin Package" section in [TensorFlow 2.6.5 Model Migration](https://www.hiascend.com/document/redirect/canncommercial-tfmigr26) for detailed installation guides and steps.

- Configure environment variables

  Select the appropriate command to configure environment variables based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on your current environment.
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

  ```bash
  python3 run_add_custom.py
  ```

  The following result indicates successful execution:

  ```log
  test pass
  ```