# Custom Operator Project + TensorFlow Custom Operator Example

## Overview

This example demonstrates how to map an Ascend C custom Add operator to a TensorFlow custom operator and invoke the Ascend C operator through TensorFlow.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── tensorflow_custom
│   ├── CMakeLists.txt
│   ├── custom_assign_add_custom.cc
│   └── run_add_custom_tf.py
```

## Code Implementation

After creating the operator project, a `framework/tf_plugin` directory is generated under the operator project directory to store TensorFlow framework adapter plugin implementation files.

This example uses the custom Add operator to map it to a TensorFlow custom operator. The core process is as follows:
1. Load the custom operator library file via `tf.load_op_library` to obtain the custom operator interface `add_custom`.
2. Construct input data, use `tf.compat.v1.placeholder` to define input tensors, and compute results for both `tf.math.add` and `add_custom`.
3. Configure `ConfigProto`, enable `NpuOptimizer`, and disable remapping and memory optimization to ensure the operator executes as expected.
4. Call `np.allclose` to compare the results of the standard TensorFlow addition operator and the custom operator to verify computational correctness.

## Compilation and Execution

Execute the following steps in the root directory of this example to compile and run the operator.
- Compile, package, and deploy the custom operator project

  Before running this example, navigate to the [custom operator project example](../../00_compilation/custom_op/) directory to complete compilation, packaging, and deployment.

  > [!NOTE]
  > The adapter plugin code needs to be modified. The path is: `examples/01_simd_cpp_api/02_features/00_compilation/custom_op/framework/tf_plugin/tensorflow_add_custom_plugin.cc`. Modify the TensorFlow operator name OriginOpType to "AddCustom" in the plugin code as shown below:
  >
  > ```cc
  > REGISTER_CUSTOM_OP("AddCustom")
  >   .FrameworkType(TENSORFLOW)      // type: TENSORFLOW
  >   .OriginOpType("AddCustom")      // name in tf module
  >   .ParseParamsByOperatorFn(AutoMappingByOpFn);
  > ```

- Install TensorFlow Plugin Package

  Refer to the "Installing Framework Plugin Package" section in [TensorFlow 2.6.5 Model Migration](https://www.hiascend.com/document/redirect/canncommercial-tfmigr26) for detailed installation instructions and steps.

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
  cmake .. && make -j
  python3 ../run_add_custom_tf.py
  ```

  The execution is successful when the following result appears:

  ```log
  test pass
  ```