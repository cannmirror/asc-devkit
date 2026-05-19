# torch.library Custom Operator Direct Invocation Sample

## Overview

This sample demonstrates how to register custom operators using PyTorch's torch.library mechanism, based on the Add operator.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── torch_library
│   ├── CMakeLists.txt          // Build project file
│   ├── add_custom_test.py      // PyTorch invocation script
│   └── add_custom.asc          // Ascend C sample implementation & torch.library registration
```

## Sample Description

- Sample functionality:

  The Add computation formula is:

  ```
  z = x + y
  ```

- Sample specifications:
  <table border="2" align="center">
  <caption>Table 1: AddCustom Sample Specification Description</caption>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="4" align="center">AddCustom</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[8, 2048]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">[8, 2048]</td><td align="center">half</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">z</td><td align="center">[8, 2048]</td><td align="center">half</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">add_custom</td></tr>
  </table>

- Custom operator registration:

  This sample defines a namespace named `ascendc_ops` in `add_custom.asc` and registers the `ascendc_add` function within it.

  PyTorch provides the `TORCH_LIBRARY` macro as the core interface for custom sample registration, used to create and initialize custom operator libraries. After registration, it can be invoked on the Python side using `torch.ops.namespace.op_name`. For example:

  ```c++
  TORCH_LIBRARY(ascendc_ops, m) {
      m.def(ascendc_add"(Tensor x, Tensor y) -> Tensor");
  }
  ```

  `TORCH_LIBRARY_IMPL` is used to bind operators to specific `DispatchKey` (PyTorch device dispatch identifier). For NPU devices, the operator implementation needs to be registered to `PrivateUse1`, which is the dedicated `DispatchKey`. For example:

  ```c++
  TORCH_LIBRARY_IMPL(ascendc_ops, PrivateUse1, m)
  {
      m.impl("ascendc_add", TORCH_FN(ascendc_ops::ascendc_add));
  }
  ```

  In the `ascendc_add` function, the current NPU stream is obtained via `c10_npu::getCurrentNPUStream()`, and the custom kernel function `add_custom` is invoked using the kernel launch operator `<<<>>>` to execute the operator on the NPU.

- Python test script

  In the `add_custom_test.py` invocation script, load the generated custom operator library via `torch.ops.load_library`, call the registered `ascendc_add` function, and verify the numerical correctness of the custom operator by comparing the NPU output with the CPU standard addition result.

## Build and Run

- Install PyTorch and Ascend Extension for PyTorch plugin

  Refer to the installation instructions from the [pytorch: Ascend Extension for PyTorch](https://gitcode.com/Ascend/pytorch) open source repository or [Ascend Extension for PyTorch Ascend Community](https://hiascend.com/document/redirect/Pytorch-index), select a supported `Python` version matching release, and complete the installation of `torch` and `torch-npu`.

- Install prerequisite dependencies

  ```bash
  pip3 install expecttest
  ```

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
  Execute the following steps in the root directory of this sample to run it.

  ```bash
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project
  python3 ../add_custom_test.py    # Execute sample
  ```

- Build option description

  | Option | Available Values | Description |
  | ----------------| -----------------------------| --------------------------------------------------------------------------------------|
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU Architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution result

  The following result indicates successful accuracy comparison:

  ```bash
  Ran 1 test in **s
  OK
  ```