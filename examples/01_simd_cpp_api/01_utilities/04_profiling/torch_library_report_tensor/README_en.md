# torch.library Invocation Integrated Profiling for Shape Information Reporting

## Overview

This sample demonstrates how to report operator Shape information during Profiling performance data collection based on the torch.library invocation method, assisting users in operator performance analysis.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── torch_library
│   ├── CMakeLists.txt           // Build project file
│   ├── add_custom_test.py       // PyTorch invocation script
│   ├── add_custom.asc           // Integrated Profiling for Shape information collection
```

## Sample Description

- Sample Function

  Using Add computation as an example, the computation formula is:

  ```
  z = x + y
  ```

- Sample Specifications

  <table border="2" align="center">
  <caption>Table 1: AddCustom Sample Specifications Description</caption>
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

- Sample Implementation

  This sample defines a namespace named `ascendc_ops` in `add_custom.asc` and registers the `ascendc_add` function within it. It calls `aclprofRangePushEx` and `aclprofRangePop` before and after `<<<>>>` to report Shape information to Profiling.

  Sample information reporting requires constructing an `aclprofEventAttributes` structure containing version, size, type, and tensor information. The `messageType` is fixed to MESSAGE_TYPE_TENSOR_INFO type as 0, and `aclprofTensorInfo` is the tensor information to be reported. The `opNameId` and `opTypeId` fields are obtained by converting the sample name and sample type through the `aclprofStr2Id` interface. Each data packet has a maximum `tensorNum` of 5; if exceeding 5, please split into multiple packets for reporting. `tensors` is the `aclprofTensor` structure, obtained from the sample's `at::Tensor`.

- Python Test Script

  In the `add_custom_test.py` invocation script, load the generated custom sample library through `torch.ops.load_library` and call the registered `ascendc_add` function.

## Build and Run

- Install PyTorch and Ascend Extension for PyTorch Plugin

  Please refer to the installation instructions from the [pytorch: Ascend Extension for PyTorch](https://gitcode.com/Ascend/pytorch) open source repository or [Ascend Extension for PyTorch Ascend Community](https://hiascend.com/document/redirect/Pytorch-index), select the supported `Python` version matching release, and complete the installation of `torch` and `torch-npu`.

- Install Prerequisites

  ```bash
  pip3 install expecttest
  ```

- Configure Environment Variables

  Select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package in your current environment.
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

- Sample Execution

  Execute the following steps in the sample root directory to run this sample.

  ```bash
  mkdir -p build; cd build
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..; make -j
  msprof --application="python3 ../add_custom_test.py" --output="../result"
  ```

- Build Options Description

| Option | Available Values | Description |
|--------|------------------|-------------|
| `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU Architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products; dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result
  The execution result is shown below, indicating successful data reporting, where DIR_NAME and PATH are the output file name and data storage directory respectively.

  ```bash
  [INFO] Query all data in ${DIR_NAME} done.
  [INFO] Profiling finished.
  [INFO] Process profiling data complete. Data is saved in ${PATH}
  ```

- Shape Information Display

  Open `PROF_000001_*/mindstudio_profiler_output/op_summary_*.csv` to view Shape information. In the sample, Shape information is written to the following fields.

  <table>
    <tr>
      <td align="center">...</td>
      <td align="center">Op Name</td>
      <td align="center">Op Type</td>
      <td align="center">Input Shapes</td>
      <td align="center">Input Data Types</td>
      <td align="center">Input Formats</td>
      <td align="center">Output Shapes</td>
      <td align="center">Output Data Types</td>
      <td align="center">Output Formats</td>
      <td align="center">...</td>
    </tr>
    <tr>
      <td align="center">...</td>
      <td align="center">_Z10add_customPhS_S_j</td>
      <td align="center">_Z10add_customPhS_S_j</td>
      <td align="center">"8,2048;8:2048"</td>
      <td align="center">FLOAT16;FLOAT16</td>
      <td align="center">ND;ND</td>
      <td align="center">"8,2048"</td>
      <td align="center">FLOAT16</td>
      <td align="center">ND</td>
      <td align="center">...</td>
    </tr>
  </table>
