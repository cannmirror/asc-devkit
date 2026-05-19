# DataCopy Quantization Activation Transfer Example

## Overview

This example implements on-the-fly quantization activation data transfer based on DataCopy, transferring matrix multiplication results from CO1 (L0C Buffer) to GM (Global Memory), with support for combined on-the-fly NZ2ND conversion, quantization, and ReLU activation capabilities. This example provides 6 different test scenarios.

    <table>
   	 	<tr>
   	 		<td>scenarioNum</td>
          <td>Output Format</td>
   	 		<td>Quantization Mode</td>
          <td>Type Conversion</td>
          <td>Activation Mode</td>
   	 	</tr>
   	 	<tr>
   	 		<td>1</td>
          <td>ND</td>
   	 		<td>Scalar Quantization Mode</td>
          <td>s322f16</td>
          <td>ReLU Activation</td>
   	 	</tr>
   	 	<tr>
   	 		<td>2</td>
          <td>NZ</td>
   	 		<td>Vector Quantization Mode</td>
          <td>s322f16</td>
          <td>None</td>
   	 	</tr>
        <tr>
   	 		<td>3</td>
          <td>NZ</td>
   	 		<td>Scalar Quantization Mode</td>
          <td>f322s8</td>
          <td>None</td>
   	 	</tr>
        <tr>
   	 		<td>4</td>
          <td>ND</td>
   	 		<td>Vector Quantization Mode</td>
          <td>f322s8</td>
          <td>ReLU Activation</td>
   	 	</tr>
        <tr>
   	 		<td>5</td>
          <td>ND</td>
   	 		<td>Scalar Quantization Mode</td>
          <td>s322s8</td>
          <td>None</td>
   	 	</tr>
        <tr>
   	 		<td>6</td>
          <td>NZ</td>
   	 		<td>Vector Quantization Mode</td>
          <td>s322s8</td>
          <td>ReLU Activation</td>
   	 	</tr>
   	 </table>

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── data_copy_l0c2gm
│   ├── scripts
│   │   ├── gen_data.py         // Input data and ground truth data generation script
│   │   └── verify_result.py    // Verification script for comparing output data with ground truth
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── data_copy_l0c2gm.asc    // Ascend C example implementation & invocation example
```

## Example Description

- Example Functionality
  Transfers matrix multiplication results from CO1 (L0C Buffer) to GM (Global Memory), supporting two on-the-fly quantization modes:
  - Scalar Quantization: Use the SetFixpipePreQuantFlag interface to set scalar quantization parameters.
  - Tensor Quantization: Use the SetFixPipeConfig interface to set tensor quantization parameters.
  Also supports combination with on-the-fly NZ2ND conversion and ReLU activation capabilities. For interface documentation, refer to On-the-fly Quantization Activation Transfer.

- Example Specifications

  <table>
  <caption>Scenario 1: Scalar Quantization + ReLU Activation</caption>
  <tr><td rowspan="4" align="center">Example Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[128, 128]</td><td align="center">int8_t</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">[128, 256]</td><td align="center">int8_t</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">Example Output</td></tr>
  <tr><td align="center">z</td><td align="center">[128, 256]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">KernelDataCopyL0c2Gm</td></tr>
  </table>

  <table>
  <caption>Scenario 2: Vector Quantization</caption>
  <tr><td rowspan="4" align="center">Example Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[128, 128]</td><td align="center">int8_t</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">[128, 256]</td><td align="center">int8_t</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">Example Output</td></tr>
  <tr><td align="center">z</td><td align="center">[128, 256]</td><td align="center">half</td><td align="center">NZ</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">KernelDataCopyL0c2Gm</td></tr>
  </table>

  <table>
  <caption>Scenario 3: Scalar Quantization</caption>
  <tr><td rowspan="4" align="center">Example Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[128, 128]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">[128, 256]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">Example Output</td></tr>
  <tr><td align="center">z</td><td align="center">[128, 256]</td><td align="center">int8_t</td><td align="center">NZ</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">KernelDataCopyL0c2Gm</td></tr>
  </table>

  <table>
  <caption>Scenario 4: Vector Quantization + ReLU Activation</caption>
  <tr><td rowspan="4" align="center">Example Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[128, 128]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">[128, 256]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">Example Output</td></tr>
  <tr><td align="center">z</td><td align="center">[128, 256]</td><td align="center">int8_t</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">KernelDataCopyL0c2Gm</td></tr>
  </table>

  <table>
  <caption>Scenario 5: Scalar Quantization</caption>
  <tr><td rowspan="4" align="center">Example Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[128, 128]</td><td align="center">int8_t</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">[128, 256]</td><td align="center">int8_t</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">Example Output</td></tr>
  <tr><td align="center">z</td><td align="center">[128, 256]</td><td align="center">int8_t</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">KernelDataCopyL0c2Gm</td></tr>
  </table>

  <table>
  <caption>Scenario 6: Vector Quantization + ReLU Activation</caption>
  <tr><td rowspan="4" align="center">Example Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[128, 128]</td><td align="center">int8_t</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">[128, 256]</td><td align="center">int8_t</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">Example Output</td></tr>
  <tr><td align="center">z</td><td align="center">[128, 256]</td><td align="center">int8_t</td><td align="center">NZ</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">KernelDataCopyL0c2Gm</td></tr>
  </table>

- Example Implementation

  - Kernel Implementation

    - Call DataCopy basic API to transfer data from GM (Global Memory) to A1 (L1 Buffer) and B1 (L1 Buffer), with ND to NZ format conversion.

    - Call LoadData interface to transfer data from A1 (L1 Buffer) and B1 (L1 Buffer) to A2 (L0A Buffer) and B2 (L0B Buffer).

    - Call Mmad interface to perform matrix multiplication on input matrix A with shape [128, 128] and input matrix B with shape [128, 256], producing an output result matrix with shape [128, 256].

    - Configure DataCopyCO12DstParams parameters for DataCopy on-the-fly quantization activation transfer, transferring the computed results from Mmad from CO1 (L0C Buffer) to GM (Global Memory).

  - Invocation Implementation
    Use the kernel call operator `<<<>>>` to invoke the kernel function.

## Build and Run
 
Execute the following steps in the root directory of this example to build and run the example.
- Configure Environment Variables
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

- Example Execution

  ```bash
  SCENARIO=1
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DSCENARIO_NUM=$SCENARIO -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO   # Generate test input data
  ./demo                           # Execute the compiled executable to run the example
  python3 ../scripts/verify_result.py -scenarioNum=$SCENARIO ./output/output.bin ./output/golden.bin  # Verify output results
  ```

  When using NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  Example:
  ```bash
  cmake -DSCENARIO_NUM=$SCENARIO -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Build Option Description

  | Option | Available Values | Description |
  | ----------------| -----------------------------| --------------------------------------------------------------------------------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `sim` | Run mode: NPU execution, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  | `SCENARIO_NUM` | `1`, `2`, `3`, `4`, `5`, `6` | Scenario number: 1=Scalar Quantization+ReLU Activation+ND Output, 2=Vector Quantization+NZ Output, 3=Scalar Quantization+NZ Output, 4=Vector Quantization+ReLU Activation+ND Output, 5=Scalar Quantization+ND Output, 6=Vector Quantization+ReLU Activation+NZ Output |

- Execution Result
  When the execution result shows the following, it indicates successful precision comparison:
  ```bash
  test pass!
  ```