# fixpipe_l0c2l1 Example

## Overview

This example introduces how to use Fixpipe to move matrix multiplication results from L0C (L0C Buffer) to L1 (L1 Buffer), supporting data type conversion, inline quantization, ReLU, and other functions. These interfaces are used to efficiently transfer matrix multiplication computation results from L0C to L1 Buffer, supporting various data format conversions and preprocessing capabilities.

Note:
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products, Atlas A2 Training Series Products/Atlas A2 Inference Series Products only support Nz output format in the L0C to L1 pathway, and do not support float output data type, must be quantized to other data types.
- Ascend 950PR/Ascend 950DT does not support moving data directly from L1 to GM. Therefore, in this example, the result matrix moved from L0C to L1 will serve as input for the next matrix multiplication, performing another matrix calculation and outputting the result to GM. (Atlas A2/A3 series products support moving data directly from L1 to GM, this example chooses to move out directly)


## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── fixpipe_l0c2l1
│   ├── scripts
│   │   ├── gen_data.py                // Input data and golden data generation script
│   │   └── verify_result.py           // Verification script for checking output data against golden data
│   ├── CMakeLists.txt                 // Build configuration file
│   ├── data_utils.h                   // Data read/write functions
│   └── fixpipe_l0c2l1.asc             // Ascend C example implementation & invocation example
```

## FixpipeParamsV220 vs FixpipeParamsArch3510 Structure Comparison

Different products support different parameter structures:
- **Ascend 950PR/Ascend 950DT**: Supports both `FixpipeParamsV220` and `FixpipeParamsArch3510` parameter structures, recommended to use `FixpipeParamsArch3510`
- **Atlas A3 Training/Inference Series, Atlas A2 Training/Inference Series**: Only supports `FixpipeParamsV220`

This example selects different architectures through the build parameter `CMAKE_ASC_ARCHITECTURES`, automatically selecting the corresponding parameter structure based on architecture:
- `dav-2201` architecture: Uses `FixpipeParamsV220`
- `dav-3510` architecture: Uses `FixpipeParamsArch3510`

<table>
<caption style="font-weight: normal;">
  	     <span style="font-weight: bold; font-size: 1.2em;">Table 1: Parameter Structure Comparison</span>
<tr><td rowspan="1" align="center">Member Name</td><td align="center">FixpipeParamsV220</td><td align="center">FixpipeParamsArch3510</td><td align="center">Description</td></tr>
<tr><td align="center">`nSize`</td><td align="center">✅</td><td align="center">✅</td><td>Size of output matrix in N direction</td></tr>
<tr><td align="center">`mSize`</td><td align="center">✅</td><td align="center">✅</td><td>Size of output matrix in M direction</td></tr>
<tr><td align="center">`srcStride`</td><td align="center">✅</td><td align="center">✅</td><td>Starting address offset of adjacent Z layouts in source Nz matrix</td></tr>
<tr><td align="center">`dstStride`</td><td align="center">✅</td><td align="center">✅</td><td>Starting address offset of adjacent Z layouts in destination matrix (Nz format, note the difference in units between the two structures) or number of elements per row (ND/DN format)</td></tr>
<tr><td align="center">`quantPre`</td><td align="center">✅</td><td align="center">✅</td><td>Quantization mode control</td></tr>
<tr><td align="center">`deqScalar`</td><td align="center">✅</td><td align="center">✅</td><td>Scalar quantization parameter</td></tr>
<tr><td align="center">`reluEn`</td><td align="center">✅</td><td align="center">✅</td><td>ReLU switch</td></tr>
<tr><td align="center">`unitFlag`</td><td align="center">✅</td><td align="center">✅</td><td>Mmad and Fixpipe fine-grained parallel control</td></tr>
<tr><td align="center">`isChannelSplit`</td><td align="center">✅</td><td align="center">✅</td><td>Channel split switch</td></tr>
<tr><td align="center">`ndNum` / `srcNdStride` / `dstNdStride`</td><td align="center">✅</td><td align="center">✅ (in TransformParams)</td><td>Parameters controlling multi-matrix transfer in Nz2ND scenario, independent members in V220, integrated into `TransformParams` structure in Arch3510</td></tr>
<tr><td align="center">`dnNum` / `srcNzMatrixStride` / `dstDnMatrixStride` / `srcNzC0Stride`</td><td align="center">❌</td><td align="center">✅ (in TransformParams)</td><td>Parameters controlling multi-matrix transfer in Nz2DN scenario, only supported by Arch3510</td></tr>
<tr><td align="center">`TransformParams`</td><td align="center">❌</td><td align="center">✅</td><td>Type selector based on template parameters, automatically selects parameter type based on CO2Layout</td></tr>
<tr><td align="center">`dualDstCtrl`</td><td align="center">❌</td><td align="center">✅</td><td>Dual destination mode control, supports M dimension split or N dimension split</td></tr>
<tr><td align="center">`subBlockId`</td><td align="center">❌</td><td align="center">✅</td><td>Indicates target UB number in single destination mode</td></tr>
</table>

## Scenario Detailed Description

This example selects different output scenarios through the build parameter `SCENARIO_NUM`. The meanings corresponding to different values of `SCENARIO_NUM` are shown in the table below.
All scenarios are based on the same matrix multiplication specification: [M, N, K] = [128, 128, 128], kernel function name is `fixpipe_l0c2l1`.

<table>
<caption style="font-weight: normal;">
  	     <span style="font-weight: bold; font-size: 1.2em;">Table 2: Meanings of Different scenarioNum Values</span>
<tr><td rowspan="1" align="center">scenarioNum</td><td align="center">L0C Data Type</td><td align="center">L1 Data Type</td><td align="center">Output Format</td><td align="center">Quantization Enabled</td><td align="center">ReLU Enabled</td></tr>
<tr><td align="center">1</td><td align="center">float</td><td align="center">half</td><td align="center">Nz</td><td align="center">No (cast)</td><td align="center">No</td></tr>
<tr><td align="center">2</td><td align="center">float</td><td align="center">int8_t</td><td align="center">Nz</td><td align="center">Yes (scalar)</td><td align="center">No</td></tr>
<tr><td align="center">3</td><td align="center">float</td><td align="center">int8_t</td><td align="center">Nz</td><td align="center">Yes (vector)</td><td align="center">No</td></tr>
<tr><td align="center">4</td><td align="center">float</td><td align="center">half</td><td align="center">Nz</td><td align="center">No (cast)</td><td align="center">Yes</td></tr>
</table>

**Scenario 1: Output Format Nz, Output to L1 Data Type half**
- Input: A [128, 128] half type, ND format; B [128, 128] half type, ND format
- Output: C [128, 128] half type, Nz format
- Implementation: Use `Fixpipe<outputType, l0cType, AscendC::CFG_Nz>` to move data from L0C to L1, output as Nz format
- Description: L0C data is in Nz format directly output to L1 as Nz format, data maintains original format unchanged

**Scenario 2: Output Format Nz, Output to L1 Data Type int8_t, Enable Scalar Quantization**
- Input: A [128, 128] half type, ND format; B [128, 128] half type, ND format
- Output: C [128, 128] int8_t type, Nz format
- Implementation: Set `fixpipeParams.quantPre = QuantMode_t::QF322B8_PRE`, use Scalar quantization mode
- Description: Quantize float type data to int8_t type, the entire C matrix uses one quantization parameter

**Scenario 3: Output Format Nz, Output to L1 Data Type int8_t, Enable Vector Quantization**
- Input: A [128, 128] half type, ND format; B [128, 128] half type, ND format
- Output: C [128, 128] int8_t type, Nz format
- Implementation: Set `fixpipeParams.quantPre = QuantMode_t::VQF322B8_PRE`, use Vector quantization mode, and pass quantization parameters for each column through quantAlphaTensor
- Description: Quantize float type data to int8_t type, each column of C matrix corresponds to one quantization parameter. The quantization parameters used need to be copied from GM to L1

**Scenario 4: Output Format Nz, Output to L1 Data Type half, Enable ReLU**
- Input: A [128, 128] half type, ND format; B [128, 128] half type, ND format
- Output: C [128, 128] half type, Nz format
- Implementation: Set `fixpipeParams.reluEn = true` to enable ReLU function
- Description: Execute ReLU operation during data movement from L0C to L1, i.e., set negative values to 0

## Build and Run

Execute the following steps in the root directory of this example to build and run the example.
- Configure Environment Variables
  Please select the corresponding command to configure environment variables according to the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on the current environment.
  - Default path, root user installed CANN software package
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, non-root user installed CANN software package
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, installed CANN software package
    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Example Execution
  ```bash
  SCENARIO_NUM=1 ASC_ARCH=dav-2201
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_ASC_ARCHITECTURES=$ASC_ARCH ..;make -j;    # Build project (default dav-2201 NPU mode)
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO_NUM -ascArch=$ASC_ARCH  # Generate test input data
  ./demo                           # Execute the compiled executable program to run the example
  python3 ../scripts/verify_result.py output/output.bin ./output/golden.bin $SCENARIO_NUM $ASC_ARCH # Verify if output result is correct
  ```
    When using NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=sim -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_ASC_ARCHITECTURES=$ASC_ARCH ..;make -j;  # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clean the cmake cache. Execute `rm CMakeCache.txt` in the build directory and then re-run cmake.

- Build Option Description
  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `sim` | Run mode: NPU run, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture, dav-2201 corresponds to Atlas A2/A3 series, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  | `SCENARIO_NUM` | 1-4 | Scenario number |

  The execution result is as follows, indicating precision comparison succeeded.
  ```bash
  test pass!
  ```