# Mmad Example

## Overview

This example introduces matrix multiplication with ND format input and B4/B8/B16/B32 input data types (specifically using int4_t/int8_t/bfloat16/float as examples). It explains how to implement matrix multiplication computation (C = A x B + Bias) through the Mmad instruction for four input data types.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── mmad
│   ├── img                         // Illustration files in this document
│   ├── scripts
│   │   ├── gen_data.py             // Input data and golden data generation script
│   │   └── verify_result.py        // Verification script for output data and golden data
│   ├── CMakeLists.txt              // Build project file
│   ├── data_utils.h                // Data read/write functions
│   └── mmad.asc                    // Ascend C example implementation & calling example
```

## Example Description

A complete matrix multiplication involves data movement processes including: GM-->L1, L1-->L0A/L0B, L1-->BT (BiasTable Buffer), L0C-->GM. The data layout formats in different storage units are shown in Table 1 below:

<table border="2">
<caption>Table 1: Data Layout Formats in Different Storage Units</caption>
  <tr>
    <td >Storage Unit</td>
    <td>Data Layout Format</td>
  </tr>
  <tr>
    <td>GM</td>
    <td>Input matrices A, B and output matrix C are in ND layout.</td>
  </tr>
  <tr>
    <td>L1</td>
    <td>Matrices A, B are in Nz layout.</td>
  </tr>
  <tr>
    <td>L0A</td>
    <td>For Ascend 950PR/Ascend 950DT products, matrix A is in Nz layout.<br>For Atlas A3 Training Series Products/Atlas A3 Inference Series Products and Atlas A2 Training Series Products/Atlas A2 Inference Series Products, matrix A is in Zz layout.</td>
  </tr>
  <tr>
    <td>L0B</td>
    <td>Matrix B is in Zn layout.</td>
  </tr>
  <tr>
    <td>BT (BiasTable Buffer)</td>
    <td>Bias is a one-dimensional Tensor with shape [N].</td>
  </tr>
  <tr>
    <td>L0C</td>
    <td>Matrix C is in Nz layout.</td>
  </tr>
</table>

The standard matrix multiplication computation formula: C = A × B + Bias, where matrices A, B, Bias, and C need to satisfy shapes of [M,K], [K,N], [N], and [M,N] respectively. The correspondence between Bias data type and C matrix data type is shown in Table 2:

<table border="2">
<caption>Table 2: Data Type Correspondence Between L0C and Input Bias</caption>
  <tr>
    <td>Bias Data Type on GM/L1</td>
    <td>Bias Data Type on BT (BiasTable Buffer)</td>
    <td>Matrix Computation Output Data Type on L0C</td>
  </tr>
  <tr>
    <td>int32_t</td>
    <td>int32_t</td>
    <td>int32_t</td>
  </tr>
  <tr>
    <td>bfloat16</td>
    <td rowspan="3">float</td>
    <td rowspan="3">float</td>
  </tr>
  <tr>
    <td>half</td>
  </tr>
  <tr>
    <td>float</td>
  </tr>
</table>

The scenarios corresponding to different values of the scenarioNum parameter in the program are shown in Table 3:

<table border="2">
<caption>Table 3: Meaning of Different scenarioNum Values</caption>
  <tr>
    <td >scenarioNum</td>
    <td>Input Data Type</td>
    <td>Output Data Type</td>
    <td>Matrix A</td>
    <td>Matrix B</td>
    <td>Bias</td>
  </tr>
  <tr>
    <td>1</td>
    <td>int8_t</td>
    <td>int32_t</td>
    <td>Not transposed</td>
    <td>Not transposed</td>
    <td>With Bias and no biasTensor passed, C matrix initial value comes from C2</td>
  </tr>
  <tr>
    <td>2</td>
    <td>bfloat16</td>
    <td>bfloat16</td>
    <td>Not transposed</td>
    <td>Transposed</td>
    <td>No Bias, C matrix accumulation comes from CO1 initial value</td>
  </tr>
  <tr>
    <td>3</td>
    <td>float</td>
    <td>float</td>
    <td>Transposed</td>
    <td>Transposed</td>
    <td>With Bias and biasTensor passed scenario</td>
  </tr>
  <tr>
    <td>4</td>
    <td>int4b_t</td>
    <td>int32_t</td>
    <td>Not transposed</td>
    <td>Transposed</td>
    <td>Bias not enabled, C matrix initial value is 0</td>
  </tr>
</table>

### Detailed Scenario Description

This example selects different output scenarios through the compilation parameter `SCENARIO_NUM`. All scenarios are based on the same matrix multiplication specification: [M, N, K] = [30, 40, 70], kernel function name is `mmad_custom`.

**Scenario 1: int8_t input, int32_t output, C matrix initial value comes from C2**
- Input: A not transposed [30, 70] int8_t type, ND format; B not transposed [70, 40] int8_t type, ND format; Bias [40] int32_t type
- Output: C [30, 40] int32_t type, ND format
- Implementation: Use `Mmad` to implement matrix multiplication operation. Do not pass biasTensor through parameters: `mmadParams.cmatrixInitVal = false, mmadParams.cmatrixSource = true`, set C matrix initial value to come from C2
- Description: For int8_t type input with B matrix not transposed scenario, the N axis is aligned to 2 * 16, filling with a 32 * 16 fractal containing all invalid data. As shown in Figure 1 below, if setting `mmadParams.n = N`, it would read fractals numbered 3 and 7, but fail to read fractals numbered 9 and 10 containing valid data. Therefore, you need to set: `mmadParams.n = CeilAlign(N, BLOCK_CUBE * fractalNum)`. This will read all fractals. Although the matrix computation result contains results from invalid data participating in computation, the Fixpipe instruction ensures that results from invalid data are not moved out by setting `fixpipeParams.nSize = N` during data movement.
<p align="center">
  <img src="img/mmad_s8_L0B_转置.png" width="700">
</p>
<p align="center">
Figure 1: int8_t type, B not transposed, N axis actual alignment requirement differs from Mmad instruction default
</p>

**Scenario 2: bfloat16 input, float output, A not transposed, B transposed, C matrix initial value comes from CO1**
- Input: A not transposed [30, 70] bfloat16 type, ND format; B transposed [40, 70] bfloat16 type, ND format; No Bias, C matrix initial value comes from CO1
- Output: C [30, 40] float type, ND format
- Implementation: Use `Mmad` to implement matrix multiplication operation. Through parameters: `mmadParams.cmatrixInitVal = false, mmadParams.cmatrixSource = false`, set C matrix initial value to come from CO1
- Description: This scenario performs two Mmad computations. The first computation result is stored in CO1, serving as the C matrix initial value for the next computation. Finally, the two Mmad computation results are accumulated.

**Scenario 3: float input, float output, A transposed, B transposed, pass biasTensor, kDirectionAlign value set to true**
- Input: A transposed [70, 30] float type, ND format; B transposed [40, 70] float type, ND format; Bias [40] float type
- Output: C [30, 40] float type, ND format
- Implementation: Use `Mmad` to implement matrix multiplication operation, pass biasTensor. In this scenario, the `mmadParams.cmatrixSource` parameter is invalid
- Description: For float type input with A matrix transposed scenario, you need to use `mmadParams.kDirectionAlign` to resolve the issue that the K axis is actually aligned to `CeilAlign(K, 8*2)`, which differs from the Mmad instruction default requirement of aligning to `CeilAlign(K, 8)`. In this scenario, this parameter is set to true, and the K axis is aligned to `CeilAlign(K, 16)`. The matrix computation unit reads data from L0A and skips filled invalid data. In other scenarios, this parameter defaults to false, and the K axis is still aligned to `CeilAlign(K, 8)`, as shown in Figure 2:
<p align="center">
  <img src="img/mmad_f32_L0A_转置.png" width="1100">
</p>
<p align="center">
Figure 2: float type, A transposed, K axis actual alignment differs from Mmad instruction default requirement
</p>

**Scenario 4: int4b_t input, int32_t output, C matrix initial value is 0**

- Input: A not transposed [30, 70] int4b_t type, ND format; B transposed [40, 70] int4b_t type, ND format; No Bias
- Output: C [30, 40] int32_t type, ND format
- Implementation: Use `Mmad` to implement matrix multiplication operation. Through parameter: `mmadParams.cmatrixInitVal = true`, set C matrix initial value to 0
- Description: This scenario only supports Atlas A3 Training Series Products/Atlas A3 Inference Series Products/Atlas A2 Training Series Products/Atlas A2 Inference Series Products, and does not support adding Bias by passing biasTensor (scenario 3).

### Matrix Multiplication (Mmad)

The following section introduces how to configure the MmadParams structure members of the [Mmad](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0249.html) instruction. The specific meaning of each member variable will not be elaborated here.

Note that when the Mmad instruction executes, the matrix computation unit continuously reads multiple fractals from L0A/L0B to participate in matrix multiplication computation. The number of fractals read is calculated based on the values of member variables m, n, k in the MmadParams structure and the alignment requirements of the Mmad instruction for matrices A and B on L0A/L0B. Using B16 type input as an example: The Mmad instruction reads fractals continuously based on matrix A fractal as [16,16] and matrix B fractal as [16,16]. That is, the matrix computation unit reads total fractal counts from L0A/L0B as 2x5=10 and 5x3=15 respectively, and writes a total of 2x3=6 fractals to L0C. As shown in the figures below, Figure 3 shows Atlas A3 Training Series Products/Atlas A3 Inference Series Products and Atlas A2 Training Series Products/Atlas A2 Inference Series Products, Figure 4 shows Ascend 950PR/Ascend 950DT. The data layout on L0A differs: the former is Zz, the latter is Nz.
<p align="center">
  <img src="img/mmad_f16_A3.png" width="900">
</p>
<p align="center">
Figure 3: bfloat16 type, Zz layout on L0A, Mmad data layout diagram
</p>
<p align="center">
  <img src="img/mmad_f16_A5.png" width="900">
</p>
<p align="center">
Figure 4: bfloat16 type, Nz layout on L0A, Mmad data layout diagram
</p>

The Mmad computation includes padded invalid data, which needs to be excluded by the Fixpipe instruction during the L0C to GM movement process, removing invalid data filled during the Mmad computation process.

## Build and Run

Execute the following steps in the root directory of this example to build and run the operator.

- Configure environment variables

  Please select the corresponding environment variable configuration command based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on the current environment.

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

- Example execution
  ```bash
  SCENARIO=1
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO   # Generate test input data
  ./demo                           # Execute the compiled executable program, run the example
  python3 ../scripts/verify_result.py -scenarioNum=$SCENARIO output/output.bin output/golden.bin   # Verify output result correctness, confirm algorithm logic is correct
  ```

  When using NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clean cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then re-run cmake.

- Build option description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `sim` | Run mode: NPU run, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products/Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  | `SCENARIO_NUM` |  `1` (default), `2`, `3`, `4` | Scenario number, corresponding to int8_t / bfloat16 / float / int4b_t input data types respectively; `Only supported when CMAKE_ASC_ARCHITECTURES=dav-2201 for value 4` |

- Execution result

  The execution result is shown below, indicating the precision comparison passed.
  ```bash
  test pass!
  ```