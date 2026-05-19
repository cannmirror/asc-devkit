# load_data_l12l0 Example

## Overview

This example introduces the usage of relevant instructions for 14 matrix multiplication scenarios with ND format input and B4/B8/B16/B32 input data types (specifically using int4_t/int8_t/half/float as examples), covering combinations of left/right matrix transposition and non-transposition. The example focuses on the usage of the `LoadData` instruction. The overall process is as follows:

(1) How to call the `DataCopy` instruction with the `Nd2NzParams` structure parameter (referred to as `DataCopyND2NZ` in this example) when Matrix A (left input matrix for matrix multiplication) and Matrix B (right input matrix for matrix multiplication) are moved from GM -> L1.

(2) How to call `LoadData` with `LoadData2DParams` structure parameter (referred to as `Load2D` in this example), `LoadDataWithTranspose`, and `LoadData` with `LoadData3DParamsV2` structure parameter (referred to as `Load3DV2` in this example) for different scenarios when moving Matrix A and Matrix B from L1 -> L0A/L0B.

(3) Use the `Mmad` instruction to implement matrix multiplication calculation (C = A * B).

(4) Use the `Fixpipe` instruction to move the result matrix C from L0C -> GM.

The parameter configuration of each instruction and the data layout changes of each matrix before and after executing the instructions are explained with diagrams.

## Supported Products

- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── load_data_l12l0
│   ├── img                         // Diagrams
│   ├── scripts
│   │   ├── gen_data.py             // Input data and golden data generation script
│   │   └── verify_result.py        // Verification script for output data and golden data
│   ├── CMakeLists.txt              // Build project file
│   ├── data_utils.h                // Data read/write functions
│   └── load_data_l12l0.asc         // Ascend C example implementation & calling example
```

## Example Description

A complete matrix multiplication involves data movement processes including: GM -> L1, L1 -> L0A/L0B, L0C -> GM. The data layout formats of A and B matrices in different storage units are different, as shown in [Table 1](#table1):

<a name="table1"></a>
<table border="2" align="center">
<caption style="font-weight: normal;">
    <span style="font-weight: bold; font-size: 1.2em;">📌 Table 1: Data Layout Formats in Different Storage Units</span></caption>
  <tr>
    <td></td>
    <td align="center"><span style="font-weight: bold;">GM</span></td>
    <td align="center"><span style="font-weight: bold;">L1</span></td>
    <td align="center"><span style="font-weight: bold;">L0A</span></td>
    <td align="center"><span style="font-weight: bold;">L0B</span></td>
    <td align="center"><span style="font-weight: bold;">L0C</span></td>
  </tr>
  <tr>
    <td align="center"><span style="font-weight: bold;">Data Layout Format</span></td>
    <td align="center">ND</td>
    <td align="center">Nz</td>
    <td align="center">Zz</td>
    <td align="center">Zn</td>
    <td align="center">Zz</td>
  </tr>
</table>

When input data types are int4_t/int8_t/half/float respectively, the boolean variables isAtranspose and isBtranspose control whether A and B matrices are transposed during input, combining to create 13 scenarios.

Additionally, when matrix A is transposed during input and matrix B is not transposed during input, if A and B matrices on L1 are in Zz layout, the `LoadDataWithTranspose` interface can be called during L1->L0 movement. When the layout is the usual Nz layout, the `LoadData3DV2` interface needs to be called to complete the L1 -> L0 movement. Therefore, based on the original 13 scenarios, a special scenario needs to be added: A and B matrices on L1 are in Zz layout, and `LoadDataWithTranspose` interface is called for L1 -> L0 movement.

The following sections will introduce the data layout methods, alignment requirements, called instructions, and how to configure corresponding parameters for A and B matrices at various stages before and after the complete matrix multiplication process in the above 14 scenarios, focusing on the `LoadData` instruction.

(1) From L1 -> L0 path, commonly used movement instructions are `Load2D`, `LoadDataWithTranspose`, and `Load3DV2`. The available instructions for different scenarios and input data types are shown in [Table 2](#table2).

(2) In the program, loadData2AMode and loadData2BMode represent the movement instructions used for L1->L0A and L1->L0B respectively, as shown in [Table 3](#table3).

(3) The parameter scenarioNum represents the above 14 scenarios. The meaning of different values of scenarioNum and the movement instructions called during L1->L0 process are shown in [Table 4](#table4).

All scenarios are based on the same matrix multiplication specification: [m, n, k] = [40, 50, 70], kernel function name is "KernelLoadDataL12L0"

<a name="table2"></a>
<table border="2" align="center">
<caption style="font-weight: normal;">
    <span style="font-weight: bold; font-size: 1.2em;">📌 Table 2: Available `LoadData` Instructions for L1->L0 in Different Scenarios</span></caption>
  <tr>
    <td></td>
    <td align="center"><span style="font-weight: bold;">B4</span></td>
    <td align="center"><span style="font-weight: bold;">B8</span></td>
    <td align="center"><span style="font-weight: bold;">B16</span></td>
    <td align="center"><span style="font-weight: bold;">B32</span></td>
  </tr>
  <tr>
    <td align="center"><span style="font-weight: bold;">A not transposed input[m, k]<br>L1->L0A no transposition needed</span></td>
    <td align="center">`Load2D`, `Load3DV2`</td>
    <td align="center">`Load2D`, `Load3DV2`</td>
    <td align="center">`Load2D`, `Load3DV2`</td>
    <td align="center">`Load2D`, `Load3DV2`</td>
  </tr>
    <tr>
    <td align="center"><span style="font-weight: bold;">A transposed input[k, m]<br>L1->L0A transposition needed</span></td>
    <td align="center">Not supported</td>
    <td align="center">`LoadDataWithTranspose`</td>
    <td align="center">`Load2D`, `Load3DV2`, `LoadDataWithTranspose`</td>
    <td align="center">`Load3DV2`,<br>`LoadDataWithTranspose` (L1 data layout is Zz)</td>
  </tr>
    <tr>
    <td align="center"><span style="font-weight: bold;">B not transposed input[k, n]<br>L1->L0B transposition needed</span></td>
    <td align="center">`LoadDataWithTranspose`</td>
    <td align="center">`LoadDataWithTranspose`</td>
    <td align="center">`Load2D`, `Load3DV2`, `LoadDataWithTranspose`</td>
    <td align="center">`Load3DV2`,<br>`LoadDataWithTranspose` (L1 data layout is Zz)</td>
  </tr>
    <tr>
    <td align="center"><span style="font-weight: bold;">B transposed input[n, k]<br>L1->L0B no transposition needed</span></td>
    <td align="center">`Load2D`</td>
    <td align="center">`Load2D`</td>
    <td align="center">`Load2D`</td>
    <td align="center">`Load2D`</td>
  </tr>
</table>

<a name="table3"></a>
<table border="2" align="center">
<caption style="font-weight: normal;">
    <span style="font-weight: bold; font-size: 1.2em;">📌 Table 3: Meaning of Different loadDataMode Values</span>
  <tr>
    <td></td>
    <td align="center"><span style="font-weight: bold;">`LoadData` Instruction Called for Different loadDataMode Values</span></td>
  </tr>
  <tr>
    <td align="center"><span style="font-weight: bold;">0</span></td>
    <td colspan="3" align="center">`Load2D`</td>
  </tr>
    <tr>
    <td align="center"><span style="font-weight: bold;">1</span></td>
    <td colspan="3" align="center">`LoadDataWithTranspose`</td>
  </tr>
    <tr>
    <td align="center"><span style="font-weight: bold;">2</span></td>
    <td colspan="3" align="center">`Load3DV2`</td>
  </tr>
</table>

<a name="table4"></a>
<table border="2" align="center">
<caption style="font-weight: normal;">
    <span style="font-weight: bold; font-size: 1.2em;">📌 Table 4: Meaning of Different scenarioNum Values</span>
  <tr>
    <td ><span style="font-weight: bold;">scenarioNum</span></td>
    <td><span style="font-weight: bold;">Input Data Type</span></td>
    <td><span style="font-weight: bold;">Output Data Type</span></td>
    <td><span style="font-weight: bold;">Matrix A</span></td>
    <td><span style="font-weight: bold;">Matrix B</span></td>
    <td><span style="font-weight: bold;">`LoadData` Instruction Type for Matrix A</span></td>
    <td><span style="font-weight: bold;">`LoadData` Instruction Type for Matrix B</span></td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">1</span></td>
    <td rowspan="2" >int4_t</td>
    <td rowspan="2" >int32_t</td>
    <td>Not transposed</td>
    <td>Transposed</td>
    <td>`Load2D`</td>
    <td>`Load2D`</td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">2</span></td>
    <td>Not transposed</td>
    <td>Not transposed</td>
    <td>`Load3DV2`</td>
    <td>`LoadDataWithTranspose`</td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">3</span></td>
    <td rowspan="3" >int8_t</td>
    <td rowspan="3" >int32_t</td>
    <td>Not transposed</td>
    <td>Transposed</td>
    <td>`Load2D`</td>
    <td>`Load2D`</td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">4</span></td>
    <td>Not transposed</td>
    <td>Transposed</td>
    <td>`Load3DV2`</td>
    <td>`Load2D`</td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">5</span></td>
    <td>Transposed</td>
    <td>Not transposed</td>
    <td>`LoadDataWithTranspose`</td>
    <td>`LoadDataWithTranspose`</td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">6</span></td>
    <td rowspan="5" >half</td>
    <td rowspan="5" >float</td>
    <td>Not transposed</td>
    <td>Transposed</td>
    <td>`Load2D`</td>
    <td>`Load2D`</td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">7</span></td>
    <td>Not transposed</td>
    <td>Transposed</td>
    <td>`Load3DV2`</td>
    <td>`Load2D`</td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">8</span></td>
    <td>Transposed</td>
    <td>Not transposed</td>
    <td>`Load2D`</td>
    <td>`Load2D`</td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">9</span></td>
    <td>Transposed</td>
    <td>Not transposed</td>
    <td>`LoadDataWithTranspose`</td>
    <td>`LoadDataWithTranspose`</td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">10</span></td>
    <td>Transposed</td>
    <td>Not transposed</td>
    <td>`Load3DV2`</td>
    <td>`Load3DV2`</td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">11</span></td>
    <td rowspan="3" >float</td>
    <td rowspan="3" >float</td>
    <td>Not transposed</td>
    <td>Transposed</td>
    <td>`Load2D`</td>
    <td>`Load2D`</td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">12</span></td>
    <td>Not transposed</td>
    <td>Transposed</td>
    <td>`Load3DV2`</td>
    <td>`Load2D`</td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">13</span></td>
    <td>Transposed</td>
    <td>Not transposed</td>
    <td>`Load3DV2`</td>
    <td>`Load3DV2`</td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">14</span></td>
    <td rowspan="1" >float</td>
    <td rowspan="1" >float</td>
    <td>Transposed</td>
    <td>Not transposed</td>
    <td>`LoadDataWithTranspose`</td>
    <td>`LoadDataWithTranspose`</td>
  </tr>
</table>

Note: When scenarioNum is 1 to 13, A and B matrices on L1 are both in Nz layout; when scenarioNum=14, A and B matrices on L1 are both in Zz layout.

**Scenario 1: Input int4_t data type, isAtranspose=False, isBtranspose=True**
- Input A [40, 70], int4_t type, ND format; B [50, 70], int4_t type, ND format
- Output C [40, 50], int32_t type, ND format
- Implementation: Use `Load2D` to move matrix A from L1 to L0A, use `Load2D` to move matrix B from L1 to L0B
- Description: Matrix A loops along the m axis, moving CeilDivision(k, fractalShape[1]) fractals along the k axis direction at once. By configuring srcStride, dstGap and other parameters, matrix A L1 -> L0A movement and large fractal layout format change are completed. Similarly, matrix B loops along the k axis, moving CeilDivision(n, fractalShape[0]) fractals along the n direction at once. By configuring srcStride, dstGap and other parameters, matrix B L1 -> L0B movement and large fractal layout format change are completed.

**Scenario 2: Input int4_t data type, isAtranspose=False, isBtranspose=False**
- Input A [40, 70], int4_t type, ND format; B [70, 50], int4_t type, ND format
- Output C [40, 50], int32_t type, ND format
- Implementation: Use `Load3DV2` to move matrix A from L1 to L0A, use `LoadDataWithTranspose` to move matrix B from L1 to L0B
- Description: When `Load3DV2` instruction is configured with N=1, kernel width and height as 1, padding as 0, sliding stride as 1, kernel dilation coefficient as 1, the data layout after image to column expansion can be regarded as Nz fractal layout on L1. Then by configuring shape information of source and destination operands (l1H, l1W, kExtension, mExtension), matrix A is converted from Nz on L1 to the required Zz fractal layout in L0A. Through `LoadDataWithTranspose`, matrix B movement from L1 to L0B is completed. The movement process accompanies transposition, ultimately achieving small fractal transposed to n format and large fractal layout as Z.

**Scenario 3: Input int8_t data type, isAtranspose=False, isBtranspose=True**
- Input A [40, 70], int8_t type, ND format; B [50, 70], int8_t type, ND format
- Output C [40, 50], int32_t type, ND format
- Implementation: Use `Load2D` to move matrix A from L1 to L0A, use `Load2D` to move matrix B from L1 to L0B
- Description: Matrix A loops along the m axis, moving CeilDivision(k, fractalShape[1]) fractals along the k axis direction at once. By configuring srcStride, dstGap and other parameters, matrix A L1 -> L0A movement and large fractal layout format change are completed. Similarly, matrix B loops along the k axis, moving CeilDivision(n, fractalShape[0]) fractals along the n direction at once. By configuring srcStride, dstGap and other parameters, matrix B L1 -> L0B movement and large fractal layout format change are completed.

**Scenario 4: Input int8_t data type, isAtranspose=False, isBtranspose=True**
- Input A [40, 70], int8_t type, ND format; B [50, 70], int8_t type, ND format
- Output C [40, 50], int32_t type, ND format
- Implementation: Use `Load3DV2` to move matrix A from L1 to L0A, use `Load2D` to move matrix B from L1 to L0B
- Description: When `Load3DV2` instruction is configured with N=1, kernel width and height as 1, padding as 0, sliding stride as 1, kernel dilation coefficient as 1, the data layout after image to column expansion can be regarded as Nz fractal layout on L1. Then by configuring shape information of source and destination operands (l1H, l1W, kExtension, mExtension), matrix A is converted from Nz on L1 to the required Zz fractal layout in L0A. Matrix B loops along the k axis, moving CeilDivision(n, fractalShape[0]) fractals along the n direction at once. By configuring srcStride, dstGap and other parameters, matrix B large fractal layout format change is completed.

**Scenario 5: Input int8_t data type, isAtranspose=True, isBtranspose=False**
- Input A [70, 40], int8_t type, ND format; B [70, 50], int8_t type, ND format
- Output C [40, 50], int32_t type, ND format
- Implementation: Use `LoadDataWithTranspose` to move matrix A from L1 to L0A, use `LoadDataWithTranspose` to move matrix B from L1 to L0B
- Description: Through `LoadDataWithTranspose`, matrix A movement from L1 to L0A is completed. The movement process accompanies transposition. Through `LoadDataWithTranspose`, matrix B movement from L1 to L0B is completed. The movement process accompanies transposition.

**Scenario 6: Input half data type, isAtranspose=False, isBtranspose=True**
- Input A [40, 70], half type, ND format; B [50, 70], half type, ND format
- Output C [40, 50], float type, ND format
- Implementation: Use `Load2D` to move matrix A from L1 to L0A, use `Load2D` to move matrix B from L1 to L0B
- Description: Matrix A loops along the m axis, moving CeilDivision(k, fractalShape[1]) fractals along the k axis direction at once. By configuring srcStride, dstGap and other parameters, matrix A L1 -> L0A movement and large fractal layout format change are completed. Similarly, matrix B loops along the k axis, moving CeilDivision(n, fractalShape[0]) fractals along the n direction at once. By configuring srcStride, dstGap and other parameters, matrix B L1 -> L0B movement and large fractal layout format change are completed.

**Scenario 7: Input half data type, isAtranspose=False, isBtranspose=True**
- Input A [40, 70], half type, ND format; B [50, 70], half type, ND format
- Output C [40, 50], float type, ND format
- Implementation: Use `Load3DV2` to move matrix A from L1 to L0A, use `Load2D` to move matrix B from L1 to L0B
- Description: When `Load3DV2` instruction is configured with N=1, kernel width and height as 1, padding as 0, sliding stride as 1, kernel dilation coefficient as 1, the data layout after image to column expansion can be regarded as Nz fractal layout on L1. Then by configuring shape information of source and destination operands (l1H, l1W, kExtension, mExtension), matrix A is converted from Nz on L1 to the required Zz fractal layout in L0A. Matrix B loops along the k axis, moving CeilDivision(n, fractalShape[0]) fractals along the n direction at once. By configuring srcStride, dstGap and other parameters, matrix B L1 -> L0B movement and large fractal layout format change are completed.

**Scenario 8: Input half data type, isAtranspose=True, isBtranspose=False**
- Input A [70, 40], half type, ND format; B [70, 50], half type, ND format
- Output C [40, 50], float type, ND format
- Implementation: Use `Load2D` to move matrix A from L1 to L0A, use `Load2D` to move matrix B from L1 to L0B
- Description: Matrix A loops along the m axis, moving CeilDivision(k, fractalShape[1]) fractals along the k axis direction at once. By configuring ifTranspose, srcStride, dstGap and other parameters, small fractal transposition and large fractal layout format change are completed when moving matrix A from L1 to L0A. Similarly, matrix B loops along the k axis, moving CeilDivision(n, fractalShape[0]) fractals along the n direction at once. By configuring ifTranspose, srcStride, dstGap and other parameters, small fractal transposition and large fractal layout format change are completed when moving matrix B from L1 to L0B.

**Scenario 9: Input half data type, isAtranspose=True, isBtranspose=False**
- Input A [70, 40], half type, ND format; B [70, 50], half type, ND format
- Output C [40, 50], float type, ND format
- Implementation: Use `LoadDataWithTranspose` to move matrix A from L1 to L0A, use `LoadDataWithTranspose` to move matrix B from L1 to L0B
- Description: Through `LoadDataWithTranspose`, matrix A movement from L1 to L0A is completed. The movement process accompanies transposition. Through `LoadDataWithTranspose`, matrix B movement from L1 to L0B is completed. The movement process accompanies transposition.

**Scenario 10: Input half data type, isAtranspose=True, isBtranspose=False**
- Input A [70, 40], half type, ND format; B [70, 50], half type, ND format
- Output C [40, 50], float type, ND format
- Implementation: Use `Load3DV2` to move matrix A from L1 to L0A, use `Load3DV2` to move matrix B from L1 to L0B
- Description: When `Load3DV2` instruction is configured with N=1, kernel width and height as 1, padding as 0, sliding stride as 1, kernel dilation coefficient as 1, the data layout after image to column expansion can be regarded as Nz fractal layout on L1. Then by configuring transposition and shape information of source and destination operands (enTranspose, l1H, l1W, kExtension, mExtension), matrix A is moved from L1 to L0A. The movement process accompanies transposition. Matrix B calling `Load3DV2` defaults to enabling transposition, no need to configure enTranspose. Other configurations are similar to matrix A movement.

**Scenario 11: Input float data type, isAtranspose=False, isBtranspose=True**
- Input A [40, 70], float type, ND format; B [50, 70], float type, ND format
- Output C [40, 50], float type, ND format
- Implementation: Use `Load2D` to move matrix A from L1 to L0A, use `Load2D` to move matrix B from L1 to L0B
- Description: Matrix A loops along the m axis, moving CeilDivision(k, fractalShape[1]) fractals along the k axis direction at once. By configuring srcStride, dstGap and other parameters, matrix A L1 -> L0A movement and large fractal layout format change are completed. Similarly, matrix B loops along the k axis, moving CeilDivision(n, fractalShape[0]) fractals along the n direction at once. By configuring srcStride, dstGap and other parameters, matrix B L1 -> L0B movement and large fractal layout format change are completed.

**Scenario 12: Input float data type, isAtranspose=False, isBtranspose=True**
- Input A [40, 70], float type, ND format; B [50, 70], float type, ND format
- Output C [40, 50], float type, ND format
- Implementation: Use `Load3DV2` to move matrix A from L1 to L0A, use `Load2D` to move matrix B from L1 to L0B
- Description: When `Load3DV2` instruction is configured with N=1, kernel width and height as 1, padding as 0, sliding stride as 1, kernel dilation coefficient as 1, the data layout after image to column expansion can be regarded as Nz fractal layout on L1. Then by configuring shape information of source and destination operands (l1H, l1W, kExtension, mExtension), matrix A is converted from Nz on L1 to the required Zz fractal layout in L0A. Matrix B loops along the k axis, moving CeilDivision(n, fractalShape[0]) fractals along the n direction at once. By configuring srcStride, dstGap and other parameters, matrix B L1 -> L0B movement and large fractal layout format change are completed.

**Scenario 13: Input float data type, isAtranspose=True, isBtranspose=False**
- Input A [70, 40], float type, ND format; B [70, 50], float type, ND format
- Output C [40, 50], float type, ND format
- Implementation: Use `Load3DV2` to move matrix A from L1 to L0A, use `Load3DV2` to move matrix B from L1 to L0B
- Description: When `Load3DV2` instruction is configured with N=1, kernel width and height as 1, padding as 0, sliding stride as 1, kernel dilation coefficient as 1, the data layout after image to column expansion can be regarded as Nz fractal layout on L1. Then by configuring transposition and shape information of source and destination operands (enTranspose, l1H, l1W, kExtension, mExtension), matrix A is moved from L1 to L0A. The movement process accompanies transposition. Matrix B calling `Load3DV2` defaults to enabling transposition, no need to configure enTranspose. Other configurations are similar to matrix A movement.

**Scenario 14: Input float data type, isAtranspose=True, isBtranspose=False**
- Input A [70, 40], float type, ND format; B [70, 50], float type, ND format
- Output C [40, 50], float type, ND format
- Implementation: Use `LoadDataWithTranspose` to move matrix A from L1 to L0A, use `LoadDataWithTranspose` to move matrix B from L1 to L0B
- Description: When the layout format on L1 is Zz, calling `LoadDataWithTranspose` implements matrix A movement from L1 to L0A and matrix B movement from L1 to L0B. The movement process accompanies transposition.

For convenience of description, the following commonly used concepts are defined:

(1) fractalShape: The shape of a small fractal is [16, 32 / sizeof(T)], where T represents the input data type. Note: For B4 input data type, the shape is [16, 64]. The fractal-related information for data types involved in this example is shown in [Table 5](#table5).

(2) fractalSize: The number of elements contained in one small fractal. Refer to [Table 5](#table5) for details.

(3) fractalNum: When transposition is needed from L1 -> L0A/L0B and the `LoadDataWithTranspose` interface is called, this interface can only transpose one block matrix at a time. For B8 and B32 data types with fractalShape of [16,32] and [16,8] respectively, two consecutive small fractals need to be combined into one block and then transposed. Therefore, this parameter represents how many small fractals are contained in one block. Refer to [Table 5](#table5) for details.

<a name="table5"></a>
<table border="2" align="center">
<caption style="font-weight: normal;">
    <span style="font-weight: bold; font-size: 1.2em;">📌 Table 5: Fractal-related Information for Different Data Types</span></caption>
  <tr>
    <td></td>
    <td align="center"><span style="font-weight: bold;">fractalShape</span></td>
    <td align="center"><span style="font-weight: bold;">fractalSize</span></td>
    <td align="center"><span style="font-weight: bold;">fractalNum</span></td>
  </tr>
  <tr>
    <td align="center"><span style="font-weight: bold;">B4</span></td>
    <td align="center">[16, 64]</td>
    <td align="center">1024</td>
    <td align="center">4</td>
  </tr>
    <tr>
    <td align="center"><span style="font-weight: bold;">B8</span></td>
    <td align="center">[16, 32]</td>
    <td align="center">512</td>
    <td align="center">2</td>
  </tr>
    <tr>
    <td align="center"><span style="font-weight: bold;">B16</span></td>
    <td align="center">[16, 16]</td>
    <td align="center">256</td>
    <td align="center">1</td>
  </tr>
    <tr>
    <td align="center"><span style="font-weight: bold;">B32</span></td>
    <td align="center">[16, 8]</td>
    <td align="center">128</td>
    <td align="center">2</td>
  </tr>
</table>

(4) CeilAlign: Ceiling alignment operation. For example, when m=30, CeilAlign(30, 16)=32, which means aligning the m axis to 16, and the aligned m axis length is 32.

```cpp
__aicore__ inline uint16_t CeilAlign(uint16_t size, uint16_t alignValue) {
    return (size + alignValue - 1) / alignValue * alignValue;
}
```

(5) CeilDivision: Ceiling division, generally used to calculate the number of loops after ceiling alignment.

(6) mAlignValue: The m axis is aligned to mAlignValue. For example, mAlignValue=32 means the m axis is aligned to 32. Similarly, there are kAlignValue and nAlignValue.

(7) mAlignL0 and mAlignL1: The aligned values of the m axis when matrix A is on L1 and L0A respectively. Similarly, there are kAlignL0, kAlignL1, nAlignL0, nAlignL1.

(8) srcoffset and dstoffset: On L1, the address offset of LocalTensor when matrix A/B loops once in the outer axis direction; on L0A/L0B, the address offset of LocalTensor when matrix A/B loops once in the outer axis direction.

Note: For convenience of understanding, this example defaults to using the m axis of matrix A and the k axis of matrix B as the outer axis loop, and does not consider the scenario where the longer axis between m axis and k axis is used as the outer axis.

Additionally, the alignment requirements of matrices A and B on L1 and L0 in row and col directions are different. The alignment requirements for the 13 scenarios corresponding to scenarioNum values 1-13 in [Table 4](#table4) (when L1 layout format is Nz) are summarized in [Table 6](#table6) and [Table 7](#table7):

<a name="table6"></a>
<table border="2" align="center">
<caption style="font-weight: normal;">
    <span style="font-weight: bold; font-size: 1.2em;">📌 Table 6: Alignment Requirements for Each Axis of Matrices A and B on L1 (L1 Layout Format is Nz)</span></caption>
  <tr>
    <td></td>
    <td align="center"><span style="font-weight: bold;">B4 (fractalNum=4)</span></td>
    <td align="center"><span style="font-weight: bold;">B8 (fractalNum=2)</span></td>
    <td align="center"><span style="font-weight: bold;">B16 (fractalNum=1)</span></td>
    <td align="center"><span style="font-weight: bold;">B32 (fractalNum=2)</span></td>
  </tr>
  <tr>
    <td rowspan="2" align="center"><span style="font-weight: bold;">Matrix A not transposed input[m, k]</span></td>
    <td colspan="4" align="center">mAlignValue = fractalShape[0]</td>
  </tr>
  <tr>
    <td colspan="4" align="center" >kAlignValue = fractalShape[1]</td>
  </tr>
  <tr>
    <td rowspan="2" align="center"><span style="font-weight: bold;">Matrix A transposed input[k, m]</span></td>
    <td colspan="3" align="center">kAlignValue = fractalShape[0] * fractalNum</td>
    <td colspan="1" align="center">kAlignValue = fractalShape[0]</td>
  </tr>
  <tr>
    <td colspan="4" align="center" >mAlignValue = fractalShape[1]</td>
  </tr>
    <tr>
    <td rowspan="2" align="center"><span style="font-weight: bold;">Matrix B not transposed input[k, n]</span></td>
    <td colspan="3" align="center">kAlignValue = fractalShape[0] * fractalNum</td>
    <td colspan="1" align="center">kAlignValue = fractalShape[0]</td>
  </tr>
  <tr>
    <td colspan="4" align="center" >nAlignValue = fractalShape[1]</td>
  </tr>
 <tr>
    <td rowspan="2" align="center"><span style="font-weight: bold;">Matrix B transposed input[n, k]</span></td>
    <td colspan="4" align="center">nAlignValue = fractalShape[0]</td>
  </tr>
  <tr>
    <td colspan="4" align="center" >kAlignValue = fractalShape[1]</td>
  </tr>
</table>

<a name="table7"></a>
<table border="2" align="center">
<caption style="font-weight: normal;">
    <span style="font-weight: bold; font-size: 1.2em;">📌 Table 7: Alignment Requirements for Each Axis of Matrices A and B on L0</span></caption>
  <tr>
    <td></td>
    <td align="center"><span style="font-weight: bold;">B4 (fractalNum=4)</span></td>
    <td align="center"><span style="font-weight: bold;">B8 (fractalNum=2)</span></td>
    <td align="center"><span style="font-weight: bold;">B16 (fractalNum=1)</span></td>
    <td align="center"><span style="font-weight: bold;">B32 (fractalNum=2)</span></td>
  </tr>
  <tr>
    <td rowspan="2" align="center"><span style="font-weight: bold;">Matrix A not transposed input[m, k], L1->L0A no transposition needed</span></td>
    <td colspan="4" align="center">mAlignValue = fractalShape[0]</td>
  </tr>
  <tr>
    <td colspan="4" align="center" >kAlignValue = fractalShape[1]</td>
  </tr>
  <tr>
    <td rowspan="2" align="center"><span style="font-weight: bold;">Matrix A transposed input[k, m], L1->L0A transposition needed</span></td>
    <td colspan="3" align="center">kAlignValue = fractalShape[1]</td>
    <td >kAlignValue = fractalShape[1] * fractalNum</td>
  </tr>
  <tr>
    <td colspan="3" align="center" >mAlignValue = fractalShape[0] * fractalNum</td>
    <td align="center" >mAlignValue = fractalShape[1]</td>
  </tr>
    <tr>
    <td rowspan="2" align="center"><span style="font-weight: bold;">Matrix B not transposed input[k, n], L1->L0B transposition needed</span></td>
    <td colspan="3" align="center">kAlignValue = fractalShape[1]</td>
      <td align="center">kAlignValue = fractalShape[0]</td>
  </tr>
  <tr>
    <td colspan="3" align="center">nAlignValue = fractalShape[0] * fractalNum</td>
    <td align="center" >nAlignValue = fractalShape[1]</td>
  </tr>
 <tr>
    <td rowspan="2" align="center"><span style="font-weight: bold;">Matrix B transposed input[n, k], L1->L0B no transposition needed</span></td>
    <td colspan="4" align="center">nAlignValue = fractalShape[0]</td>
  </tr>
  <tr>
    <td colspan="4" align="center" >kAlignValue = fractalShape[1]</td>
  </tr>
</table>

Specifically, when scenarioNum=14, the layout format on L1 is Zz. The alignment requirements of matrices A and B on L1 and L0 in height and width directions are shown in [Table 8](#table8) and [Table 9](#table9):

<a name="table8"></a>
<table border="2" align="center">
<caption style="font-weight: normal;">
    <span style="font-weight: bold; font-size: 1.2em;">📌 Table 8: Alignment Requirements for Each Axis of Matrices A and B on L1 (Zz Layout) when scenarioNum=14</span></caption>
  <tr>
    <td align="center" ></td>
    <td align="center" ><span style="font-weight: bold;">float (fractalNum=2)</span></td>
  </tr>
   <tr>
    <td rowspan="2"><span style="font-weight: bold;">Matrix A transposed input[k, m]</span></td>
    <td align="center" >kAlignValue = fractalShape[0]</td>
  </tr>
    <tr>
    <td align="center" >mAlignValue = fractalShape[1]*fractalNum</td>
  </tr>
   <tr>
    <td rowspan="2"><span style="font-weight: bold;">Matrix B not transposed input[k, n]</span></td>
    <td align="center" >kAlignValue = fractalShape[0]</td>
  </tr>
    <tr>
    <td align="center" >nAlignValue = fractalShape[1]*fractalNum</td>
  </tr>
</table>

<a name="table9"></a>
<table border="2" align="center">
<caption style="font-weight: normal;">
    <span style="font-weight: bold; font-size: 1.2em;">📌 Table 9: Alignment Requirements for Each Axis of Matrices A and B on L0 when scenarioNum=14</span></caption>
  <tr>
    <td align="center" ></td>
    <td align="center" ><span style="font-weight: bold;">float (fractalNum=2)</span></td>
  </tr>
   <tr>
    <td rowspan="2"><span style="font-weight: bold;">Matrix A transposed input[k, m], L1->L0A transposition needed</span></td>
    <td align="center" >mAlignValue = fractalShape[0]</td>
  </tr>
    <tr>
    <td align="center" >kAlignValue = fractalShape[1]*fractalNum</td>
  </tr>
   <tr>
    <td rowspan="2"><span style="font-weight: bold;">Matrix B not transposed input[k, n], L1->L0B transposition needed</span></td>
    <td align="center" >kAlignValue = fractalShape[1]*fractalNum</td>
  </tr>
    <tr>
    <td align="center" >nAlignValue = fractalShape[0]</td>
  </tr>
</table>

### 1. Overall Process Diagram

The overall process diagram of matrix multiplication is shown below:

<p align="center">
  <img src="img/cube.png" width="800">
</p>

<p align="center">
Figure 1: Matrix Multiplication Overall Process Diagram
</p>

### 2. GM to L1 (`DataCopy`)

This section mainly introduces how to call the `DataCopy` interface to complete data movement and format transformation when the data layout format of matrices A and B on GM is ND and on L1 is Nz during the GM->L1 process. Specifically, it introduces scenario 14, float input data type, matrix A transposed input [k, m], matrix B not transposed input [k, n] when moved to L1 with Zz format data movement and format transformation.

#### 2.1. Matrix A GM->L1

For B4/B8/B16/B32 input, the movement logic of matrix A from GM(ND)->L1(Nz) is similar. Divided into 2 subsections according to matrix A GM input not transposed [m, k] and transposed [k, m], with half as an example for not transposed input and int8_t as an example for transposed input with diagram explanation. For other data types, only the dstNzC0Stride parameter is different. The dstNzC0Stride parameter takes the aligned length of matrix A on L1 in the row direction. Refer to [Table 6](#table6) and [Table 8](#table8) for details. Special scenario 14 (L1 layout is Zz) uses float input data type as an example.

##### 2.1.1. Matrix A GM Input is [m, k]

When matrix A GM input is not transposed ([m, k]), the half input data layout transformation is shown in the following diagram:

<p align="center">
  <img src="img/GM_L1_FP16_A_input_m_k_to_Nz.png" width="800">
</p>

<p align="center">
Figure 2: Matrix A not transposed input ([m,k]), half data type, GM -> L1 data layout diagram
</p>

The following section introduces how to configure the `Nd2NzParams` structure members of the [`DataCopy` inline conversion ND2NZ movement](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_00127.html) instruction. The specific meaning of each member variable will not be elaborated here. Note that the unit of dstNzC0Stride is 32B, and this parameter takes the aligned row count of the Nz matrix on L1.

```cpp
nd2nzA1Params.ndNum = 1;
nd2nzA1Params.nValue = m;
nd2nzA1Params.dValue = k;
nd2nzA1Params.srcNdMatrixStride = 0;
nd2nzA1Params.srcDValue = k;

// The following parameter takes the aligned length of matrix A on L1 in the row direction
nd2nzA1Params.dstNzC0Stride = CeilAlign(m, fractalShape[0]);

nd2nzA1Params.dstNzNStride = 1;
nd2nzA1Params.dstNzMatrixStride = 0;
```

##### 2.1.2. Matrix A GM Input is [k, m]

**(1) L1 Layout is Nz**

When matrix A GM input is transposed ([k, m]) with float input data, the L1 layout format transformation to Nz is shown in the following diagram:

<p align="center">
  <img src="img/GM_L1_FP32_A_transInput_k_m_to_Nz.png" width="800">
</p>

<p align="center">
Figure 3: Matrix A transposed input, float data type, GM -> L1, ND -> Nz
</p>

When configuring the `Nd2NzParams` structure members, note that the source operand shape is [k, m] and the unit of dstNzC0Stride is 32B. This parameter takes the aligned row count of the Nz matrix on L1.

```cpp
nd2nzA1Params.ndNum = 1;
nd2nzA1Params.nValue = k;
nd2nzA1Params.dValue = m;
nd2nzA1Params.srcNdMatrixStride = 0;
nd2nzA1Params.srcDValue = m;
nd2nzA1Params.dstNzNStride = 1;
nd2nzA1Params.dstNzMatrixStride = 0;
// The following parameter takes the aligned length of matrix A on L1 in the row direction
if constexpr (AscendC::IsSameType<T, float>::value) {
  nd2nzA1Params.dstNzC0Stride = CeilAlign(k, fractalShape[0]);
}
```

**(2) L1 Layout is Zz**

When matrix A GM input is transposed ([k, m]) with float input data, the L1 layout format transformation to Zz is shown in the following diagram:

<p align="center">
  <img src="img/GM_L1_FP32_A_inputTrans_k_m_to_Zz.png" width="800">
</p>

<p align="center">
Figure 4: Matrix A transposed input, float data type, GM -> L1, ND -> Zz
</p>

As shown in Figure 4 above, when matrix A is transposed during input with float input data type, if you want to call the `LoadDataWithTranspose` interface to implement L1 -> L0A data movement and transposition, the layout of matrix A on L1 must be Zz. Therefore, during the GM -> L1 stage, when calling the `DataCopyND2NZ` instruction, you need to configure the `Nd2NzParams` structure cleverly to achieve the ND2Zz effect.

The core idea of `DataCopyND2NZ` instruction achieving ND2Zz effect: Treat one ND matrix as CeilDivision(k, 16) ND matrices by slicing along the height axis with stride 16. Since the moved CeilDivision(k, 16) Nz matrices have only one fractal in the height axis direction, the matrix A finally moved to L1 is equivalent to Zz layout.

```cpp
nd2nzA1Params.ndNum = CeilDivision(k, fractalShape[0]);
nd2nzA1Params.nValue = fractalShape[0];
nd2nzA1Params.dValue = m;
nd2nzA1Params.srcNdMatrixStride = fractalShape[0] * m;
nd2nzA1Params.srcDValue = m;
nd2nzA1Params.dstNzC0Stride = fractalShape[0];
nd2nzA1Params.dstNzNStride = 1;
nd2nzA1Params.dstNzMatrixStride = fractalShape[0] * CeilAlign(m, fractalShape[1] * fractalNum);
```

#### 2.2. Matrix B GM->L1

For B4/B8/B16/B32 input, the movement logic of matrix B from GM(ND)->L1(Nz) is similar. Divided into 2 subsections according to matrix B GM input not transposed [k, n] and transposed [n, k], with float as an example for not transposed input and half as an example for transposed input with diagram explanation. For other data types, only the dstNzC0Stride parameter is different. The dstNzC0Stride parameter takes the aligned length of matrix B on L1 in the row direction. Refer to [Table 6](#table6) and [Table 8](#table8) for details. Special scenario 14 uses float input data type as an example.

##### 2.2.1. Matrix B GM Input is [k, n]

**(1) L1 Layout is Nz**

When matrix B GM input is not transposed ([k, n]) with float input data, the L1 layout format transformation to Nz is shown in the following diagram:

<p align="center">
  <img src="img/GM_L1_FP32_B_input_k_n_to_Nz.png" width="800">
</p>

<p align="center">
Figure 5: Matrix B not transposed input, float data type, GM -> L1, ND -> Nz
</p>

When configuring the `Nd2NzParams` structure members, note that the source operand shape is [k, n], the unit of dstNzC0Stride is 32B, and this parameter takes the aligned row count of the Nz matrix on L1.

```cpp
nd2nzB1Params.ndNum = 1;
nd2nzB1Params.nValue = k;
nd2nzB1Params.dValue = n;
nd2nzB1Params.srcNdMatrixStride = 0;
nd2nzB1Params.srcDValue = n;
nd2nzB1Params.dstNzNStride = 1;
nd2nzB1Params.dstNzMatrixStride = 0;
nd2nzB1Params.dstNzC0Stride = CeilAlign(k, fractalShape[0]);
```

**(2) L1 Layout is Zz**

When matrix B GM input is not transposed ([k, n]) with float input data, the L1 layout format transformation to Zz is shown in the following diagram:

<p align="center">
  <img src="img/GM_L1_FP32_B_input_k_n_to_Zz.png" width="800">
</p>

<p align="center">
Figure 6: Matrix B not transposed input, float data type, GM -> L1, ND -> Zz
</p>

Similar to subsection 2.1.2, when the input is float and the data layout format on L1 is Zz:

During the GM->L1 stage, you need to configure the `Nd2NzParams` structure cleverly when calling the `DataCopyND2NZ` instruction to achieve the ND2ZZ effect. The core idea of `DataCopyND2NZ` instruction achieving ND2Zz effect is to treat one ND matrix as CeilDivision(k, 16) ND matrices by slicing along the height axis with stride 16. Since the moved CeilDivision(k, 16) Nz matrices have only one fractal in the height axis direction, the matrix B finally moved to L1 is equivalent to Zz layout.

```cpp
nd2nzB1Params.ndNum = CeilDivision(k, fractalShape[0]);
nd2nzB1Params.nValue = fractalShape[0];
nd2nzB1Params.dValue = n;
nd2nzB1Params.srcNdMatrixStride = fractalShape[0] * n;
nd2nzB1Params.srcDValue = n;
nd2nzB1Params.dstNzC0Stride = fractalShape[0];
nd2nzB1Params.dstNzNStride = 1;
nd2nzB1Params.dstNzMatrixStride = fractalShape[0] * CeilAlign(n, fractalShape[1] * fractalNum);
```

##### 2.2.2. Matrix B GM Input is [n, k]

When matrix B GM input is transposed ([n, k]) with half input data, the L1 layout transformation is shown in the following diagram:

<p align="center">
  <img src="img/GM_L1_FP16_B_transInput_n_k_to_Nz.png" width="800">
</p>

<p align="center">
Figure 7: Matrix B transposed, half data type, GM -> L1, ND -> Nz
</p>

The following section introduces how to configure the `Nd2NzParams` structure members of the [`DataCopy` inline conversion ND2NZ movement](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_00127.html) instruction. The specific meaning of each member variable will not be elaborated here. Note that the unit of dstNzC0Stride is 32B, and this parameter takes the aligned row count of the Nz matrix on L1.

```cpp
nd2nzB1Params.ndNum = 1;
nd2nzB1Params.nValue = n;
nd2nzB1Params.dValue = k;
nd2nzB1Params.srcNdMatrixStride = 0;
nd2nzB1Params.srcDValue = k;

// The following parameter takes the aligned length of matrix B on L1 in the row direction
nd2nzB1Params.dstNzC0Stride = CeilAlign(n, fractalShape[0]);
nd2nzB1Params.dstNzNStride = 1;
nd2nzB1Params.dstNzMatrixStride = 0;
```

### 3. L1 to L0 (`LoadData`)

Usually the data layout format of matrices A/B on L1 is Nz, while on L0A and L0B they are Zz and Zn respectively. When the L1 layout format is Nz, you can call `Load2D`, `LoadDataWithTranspose`, and `Load3DV2` interfaces to complete data movement and format transformation during the L1->L0 process. Specifically, this section shows calling `LoadDataWithTranspose` to complete L1->L0 data movement and format transformation when the L1 layout format is Zz.

#### 3.1. Matrix A L1->L0A

For B4/B8/B16/B32 input data types, the available `LoadData` related instructions are different when moving matrix A from L1 to L0A in transposed and non-transposed scenarios. See [Table 2](#table2) for details. The following sections will introduce these scenarios.

##### 3.1.1. Matrix A L1->L0A Non-transposed

When L1 -> L0A is not transposed, only the large fractal layout format changes. In this scenario, all four data types B4/B8/B16/B32 can use the `Load2D` interface and `Load3DV2` interface to implement data movement. The parameter configuration is basically the same, only the fractalShape is different. Refer to [Table 5](#table5). Half is used as an example for diagram illustration.

**(1) `Load2D` Interface**

Calling the `Load2D` interface is shown in the following diagram:

<p align="center">
  <img src="img/L1_L0A_FP16_A_Load2D.png" width="800">
</p>

<p align="center">
Figure 8: half data type, L1 -> L0A non-transposed, calling `Load2D` data layout diagram
</p>

The following section introduces how to configure the `LoadData2DParams` structure members of the [`Load2D`](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_00169.html) instruction. The specific meaning of each member variable will not be elaborated here.

As shown in Figure 8, the m axis direction is used as the outer axis for the for loop (shown in the red box), and the k axis direction is used as the inner axis to configure loadDataParams.repeatTimes. The meanings of srcoffset and dstoffset are: on L1, the address offset of LocalTensor when matrix A loops once in the m axis direction; on L0A, the address offset of LocalTensor when matrix A loops once in the m axis direction.

```cpp
uint32_t dstOffset = CeilDivision(k, fractalShape[1]) * fractalSize;
uint32_t srcOffset = fractalSize;
// Nz -> Zz
AscendC::LoadData2DParams loadDataParams;
loadDataParams.repeatTimes = CeilDivision(k, fractalShape[1]);
loadDataParams.srcStride = CeilDivision(m, fractalShape[0]);
// In the K axis direction between adjacent iterations, the gap between the end address of the previous fractal and the start address of the next fractal in the destination operand
loadDataParams.dstGap = 0;
loadDataParams.ifTranspose = false;
for (int i = 0; i < CeilDivision(m, fractalShape[0]); ++i) {
    AscendC::LoadData(a2Local[i * dstOffset], a1Local[i * srcOffset], loadDataParams);
}
```

**(2) `Load3DV2` Interface**

Calling the `Load3DV2` interface is shown below:

<p align="center">
  <img src="img/L1_L0A_F16_A_Load3DV2.png" width="5000">
</p>

<p align="center">
Figure 9: half data type, L1 -> L0A non-transposed, calling `Load3DV2` data layout diagram
</p>

The essence of `Load3D` is to complete Image to Column expansion for Feature Map in NC1HWC0 format, and then select specified data blocks from the expanded two-dimensional matrix and move them to corresponding memory locations. When configured with parameters as shown in the following code, calling one [`Load3Dv2`](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_00170.html) instruction can achieve data layout format conversion from Nz to Zz when moving from L1 to L0A. According to the process of `Load3Dv2` instruction completing img2col, it can be known that after img2col, the height of matrix A is ho * wo. According to the calculation formula of ho and wo, substituting kernel width, kernel sliding stride, kernel dilation coefficient and other parameters, it can be known: the height of matrix A is CeilAlign(m, fractalShape[0]); after img2col, the width of matrix A is ho * wo, ci * kh * kw, substituting kh=1, kw=1, it can be known that the width of matrix A is CeilAlign(k, fractalShape[1]).

```cpp
// Load3DV2: Nz -> Zz
AscendC::LoadData3DParamsV2<T> loadDataParams;
// Source operand height
loadDataParams.l1H = 1;
// Source operand width
loadDataParams.l1W = CeilAlign(m, fractalShape[0]);
// Source operand channel count
// After img2col, the result matrix height is ho * wo. According to the calculation formula of ho and wo, substituting kernel width, kernel sliding stride, kernel dilation coefficient and other parameters: ho * wo = loadDataParams.l1H * loadDataParams.l1W
// After img2col, the result matrix width is ci * kh * kw. Substituting kh=1, kw=1, the result matrix width is ci=loadDataParams.channelSize = m
loadDataParams.channelSize = CeilAlign(k, fractalShape[1]);
// The transfer length of this instruction in the destination operand width dimension. If not covering the rightmost fractal, for half type, it should be a multiple of 16; for int8_t/uint8_t, it should be a multiple of 32. If covering, there is no multiple requirement.
loadDataParams.kExtension = CeilAlign(k, fractalShape[1]);
// The transfer length of this instruction in the destination operand height dimension. If not covering the bottom fractal, for half/int8_t/uint8_t, it should be a multiple of 16. If covering, there is no multiple requirement.
loadDataParams.mExtension = CeilAlign(m, fractalShape[0]);
// The stride of kernel sliding in the source operand width dimension
loadDataParams.strideW = 1;
// The stride of kernel sliding in the source operand height dimension
loadDataParams.strideH = 1;
// Kernel width
loadDataParams.filterW = 1;
// Kernel height
loadDataParams.filterH = 1;
// Kernel width dilation coefficient
loadDataParams.dilationFilterW = 1;
// Kernel height dilation coefficient
loadDataParams.dilationFilterH = 1;
loadDataParams.filterSizeW = false;
loadDataParams.filterSizeH = false;
loadDataParams.enTranspose = false;
loadDataParams.fMatrixCtrl = false;
```

##### 3.1.2. Matrix A L1->L0A Transposed

When L1->L0A, the large fractal layout format changes, and the small fractal needs to be transposed. In this scenario, Atlas A3 Training Series Products/Atlas A3 Inference Series Products and Atlas A2 Training Series Products/Atlas A2 Inference Series Products do not support B4 data type. The available interfaces for B8/B16/B32 data types are different. Using int8_t, half, and float as examples, the available interfaces and diagram explanations are introduced for different data types in separate subsections.

###### 3.1.2.1 Input Data Type is int8_t

Calling the `LoadDataWithTranspose` interface is shown below:

<p align="center">
  <img src="img/L1_L0A_B8_A_trans_LoadDataWithTranspose.png">
</p>

<p align="center">
Figure 10: int8_t data type, L1 -> L0A transposed, calling `LoadDataWithTranspose` data layout diagram
</p>

Since the small fractal is transposed, the `LoadDataWithTranspose` interface can be called. The following section introduces how to configure the `LoadData2dTransposeParams` structure members of the [`LoadDataWithTranspose`](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0239.html) instruction. The specific meaning of each member variable will not be elaborated here.

As shown in Figure 10, the m axis direction is used as the outer axis for the for loop, as shown in the red box, and the k axis direction is used as the inner axis to configure loadDataParams.repeatTimes. Note that since transposition merges two consecutive fractals into one block, loadDataParams.repeatTimes=CeilDivision(k, fractalShape[0] * fractalNum), as the blue box and green box each represent one block.

```cpp
// LoadDataWithTranspose: Nz-> Zz
// According to the following function prototype, the data type of offset is uint32_t
// __aicore__ inline LocalTensor operator[](const uint32_t offset) const
// dstoffset needs to be calculated according to the alignment of matrix A on L0 in the width direction
uint32_t dstOffset = CeilDivision(k, fractalShape[1]) * fractalSize * fractalNum;
// srcoffset needs to be calculated according to the alignment of matrix A on L1 in the height direction
uint32_t srcOffset = CeilDivision(k, fractalShape[0] * fractalNum) * fractalSize * fractalNum;

AscendC::LoadData2dTransposeParams loadDataParams;
// The starting position of movement is the nth block matrix in the source operand (0 means the 1st block matrix in the source operand)
loadDataParams.startIndex = 0;
// Number of iterations, each iteration transposes one block matrix
loadDataParams.repeatTimes = CeilDivision(k, fractalShape[0] * fractalNum);
// Between adjacent iterations, the gap between the start addresses of the previous fractal and the next fractal in the source operand. Unit is the size of one block matrix
loadDataParams.srcStride = 1;
// Between adjacent iterations, the gap from the end address of the first fractal of the previous iteration to the start address of the first fractal of the next iteration in the destination operand. Unit: 512B
loadDataParams.dstGap = 0;
// Within each iteration, the gap between the end address of the previous fractal before transposition and the start address of the next fractal in the destination operand. Unit: 512B
loadDataParams.dstFracGap = CeilDivision(k, fractalShape[1]) - 1;
for (int i = 0; i < CeilDivision(m, fractalShape[1]); ++i) {
    AscendC::LoadDataWithTranspose(a2Local[i * dstOffset], a1Local[i * srcOffset], loadDataParams);
}
```

###### 3.1.2.2 Input Data Type is half

**(1) `Load2D` Interface**

Calling the `Load2D` interface is shown below:

<p align="center">
  <img src="img/L1_L0A_F16_A_trans_Load2D.png">
</p>

<p align="center">
Figure 11: half data type, L1 -> L0A transposed, calling `Load2D` data layout diagram
</p>

The following section introduces how to configure the `LoadData2DParams` structure members of the [`Load2D`](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_00169.html) instruction. The specific meaning of each member variable will not be elaborated here.

As shown in Figure 11, the m axis direction is used as the outer axis for the for loop (shown in the red box), and the k axis direction is used as the inner axis to configure loadDataParams.repeatTimes. The meanings of srcoffset and dstoffset are: on L1, the address offset of LocalTensor when matrix A loops once in the m axis direction; on L0A, the address offset of LocalTensor when matrix A loops once in the m axis direction. The loadDataParams.ifTranspose parameter is configured as true, indicating that each small fractal is transposed from L1 -> L0A.

```cpp
uint32_t dstOffset = CeilDivision(k, fractalShape[0]) * fractalSize;
uint32_t srcOffset = CeilDivision(k, fractalShape[0]) * fractalSize;
AscendC::LoadData2DParams loadDataParams;
// Number of iterations, each iteration can process 512B data
loadDataParams.repeatTimes = CeilDivision(k, fractalShape[0]);
// Between adjacent iterations, the gap between the start addresses of the previous fractal and the next fractal in the source operand. Unit: 512B
loadDataParams.srcStride = 1;
// Between adjacent iterations, the gap between the end address of the previous fractal and the start address of the next fractal in the destination operand. Unit: 512B
loadDataParams.dstGap = 0;
// Whether to enable transposition function, transposing each fractal matrix. Default is false
loadDataParams.ifTranspose = true;
for (int i = 0; i < CeilDivision(m, fractalShape[1]); ++i) {
    AscendC::LoadData(a2Local[i * dstOffset], a1Local[i * srcOffset], loadDataParams);
}
```

**(2) `LoadDataWithTranspose` Interface**

The diagram for calling the `LoadDataWithTranspose` interface is consistent with `Load2D`. The m axis direction is used as the outer axis for the for loop (shown in the red box in Figure 11), and the k axis direction is used as the inner axis to configure loadDataParams.repeatTimes. Default small fractal transposition.

```cpp
// dstoffset needs to be calculated according to the alignment of matrix A on L0 in the width direction
uint32_t dstOffset = CeilDivision(k, fractalShape[1]) * fractalSize * fractalNum;
// srcoffset needs to be calculated according to the alignment of matrix A on L1 in the height direction
uint32_t srcOffset = CeilDivision(k, fractalShape[0] * fractalNum) * fractalSize * fractalNum;

AscendC::LoadData2dTransposeParams loadDataParams;
// The starting position of movement is the nth block matrix in the source operand (0 means the 1st block matrix in the source operand)
loadDataParams.startIndex = 0;
// Number of iterations, each iteration transposes one block matrix
loadDataParams.repeatTimes = CeilDivision(k, fractalShape[0] * fractalNum);
// Between adjacent iterations, the gap between the start addresses of the previous fractal and the next fractal in the source operand. Unit is the size of one block matrix
loadDataParams.srcStride = 1;
// Between adjacent iterations, the gap from the end address of the first fractal of the previous iteration to the start address of the first fractal of the next iteration in the destination operand. Unit: 512B
loadDataParams.dstGap = 0;
// Within each iteration, the gap between the end address of the previous fractal before transposition and the start address of the next fractal in the destination operand. Unit: 512B
loadDataParams.dstFracGap = CeilDivision(k, fractalShape[1]) - 1;
for (int i = 0; i < CeilDivision(m, fractalShape[1]); ++i) {
```