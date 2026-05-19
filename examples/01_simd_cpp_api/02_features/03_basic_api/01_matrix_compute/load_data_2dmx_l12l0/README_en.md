# load_data_2dmx_l12l0 Sample

## Overview
This sample demonstrates how to use relevant instructions in MX matrix multiplication scenarios with ND format input, where A/B matrices use FP4 (fp4x2_e1m2_t / fp4x2_e2m1_t) and FP8 (fp8_e4m3fn_t / fp8_e5m2_t) data types, and quantization matrices scaleA/scaleB use fp8_e8m0_t data type. The sample covers 6 quantization matrix multiplication scenarios formed by combinations of left matrix and left quantization matrix, right matrix and right quantization matrix transpose/non-transpose. This sample focuses on the usage of the `LoadData` instruction and its `LoadData2DParamsV2` and `LoadData2DMxParams` structure parameters. The overall workflow is as follows:<br>
(1) Matrix A (left input matrix of matrix multiplication) and Matrix B (right input matrix of matrix multiplication) are transferred from GM -> L1 using the `DataCopy` instruction, with transfer controlled by the `Nd2NzParams` structure parameter;<br>
(2) Matrix scaleA (left quantization input matrix) and Matrix scaleB (right quantization input matrix) are transferred from GM -> L1 using the `DataCopy` instruction, with transfer controlled by `Nd2NzParams` or `Dn2NzParams` structure parameter;<br>
(3) A/B/scaleA/scaleB matrices are transferred from L1 -> L0A/L0B/L0A_MX/L0B_MX using the `LoadData` instruction, where A/B matrix data transfer is controlled by `LoadData2DParamsV2` structure parameter, and scaleA/scaleB matrix data transfer is controlled by `LoadData2DMxParams` structure parameter;<br>
(4) Matrix multiplication with quantization is implemented using the `Mmad` instruction (C = (scaleA ⊗ A) * (scaleB ⊗ B), where "⊗" represents broadcast multiplication. When left/right matrices multiply with left/right quantization coefficient matrices, every 32 elements in the K direction share one quantization factor);<br>
(5) The result matrix C is transferred from L0C -> GM using the `Fixpipe` instruction.<br>
The parameter configuration of each instruction and the data layout changes before and after executing instructions are illustrated with diagrams.

## Supported Products
- Ascend 950PR/Ascend 950DT

## Directory Structure
```
├── load_data_2dmx_l12l0
│   ├── scripts
│   │   ├── gen_data.py             // Input data and golden data generation script
│   │   └── verify_result.py        // Verification script for comparing output data with golden data
│   ├── CMakeLists.txt              // Build project file
│   ├── data_utils.h                // Data read/write functions
│   ├── load_data_2dmx_l12l0.asc    // Ascend C sample implementation & sample invocation
│   └── README.md                   // Sample documentation
```

## Sample Description
A complete MX matrix multiplication involves data transfer processes: GM -> L1 -> L0A/L0B/L0A_MX/L0B_MX -> L0C -> GM. The A/B matrices and scaleA/scaleB matrices have different data layout formats in different storage units, as shown in [Table 1](#table-1):<br>

<a name="table-1"></a>
<table border="2" align="center">
<caption style="font-weight: normal;">
    <span style="font-weight: bold; font-size: 1.2em;">📌 Table 1: Data Layout Formats in Different Storage Units</span></caption>
  <tr>
    <td></td>
    <td align="center"><span style="font-weight: bold;">GM</span></td>
    <td align="center"><span style="font-weight: bold;">L1</span></td>
    <td align="center"><span style="font-weight: bold;">L0A</span></td>
    <td align="center"><span style="font-weight: bold;">L0A_MX</span></td>
    <td align="center"><span style="font-weight: bold;">L0B</span></td>
    <td align="center"><span style="font-weight: bold;">L0B_MX</span></td>
    <td align="center"><span style="font-weight: bold;">L0C</span></td>
  </tr>
  <tr>
    <td align="center"><span style="font-weight: bold;">Matrix A</span></td>
    <td align="center">ND</td>
    <td align="center">Nz</td>
    <td align="center">Nz</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><span style="font-weight: bold;">Matrix B</span></td>
    <td align="center">ND</td>
    <td align="center">Nz</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">Zn</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><span style="font-weight: bold;">Matrix scaleA</span></td>
    <td align="center">ND</td>
    <td align="center">Zz</td>
    <td align="center">-</td>
    <td align="center">Zz</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><span style="font-weight: bold;">Matrix scaleB</span></td>
    <td align="center">ND</td>
    <td align="center">Nn</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">Nn</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><span style="font-weight: bold;">Matrix C</span></td>
    <td align="center">ND</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">Nz</td>
  </tr>
</table>

When input data types are FP4/FP8, Boolean variables isAtranspose and isBtranspose control whether A and scaleA, B and scaleB matrices are input with transpose, forming 4 basic scenarios. Additionally, when Matrix A is input with transpose [k, m], if a single `LoadData` call would transfer more than 1 fractal of dirty data, a for-loop approach is needed to avoid transferring redundant dirty data. Therefore, for-loop scenarios (scenarios 5 and 6) are added for FP4 and FP8 input data types when A/B matrices are input with transpose, totaling 6 scenarios.

The following sections describe the data layout, alignment requirements, instructions called, and parameter configuration for A/B matrices and scaleA/scaleB matrices in each stage of the complete MX matrix multiplication workflow for the above 6 scenarios, focusing on the usage of the `LoadData` instruction and its `LoadData2DParamsV2` and `LoadData2DMxParams` structure parameters.<br>
(1) From L1 -> L0 path, call `LoadData` instruction with two structure parameters. A/B matrices use `LoadData2DParamsV2` structure parameter, scale matrices use `LoadData2DMxParams` structure parameter. Available transfer methods for different scenarios and data types are shown in [Table 2](#table-2);<br>
(2) Use parameter scenarioNum to represent the above 6 scenarios. The meaning of different scenarioNum values and transfer instructions called during L1 -> L0 process are shown in [Table 3](#table-3).<br>
All scenarios use the same matrix multiplication specification: [m, n, k] = [40, 50, 70], kernel function name is "KernelLoadDataL12L0Load2DMX".

<a name="table-2"></a>
<table border="2" align="center">
<caption style="font-weight: normal;">
    <span style="font-weight: bold; font-size: 1.2em;">📌 Table 2: L1 -> L0, Transfer Instructions for A/B Matrices and Scale Matrices in Different Scenarios</span></caption>
  <tr>
    <td></td>
    <td align="center"><span style="font-weight: bold;">FP4</span></td>
    <td align="center"><span style="font-weight: bold;">FP8</span></td>
  </tr>
  <tr>
    <td align="center"><span style="font-weight: bold;">isAtranspose=false<br>A input without transpose [m, k]<br>scaleA input without transpose [m, scaleK/2, 2]<br>L1 -> L0A does not require transpose</span></td>
    <td align="center">LoadData(LoadData2DParamsV2(no transpose), LoadData2DMxParams)</td>
    <td align="center">LoadData(LoadData2DParamsV2(no transpose), LoadData2DMxParams)</td>
  </tr>
    <tr>
    <td align="center"><span style="font-weight: bold;">isAtranspose=true<br>A input with transpose [k, m]<br>scaleA input with transpose [scaleK, m, 2]<br>L1 -> L0A requires transpose<br>(single call)</span></td>
    <td align="center">LoadData(LoadData2DParamsV2(transpose), LoadData2DMxParams)</td>
    <td align="center">LoadData(LoadData2DParamsV2(transpose), LoadData2DMxParams)</td>
  </tr>
    <tr>
    <td align="center"><span style="font-weight: bold;">isBtranspose=false<br>B input without transpose [k, n]<br>scaleB input without transpose [scaleK, n, 2]<br>L1 -> L0B requires transpose</span></td>
    <td align="center">LoadData(LoadData2DParamsV2(transpose), LoadData2DMxParams)</td>
    <td align="center">LoadData(LoadData2DParamsV2(transpose), LoadData2DMxParams)</td>
  </tr>
    <tr>
    <td align="center"><span style="font-weight: bold;">isBtranspose=true<br>B input with transpose [n, k]<br>scaleB input with transpose [n, scaleK/2, 2]<br>L1 -> L0B does not require transpose</span></td>
    <td align="center">LoadData(LoadData2DParamsV2(no transpose), LoadData2DMxParams)</td>
    <td align="center">LoadData(LoadData2DParamsV2(no transpose), LoadData2DMxParams)</td>
  </tr>
</table>

<a name="table-3"></a>
<table border="2" align="center">
<caption style="font-weight: normal;">
    <span style="font-weight: bold; font-size: 1.2em;">📌 Table 3: Meaning of Different scenarioNum Values</span></caption>
  <tr>
    <td ><span style="font-weight: bold;">scenarioNum</span></td>
    <td><span style="font-weight: bold;">A/B Data Type</span></td>
    <td><span style="font-weight: bold;">Matrix A Type</span></td>
    <td><span style="font-weight: bold;">Matrix B Type</span></td>
    <td><span style="font-weight: bold;">Output Data Type</span></td>
    <td><span style="font-weight: bold;">isAtranspose</span></td>
    <td><span style="font-weight: bold;">isBtranspose</span></td>
    <td><span style="font-weight: bold;">Matrix A Transfer Method</span></td>
    <td><span style="font-weight: bold;">Matrix B Transfer Method</span></td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">1</span></td>
    <td>FP4</td>
    <td>fp4x2_e1m2_t</td>
    <td>fp4x2_e2m1_t</td>
    <td>float</td>
    <td>false</td>
    <td>true</td>
    <td>LoadData(LoadData2DParamsV2(no transpose), LoadData2DMxParams)</td>
    <td>LoadData(LoadData2DParamsV2(no transpose), LoadData2DMxParams)</td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">2</span></td>
    <td>FP4</td>
    <td>fp4x2_e2m1_t</td>
    <td>fp4x2_e1m2_t</td>
    <td>float</td>
    <td>true</td>
    <td>false</td>
    <td>LoadData(LoadData2DParamsV2(transpose), LoadData2DMxParams)</td>
    <td>LoadData(LoadData2DParamsV2(transpose), LoadData2DMxParams)</td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">3</span></td>
    <td>FP8</td>
    <td>fp8_e4m3fn_t</td>
    <td>fp8_e5m2_t</td>
    <td>float</td>
    <td>false</td>
    <td>true</td>
    <td>LoadData(LoadData2DParamsV2(no transpose), LoadData2DMxParams)</td>
    <td>LoadData(LoadData2DParamsV2(no transpose), LoadData2DMxParams)</td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">4</span></td>
    <td>FP8</td>
    <td>fp8_e5m2_t</td>
    <td>fp8_e4m3fn_t</td>
    <td>float</td>
    <td>true</td>
    <td>false</td>
    <td>LoadData(LoadData2DParamsV2(transpose), LoadData2DMxParams)</td>
    <td>LoadData(LoadData2DParamsV2(transpose), LoadData2DMxParams)</td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">5</span></td>
    <td>FP4</td>
    <td>fp4x2_e2m1_t</td>
    <td>fp4x2_e1m2_t</td>
    <td>float</td>
    <td>true</td>
    <td>false</td>
    <td>for-loop + LoadData(LoadData2DParamsV2(transpose), LoadData2DMxParams)</td>
    <td>LoadData(LoadData2DParamsV2(transpose), LoadData2DMxParams)</td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">6</span></td>
    <td>FP8</td>
    <td>fp8_e5m2_t</td>
    <td>fp8_e4m3fn_t</td>
    <td>float</td>
    <td>true</td>
    <td>false</td>
    <td>for-loop + LoadData(LoadData2DParamsV2(transpose), LoadData2DMxParams)</td>
    <td>LoadData(LoadData2DParamsV2(transpose), LoadData2DMxParams)</td>
  </tr>
</table>

**Scenario 1: Input FP4 data type, isAtranspose=false, isBtranspose=true**
- Input A [40, 70], fp4x2_e1m2_t type, ND format; B [50, 70], fp4x2_e2m1_t type, ND format;
- scaleA [40, 4], fp8_e8m0_t type; scaleB [4, 50], fp8_e8m0_t type;
- Output C [40, 50], float type, ND format;
- Implementation: Call `LoadData` instruction with `LoadData2DParamsV2` (ifTranspose=false) and `LoadData2DMxParams` structure parameters to simultaneously transfer Matrix A from L1 to L0A and Matrix scaleA from L1 to L0A_MX, as shown in Figure 1. Similarly for Matrix B, simultaneously transfer Matrix B from L1 to L0B and Matrix scaleB from L1 to L0B_MX, as shown in Figure 2;
- Notes:
  - Matrix A is input without transpose [m, k] on GM, L1 -> L0A does not require transpose, `LoadData2DParamsV2` ifTranspose is set to false;
  - Matrix B is input with transpose [n, k] on GM, L1 -> L0B does not require transpose, `LoadData2DParamsV2` ifTranspose is set to false;
  - scaleA is input as [m, scaleK] on GM, scaleB is input as [n, scaleK] on GM. scaleA/scaleB matrices are transferred from L1 to L0A_MX/L0B_MX through `LoadData2DMxParams` in the same `LoadData` call.

<p align="center">
  <img src="img/whole_process/B4_A_scaleA_.png" width="1000">
</p>

<p align="center">
Figure 1: Scenario 1 MX Matrix Multiplication GM -> L1 -> L0A/L0A_MX Flowchart
</p>

<p align="center">
  <img src="img/whole_process/B4_B_scaleB_NK.png" width="1000">
</p>

<p align="center">
Figure 2: Scenario 1 MX Matrix Multiplication GM -> L1 -> L0B/L0B_MX Flowchart
</p>

**Scenario 2: Input FP4 data type, isAtranspose=true, isBtranspose=false**
- Input A [70, 40], fp4x2_e2m1_t type, ND format; B [70, 50], fp4x2_e1m2_t type, ND format;
- scaleA [4, 40, 2], fp8_e8m0_t type; scaleB [4, 50, 2], fp8_e8m0_t type;
- Output C [40, 50], float type, ND format;
- Implementation: Call `LoadData` instruction with `LoadData2DParamsV2` (ifTranspose=true) and `LoadData2DMxParams` structure parameters to simultaneously transfer Matrix A from L1 to L0A and Matrix scaleA from L1 to L0A_MX, as shown in Figure 3. Similarly for Matrix B, simultaneously transfer Matrix B from L1 to L0B and Matrix scaleB from L1 to L0B_MX, as shown in Figure 4;
- Notes:
  - Matrix A is input with transpose [k, m] on GM, L1 -> L0A requires transpose, `LoadData2DParamsV2` ifTranspose is set to true. Small fractal transpose and large fractal layout format change occur, with more than 1 fractal of redundant data transferred in the m direction;
  - Matrix B is input without transpose [k, n] on GM, L1 -> L0B requires transpose, `LoadData2DParamsV2` ifTranspose is set to true. Small fractal transpose and large fractal layout format change occur, with no more than 1 fractal of redundant data transferred in the n direction;
  - scaleA is input as [scaleK/2, m, 2] on GM, scaleB is input as [scaleK/2, n, 2] on GM. scaleA/scaleB matrices are transferred from L1 to L0A_MX/L0B_MX through `LoadData2DMxParams` in the same `LoadData` call.<br>
  - Since A transpose causes more than 1 fractal of redundant data to be transferred in the m direction, the `Mmad` instruction needs to set mmadParams.m = CeilAlign(m, fractalShape[0] * fractalNum) to allow the redundant fractal to participate in computation. The `Fixpipe` transfer skips the results from invalid fractals participating in computation.

<p align="center">
  <img src="img/whole_process/B4_A_scaleA_trans_KM.png" width="1000">
</p>

<p align="center">
Figure 3: Scenario 2 MX Matrix Multiplication GM -> L1 -> L0A/L0A_MX Flowchart
</p>

<p align="center">
  <img src="img/whole_process/B4_B_scaleB_trans_KN.png" width="1000">
</p>

<p align="center">
Figure 4: Scenario 2 MX Matrix Multiplication GM -> L1 -> L0B/L0B_MX Flowchart
</p>

**Scenario 3: Input FP8 data type, isAtranspose=false, isBtranspose=true**
- Input A [40, 70], fp8_e4m3fn_t type, ND format; B [50, 70], fp8_e5m2_t type, ND format;
- scaleA [40, 4], fp8_e8m0_t type; scaleB [4, 50], fp8_e8m0_t type;
- Output C [40, 50], float type, ND format;
- Implementation: Call `LoadData` instruction with `LoadData2DParamsV2` (ifTranspose=false) and `LoadData2DMxParams` structure parameters to simultaneously transfer Matrix A from L1 to L0A and Matrix scaleA from L1 to L0A_MX, as shown in Figure 5. Similarly for Matrix B, simultaneously transfer Matrix B from L1 to L0B and Matrix scaleB from L1 to L0B_MX, as shown in Figure 6;
- Notes: Similar to Scenario 1, but with FP8 data type.
  - Matrix A is input without transpose [m, k] on GM, L1 -> L0A does not require transpose, `LoadData2DParamsV2` ifTranspose is set to false;
  - Matrix B is input with transpose [n, k] on GM, L1 -> L0B does not require transpose, `LoadData2DParamsV2` ifTranspose is set to false.
  - scaleA is input as [m, scaleK] on GM, scaleB is input as [n, scaleK] on GM. scaleA/scaleB matrices are transferred from L1 to L0A_MX/L0B_MX through `LoadData2DMxParams` in the same `LoadData` call.<br>
  - With FP8 data type, when Matrix A is input without transpose [m, k] and Matrix B is input with transpose [n, k], the k direction is in the col direction. `DataCopy` only aligns the k direction to 32B, so you need to set the tail data in the k direction to 0 on L1 to prevent dirty data from participating in computation;

<p align="center">
  <img src="img/whole_process/B8_A_scaleA_.png" width="1000">
</p>

<p align="center">
Figure 5: Scenario 3 MX Matrix Multiplication GM -> L1 -> L0A/L0A_MX Flowchart
</p>

<p align="center">
  <img src="img/whole_process/B8_B_scaleB_NK.png" width="1000">
</p>

<p align="center">
Figure 6: Scenario 3 MX Matrix Multiplication GM -> L1 -> L0B/L0B_MX Flowchart
</p>

**Scenario 4: Input FP8 data type, isAtranspose=true, isBtranspose=false**
- Input A [70, 40], fp8_e5m2_t type, ND format; B [70, 50], fp8_e4m3fn_t type, ND format;
- scaleA [4, 40, 2], fp8_e8m0_t type; scaleB [4, 50, 2], fp8_e8m0_t type;
- Output C [40, 50], float type, ND format;
- Implementation: Call `LoadData` instruction with `LoadData2DParamsV2` (ifTranspose=true) and `LoadData2DMxParams` structure parameters to simultaneously transfer Matrix A from L1 to L0A and Matrix scaleA from L1 to L0A_MX, as shown in Figure 7. Similarly for Matrix B, simultaneously transfer Matrix B from L1 to L0B and Matrix scaleB from L1 to L0B_MX, as shown in Figure 8;
- Notes: Similar to Scenario 2, but with FP8 data type.
  - Matrix A is input with transpose [k, m] on GM, L1 -> L0A requires transpose, `LoadData2DParamsV2` ifTranspose is set to true. Small fractal transpose and large fractal layout format change occur, with more than 1 fractal of redundant data transferred in the m direction;
  - Matrix B is input without transpose [k, n] on GM, L1 -> L0B requires transpose, `LoadData2DParamsV2` ifTranspose is set to true. Small fractal transpose and large fractal layout format change occur;
  - scaleA is input as [scaleK/2, m, 2] on GM, scaleB is input as [scaleK/2, n, 2] on GM. scaleA/scaleB matrices are transferred from L1 to L0A_MX/L0B_MX through `LoadData2DMxParams` in the same `LoadData` call.<br>
  - Additionally, when Matrix A is input with transpose [k, m], the remaining dirty data in the k direction needs to be set to 0 on L1; when Matrix B is input without transpose [k, n], the remaining dirty data in the k direction also needs to be set to 0 on L1. The `Mmad` instruction also needs to set mmadParams.m = CeilAlign(m, fractalShape[0] * fractalNum).

<p align="center">
  <img src="img/whole_process/B8_A_scaleA_trans_KM.png" width="1000">
</p>

<p align="center">
Figure 7: Scenario 4 MX Matrix Multiplication GM -> L1 -> L0A/L0A_MX Flowchart
</p>

<p align="center">
  <img src="img/whole_process/B8_B_scaleB_trans_KN.png" width="1000">
</p>

<p align="center">
Figure 8: Scenario 4 MX Matrix Multiplication GM -> L1 -> L0B/L0B_MX Flowchart
</p>

**Scenario 5: Input FP4 data type, isAtranspose=true, isBtranspose=false**
- Input A [70, 40], fp4x2_e2m1_t type, ND format; B [70, 50], fp4x2_e1m2_t type, ND format;
- scaleA [4, 40, 2], fp8_e8m0_t type; scaleB [4, 50, 2], fp8_e8m0_t type;
- Output C [40, 50], float type, ND format;
- Implementation: Matrix A uses a for-loop to call `LoadData` instruction with `LoadData2DParamsV2` (ifTranspose=true) and `LoadData2DMxParams` structure parameters. Each loop transfers part of Matrix A to L0A, while Matrix scaleA is transferred entirely to L0A_MX only in the first for-loop, subsequent loops do not perform transfer, as shown in Figure 9. Matrix B uses a single `LoadData` call to simultaneously transfer Matrix B to L0B and Matrix scaleB to L0B_MX, as shown in Figure 10;
- Notes:
  - Matrix A is input with transpose [k, m] on GM, L1 -> L0A requires transpose. Use for-loop to call `LoadData` to avoid transferring redundant dirty data. Small fractal transpose and large fractal layout format change occur. Each for-loop skips tail dirty data fractals in the m direction on L0A, with no more than 1 fractal of redundant data transferred in the m direction;
  - Matrix B is input without transpose [k, n] on GM, L1 -> L0B requires transpose. Single `LoadData` call for transfer, `LoadData2DParamsV2` ifTranspose is set to true. Small fractal transpose and large fractal layout format change occur, with no more than 1 fractal of redundant data transferred in the n direction;
  - scaleA is input as [scaleK/2, m, 2] on GM, scaleB is input as [scaleK/2, n, 2] on GM. scaleA/scaleB matrices are transferred from L1 to L0A_MX/L0B_MX through `LoadData2DMxParams` in the same `LoadData` call.<br>
  - Additionally, when Matrix A is input with transpose [k, m], the remaining dirty data in the k direction needs to be set to 0 on L1; when Matrix B is input without transpose [k, n], the remaining dirty data in the k direction also needs to be set to 0 on L1.

<p align="center">
  <img src="img/whole_process/B4_A_scaleA_for_trans_KM.png" width="1000">
</p>

<p align="center">
Figure 9: Scenario 5 MX Matrix Multiplication GM -> L1 -> L0A/L0A_MX Flowchart
</p>

<p align="center">
  <img src="img/whole_process/B4_B_scaleB_trans_KN.png" width="1000">
</p>

<p align="center">
Figure 10: Scenario 5 MX Matrix Multiplication GM -> L1 -> L0B/L0B_MX Flowchart
</p>

**Scenario 6: Input FP8 data type, isAtranspose=true, isBtranspose=false**
- Input A [70, 40], fp8_e5m2_t type, ND format; B [70, 50], fp8_e4m3fn_t type, ND format;
- scaleA [4, 40, 2], fp8_e8m0_t type; scaleB [4, 50, 2], fp8_e8m0_t type;
- Output C [40, 50], float type, ND format;
- Implementation: Matrix A uses a for-loop to call `LoadData` instruction with `LoadData2DParamsV2` (ifTranspose=true) and `LoadData2DMxParams` structure parameters. Each loop transfers part of Matrix A to L0A, while Matrix scaleA is transferred entirely to L0A_MX only in the first for-loop, subsequent loops do not perform transfer, as shown in Figure 11. Matrix B uses a single `LoadData` call to simultaneously transfer Matrix B to L0B and Matrix scaleB to L0B_MX, as shown in Figure 12;
- Notes: Similar to Scenario 5, but with FP8 data type.
  - Matrix A is input with transpose [k, m] on GM. Use for-loop to call `LoadData` to avoid transferring redundant dirty data. Small fractal transpose and large fractal layout format change occur;
  - Matrix B is input without transpose [k, n] on GM. Single `LoadData` call for transfer, `LoadData2DParamsV2` ifTranspose is set to true. Small fractal transpose and large fractal layout format change occur.
  - scaleA is input as [scaleK/2, m, 2] on GM, scaleB is input as [scaleK/2, n, 2] on GM. scaleA/scaleB matrices are transferred from L1 to L0A_MX/L0B_MX through `LoadData2DMxParams` in the same `LoadData` call.<br>
  - Additionally, both Matrix A and Matrix B need to set the remaining dirty data in the k direction to 0 on L1.

<p align="center">
  <img src="img/whole_process/B8_A_scaleA_for_trans_KM.png" width="1000">
</p>

<p align="center">
Figure 11: Scenario 6 MX Matrix Multiplication GM -> L1 -> L0A/L0A_MX Flowchart
</p>

<p align="center">
  <img src="img/whole_process/B8_B_scaleB_trans_KN.png" width="1000">
</p>

<p align="center">
Figure 12: Scenario 6 MX Matrix Multiplication GM -> L1 -> L0B/L0B_MX Flowchart
</p>


For clarity, the following commonly used concepts are defined:

(1) fractalShape: The shape of a small fractal. For FP4 data type, it is [16, 64]; for FP8 data type, it is [16, 32]. Fractal-related information for data types involved in this sample is shown in [Table 5](#table-5).

(2) fractalSize: The number of elements contained in one small fractal. For FP4, it is 1024; for FP8, it is 512.

(3) fractalNum: The number of small fractals contained in one block. For FP4, it is 4; for FP8, it is 2. When `LoadData` `LoadData2DParamsV2` parameter is configured for transpose transfer, consecutive fractalNum small fractals are merged into one block and then transposed.

(4) packedK: The actual number of k-axis elements stored for A/B matrices on GM. For FP4 data type, 2 fp4 elements are packed into 1 fp4x2 element, so packedK = CeilDivision(k, 2); for FP8 data type, packedK = k.

(5) scaleK: The aligned length of the scale matrix k-axis. scaleK = CeilDivision(k, SCALE_BASE_FACTOR) * SCALE_EVEN_NUMBER, where SCALE_BASE_FACTOR=64, SCALE_EVEN_NUMBER=2. In this sample, when k=70, scaleK = CeilDivision(70, 64) * 2 = 4.

(6) alignK: The aligned length of A/B matrix k-axis. alignK = CeilAlign(k, SCALE_BASE_FACTOR) = CeilAlign(k, 64). In this sample, when k=70, alignK = CeilAlign(70, 64) = 128.

<a name="table-5"></a>
<table border="2" align="center">
<caption style="font-weight: normal;">
    <span style="font-weight: bold; font-size: 1.2em;">📌 Table 5: Fractal-Related Information for Different Data Types</span></caption>
  <tr>
    <td></td>
    <td align="center"><span style="font-weight: bold;">fractalShape</span></td>
    <td align="center"><span style="font-weight: bold;">fractalSize</span></td>
    <td align="center"><span style="font-weight: bold;">fractalNum</span></td>
    <td align="center"><span style="font-weight: bold;">packedK</span></td>
  </tr>
  <tr>
    <td align="center"><span style="font-weight: bold;">FP4</span></td>
    <td align="center">[16, 64]</td>
    <td align="center">1024</td>
    <td align="center">4</td>
    <td align="center">CeilDivision(k, 2)</td>
  </tr>
    <tr>
    <td align="center"><span style="font-weight: bold;">FP8</span></td>
    <td align="center">[16, 32]</td>
    <td align="center">512</td>
    <td align="center">2</td>
    <td align="center">k</td>
  </tr>
</table>

(7) CeilAlign: Round-up alignment operation. For example, when m=40, CeilAlign(40, 16)=48, which means aligning the m-axis to 16, resulting in an aligned m-axis length of 48.

      __aicore__ inline uint16_t CeilAlign(uint16_t size, uint16_t alignValue) {
          return (size + alignValue - 1) / alignValue * alignValue;
      }

(8) CeilDivision: Round-up division, generally used to calculate the loop count after round-up alignment.

(9) mAlignValue: Align the m-axis to mAlignValue. For example, mAlignValue=16 means the m-axis is aligned to 16. Similarly, there are kAlignValue and nAlignValue. In MX scenarios, kAlignValue = SCALE_BASE_FACTOR = 64.

(10) mAlignL1 and mAlignL0: The aligned values of Matrix A on L1 and L0A respectively. Similarly, there are kaAlignL1, kaAlignL0, nAlignL1, nAlignL0, kbAlignL1, kbAlignL0.

A and B matrices have different alignment requirements on L1 and L0 in each axis direction. The alignment requirements for the 6 scenarios in [Table 3](#table-3) are summarized in [Table 6](#table-6) and [Table 7](#table-7):

<a name="table-6"></a>
<table border="2" align="center">
<caption style="font-weight: normal;">
    <span style="font-weight: bold; font-size: 1.2em;">📌 Table 6: Alignment Requirements for A and B Matrices on Each Axis in L1 (L1 Layout Format is Nz)</span></caption>
  <tr>
    <td></td>
    <td align="center"><span style="font-weight: bold;">FP4 (fractalNum=4)</span></td>
    <td align="center"><span style="font-weight: bold;">FP8 (fractalNum=2)</span></td>
  </tr>
  <tr>
    <td rowspan="2" align="center"><span style="font-weight: bold;">Matrix A input without transpose [m, k]</span></td>
    <td colspan="2" align="center">mAlignValue = fractalShape[0]</td>
  </tr>
  <tr>
    <td colspan="2" align="center" >kAlignValue = SCALE_BASE_FACTOR = 64</td>
  </tr>
  <tr>
    <td rowspan="2" align="center"><span style="font-weight: bold;">Matrix A input with transpose [k, m]</span></td>
    <td align="center">kAlignValue = SCALE_BASE_FACTOR = 64</td>
    <td align="center">kAlignValue = SCALE_BASE_FACTOR = 64</td>
  </tr>
  <tr>
    <td align="center" >mAlignValue = fractalShape[1] = 64</td>
    <td align="center" >mAlignValue = fractalShape[1] = 32</td>
  </tr>
    <tr>
    <td rowspan="2" align="center"><span style="font-weight: bold;">Matrix B input without transpose [k, n]</span></td>
    <td align="center">kAlignValue = SCALE_BASE_FACTOR = 64</td>
    <td align="center">kAlignValue = SCALE_BASE_FACTOR = 64</td>
  </tr>
  <tr>
    <td align="center" >nAlignValue = fractalShape[1] = 64</td>
    <td align="center" >nAlignValue = fractalShape[1] = 32</td>
  </tr>
 <tr>
    <td rowspan="2" align="center"><span style="font-weight: bold;">Matrix B input with transpose [n, k]</span></td>
    <td colspan="2" align="center">nAlignValue = fractalShape[0]</td>
  </tr>
  <tr>
    <td colspan="2" align="center" >kAlignValue = SCALE_BASE_FACTOR = 64</td>
  </tr>
</table>

<a name="table-7"></a>
<table border="2" align="center">
<caption style="font-weight: normal;">
    <span style="font-weight: bold; font-size: 1.2em;">📌 Table 7: Alignment Requirements for A and B Matrices on Each Axis in L0</span></caption>
  <tr>
    <td></td>
    <td align="center"><span style="font-weight: bold;">FP4 (fractalNum=4)</span></td>
    <td align="center"><span style="font-weight: bold;">FP8 (fractalNum=2)</span></td>
  </tr>
  <tr>
    <td rowspan="2" align="center"><span style="font-weight: bold;">Matrix A input without transpose [m, k], L1 -> L0A does not require transpose<br>(Scenario 1/3)</span></td>
    <td colspan="2" align="center">mAlignValue = fractalShape[0]</td>
  </tr>
  <tr>
    <td colspan="2" align="center" >kAlignValue = SCALE_BASE_FACTOR = 64</td>
  </tr>
  <tr>
    <td rowspan="2" align="center"><span style="font-weight: bold;">Matrix A input with transpose [k, m], L1 -> L0A requires transpose<br>(Scenario 2/4, single call)</span></td>
    <td align="center">mAlignValue = fractalShape[0] * fractalNum = 64</td>
    <td align="center">mAlignValue = fractalShape[0] * fractalNum = 32</td>
  </tr>
  <tr>
    <td colspan="2" align="center" >kAlignValue = SCALE_BASE_FACTOR = 64</td>
  </tr>
  <tr>
    <td rowspan="2" align="center"><span style="font-weight: bold;">Matrix A input with transpose [k, m], L1 -> L0A requires transpose<br>(Scenario 5/6, for-loop call)</span></td>
    <td colspan="2" align="center">mAlignValue = fractalShape[0]</td>
  </tr>
  <tr>
    <td colspan="2" align="center" >kAlignValue = SCALE_BASE_FACTOR = 64</td>
  </tr>
    <tr>
    <td rowspan="2" align="center"><span style="font-weight: bold;">Matrix B input without transpose [k, n], L1 -> L0B requires transpose<br>(Scenario 2/4/5/6)</span></td>
    <td align="center">nAlignValue = fractalShape[0] * fractalNum = 64</td>
    <td align="center">nAlignValue = fractalShape[0] * fractalNum = 32</td>
  </tr>
  <tr>
    <td colspan="2" align="center" >kAlignValue = SCALE_BASE_FACTOR = 64</td>
  </tr>
 <tr>
    <td rowspan="2" align="center"><span style="font-weight: bold;">Matrix B input with transpose [n, k], L1 -> L0B does not require transpose<br>(Scenario 1/3)</span></td>
    <td colspan="2" align="center">nAlignValue = fractalShape[0]</td>
  </tr>
  <tr>
    <td colspan="2" align="center" >kAlignValue = SCALE_BASE_FACTOR = 64</td>
  </tr>
</table>

Alignment requirements for scaleA/scaleB matrices are as follows:

<a name="table-8"></a>
<table border="2" align="center">
<caption style="font-weight: normal;">
    <span style="font-weight: bold; font-size: 1.2em;">📌 Table 8: Alignment Requirements for scaleA/scaleB Matrices on Each Axis in L1/L0</span></caption>
  <tr>
    <td></td>
    <td align="center"><span style="font-weight: bold;">Matrix scaleA (L1 layout format is Zz)</span></td>
    <td align="center"><span style="font-weight: bold;">Matrix scaleB (L1 layout format is Nn)</span></td>
  </tr>
  <tr>
    <td align="center"><span style="font-weight: bold;">m-axis / n-axis alignment</span></td>
    <td align="center">scaleMAlignL1 = CeilAlign(m, fractalShape[0])</td>
    <td align="center">scaleNAlignL1 = CeilAlign(n, fractalShape[0])</td>
  </tr>
  <tr>
    <td align="center"><span style="font-weight: bold;">k-axis alignment</span></td>
    <td align="center">scaleK = CeilDivision(k, 64) * 2</td>
    <td align="center">scaleK = CeilDivision(k, 64) * 2</td>
  </tr>
</table>

### 1. Overall Workflow

The overall MX matrix multiplication workflow is as follows:

```
GM(ND) --DataCopy--> L1(Nz) --LoadData(LoadData2DParamsV2, LoadData2DMxParams)--> L0A(Nz)/L0B(Zn)/L0A_MX/L0B_MX --Mmad--> L0C(Nz) --Fixpipe--> GM(ND)
```

**Step-by-step Explanation**:

1. **GM -> L1**:
   - Call `DataCopy` instruction with `Nd2NzParams` structure parameter to implement ND to Nz format conversion (A/B matrices)
   - Call `DataCopy` instruction with `Nd2NzParams` or `Dn2NzParams` structure parameter and process as B16 data type to implement ND to Zz and Nn format conversion (scaleA/scaleB matrices)
   - Use `Fill` to align and fill with zeros to prevent dirty data from participating in computation
2. **L1 -> L0**:
   - Call `LoadData` instruction with `LoadData2DParamsV2` and `LoadData2DMxParams` structure parameters
   - `LoadData2DParamsV2` controls A/B matrix transfer, `LoadData2DMxParams` controls scale matrix transfer
3. **Matrix Multiplication**: Use `Mmad` interface to execute MX matrix multiplication
4. **L0C -> GM**: Use `Fixpipe` interface to transfer out results

### 2. GM to L1 (`DataCopy` and `Fill`)
This section describes the transfer of A/B/scaleA/scaleB from GM to L1 and the fill-zero operations required on L1 due to instruction constraints:<br>
(1) When A/B matrices have ND layout format on GM and Nz format on L1, call `DataCopy` instruction with `Nd2NzParams` structure parameter during GM -> L1 process to complete data transfer and format conversion;<br>
(2) When scaleA/scaleB matrices have ND format on GM and Zz and Nn formats on L1 respectively, call `DataCopy` instruction with `Nd2NzParams` or `Dn2NzParams` structure parameter during GM -> L1 process to complete data transfer and format conversion. Since transpose is not supported during L1 -> L0A_MX/L0B_MX process, the layout format of scaleA/scaleB needs to be changed to match the layout format on L0A_MX and L0B_MX during GM -> L1 process.<br>
(3) Since MX matrix multiplication requires the Mmad instruction to align the k direction to 64, and the alignment behavior of `DataCopy` during GM -> L1 stage depends on the axis where k is located, you need to set the remaining k-direction data to 0 by scenario to prevent dirty data from participating in computation:
- When k is in the col direction, for FP8 data type, `DataCopy` only aligns the k direction to 32B (that is, to 32 elements), which needs to be further padded to align to 64 elements.
- When k is in the row direction, `DataCopy` aligns the k direction to 16, which needs to be further padded to align to 64;
#### 2.1. Matrix A GM -> L1

##### 2.1.1. Matrix A GM input as [m, k]

<p align="center">
  <img src="img/GM2L1/FP4_A_GM2L1_MK.png" width="900">
</p>

<p align="center">
Figure 13: FP4 data type, Matrix A [m, k] input, GM -> L1, ND -> Nz
</p>

<p align="center">
  <img src="img/GM2L1/FP8_A_GM2L1_MK.png" width="900">
</p>

<p align="center">
Figure 14: FP8 data type, Matrix A [m, k] input, GM -> L1, ND -> Nz
</p>

**(1) DataCopy**

When Matrix A is input without transpose on GM ([m, k]), call `DataCopy` instruction with `Nd2NzParams` structure parameter to transfer Matrix A from GM (ND) to L1 (Nz). According to interface constraints, if the input is FP4 data type, during the DataCopy instruction ND2Nz process, the instruction internally processes based on B8 type, and parameter configuration follows B8 type settings. When configuring `Nd2NzParams` structure, for FP4 data type, dValue takes packedK = CeilDivision(k, 2), for FP8 data type, dValue takes packedK = k; dstNzC0Stride unit is 32B, this parameter takes the aligned row count of the Nz matrix on L1. FP4 data type is shown in Figure 13, FP8 data type is shown in Figure 14.

            AscendC::Nd2NzParams nd2nzA1Params;
            nd2nzA1Params.ndNum = 1; // Number of ND matrices
            nd2nzA1Params.nValue = m; // Number of rows in source ND matrix
            nd2nzA1Params.dValue = packedK; // Number of columns in source ND matrix
            nd2nzA1Params.srcNdMatrixStride = 0; // Starting address offset between adjacent ND matrices in source operand
            nd2nzA1Params.srcDValue = packedK;  // Starting address offset between adjacent rows within the same ND matrix in source operand
            nd2nzA1Params.dstNzC0Stride = mAlignL1; // After ND to Nz conversion, starting address interval of each segment after splitting data in the same row, unit 32B
            nd2nzA1Params.dstNzNStride = 1;     // Offset in dst after row x and row x+1 of ND matrix are converted to Nz
            nd2nzA1Params.dstNzMatrixStride = 0; // Offset of starting addresses between adjacent Nz matrices in destination Nz matrix, meaningless when configured as 0
            AscendC::DataCopy(a1Local, aGM, nd2nzA1Params);

<br>

**(2) Fill Operation**

When Matrix A is input without transpose [m, k], k is in the col direction. For FP8 data type, `DataCopy` only aligns the k direction to 32B, while MX matrix multiplication requires the Mmad instruction to align the k direction to 64. At this point, directly call `AscendC::Fill` to set the tail 1 block of data in the k direction to 0, as shown in Figure 14:

            if constexpr (AscendC::IsSameType<TA, fp8_e4m3fn_t>::value || AscendC::IsSameType<TA, fp8_e5m2_t>::value) {
                // Fill Matrix A L1 data with 0 as uint16 type; when dst is in A1, Fill blockNum unit is 32B.
                const uint32_t heightAlign = CeilAlign(m, fractalShape[0]);
                auto padTensor = a1Local.template ReinterpretCast<uint16_t>();
                AscendC::InitConstValueParams<uint16_t> initConstValueParams;
                // repeatTimes indicates iteration count; iterate in row direction, covering each row after m is aligned to 16.
                initConstValueParams.repeatTimes = heightAlign;
                // blockNum indicates the number of data blocks (32B) initialized in each iteration; here only fill the tail 1 32B data in col direction each time.
                initConstValueParams.blockNum = 1;
                // initValue indicates initialization value; fill invalid data with 0 to prevent participating in Mmad computation.
                initConstValueParams.initValue = 0;
                // dstOffset locates to the end of currently transferred data in col direction, subsequent Fill fills tail 1 32B data row by row.
                uint32_t dstOffset = heightAlign * (CeilAlign(packedK, SCALE_CEIL_NUMBER) / 2);
                AscendC::Fill(padTensor[dstOffset], initConstValueParams);
            }

##### 2.1.2. Matrix A GM input as [k, m]

<p align="center">
  <img src="img/GM2L1/FP4_A_GM2L1_TRANS_KM.png" width="700">
</p>

<p align="center">
Figure 15: FP4 data type, Matrix A [k, m] input, GM -> L1, ND -> Nz
</p>

<p align="center">
  <img src="img/GM2L1/FP8_A_GM2L1_TRANS_KM.png" width="700">
</p>

<p align="center">
Figure 16: FP8 data type, Matrix A [k, m] input, GM -> L1, ND -> Nz
</p>

**(1) DataCopy**

When Matrix A is input with transpose on GM ([k, m]), call `DataCopy` instruction with `Nd2NzParams` structure parameter to transfer Matrix A from GM (ND) to L1 (Nz). According to interface constraints, if the input is FP4 data type, during the DataCopy instruction ND2Nz process, the instruction internally processes based on B8 type, and parameter configuration follows B8 type settings. When configuring `Nd2NzParams` structure, source operand shape is [k, m], dstNzC0Stride unit is 32B, this parameter takes the aligned row count of Nz matrix on L1 (that is, the aligned length in k direction alignK). FP4 data type is shown in Figure 15, FP8 data type is shown in Figure 16.

            AscendC::Nd2NzParams nd2nzA1Params;
            uint16_t aColValue = isFP4 ? CeilDivision(m, 2) : m;
            nd2nzA1Params.ndNum = 1; // Number of ND matrices
            nd2nzA1Params.nValue = k; // Number of rows in source ND matrix
            nd2nzA1Params.dValue = aColValue; // Number of columns in source ND matrix, for FP4 type 2 consecutive data are merged into 1 FP8 data for transfer
            nd2nzA1Params.srcNdMatrixStride = 0; // Starting address offset between adjacent ND matrices in source operand
            nd2nzA1Params.srcDValue = aColValue; // Starting address offset between adjacent rows within the same ND matrix in source operand
            nd2nzA1Params.dstNzC0Stride = alignK; // After ND to Nz conversion, starting address interval of each segment after splitting data in the same row, unit 32B
            nd2nzA1Params.dstNzNStride = 1;      // Offset in dst after row x and row x+1 of ND matrix are converted to Nz
            nd2nzA1Params.dstNzMatrixStride = 0; // Offset of starting addresses between adjacent Nz matrices in destination Nz matrix, meaningless when configured as 0
            AscendC::DataCopy(a1Local, aGM, nd2nzA1Params);


**(2) Fill Operation**

When Matrix A is input with transpose [k, m], k is in the row direction. `DataCopy` aligns the k direction to 16, while MX matrix multiplication requires the Mmad instruction to align the k direction to 64. You need to directly call `AscendC::Fill` to set the dirty data exceeding the original length in the k direction to 0. FP4 data type is shown in Figure 15, FP8 data type is shown in Figure 16:

            // Pad invalid data in row direction within [k, alignK] range.
            // Fill Matrix A L1 data with 0 as uint16 type; when dst is in A1, Fill blockNum and dstGap unit is 32B.
            auto padTensor = a1Local.template ReinterpretCast<uint16_t>();
            AscendC::InitConstValueParams<uint16_t> initConstValueParams;
            // repeatTimes indicates iteration count; iterate in col direction.
            initConstValueParams.repeatTimes = CeilDivision(m, FP8_C0SIZE);
            // blockNum indicates the number of data blocks (32B) initialized in each iteration; here fill the number of invalid rows in row direction tail each time.
            initConstValueParams.blockNum = alignK - k;
            // dstGap indicates the distance from previous iteration end address to next iteration start address; skip valid data in row direction.
            initConstValueParams.dstGap = k;
            // initValue indicates initialization value; fill invalid data with 0 to prevent participating in Mmad computation.
            initConstValueParams.initValue = 0;
            // Starting address locates to the first fractal in row direction that needs to be padded with 0.
            AscendC::Fill(padTensor[k * fractalShape[0]], initConstValueParams);

#### 2.2. Matrix B GM -> L1

##### 2.2.1. Matrix B GM input as [k, n]
<p align="center">
  <img src="img/GM2L1/FP4_B_GM2L1_TRANS_KN.png" width="700">
</p>

<p align="center">
Figure 17: FP4 data type, Matrix B [k, n] input, GM -> L1, ND -> Nz
</p>

<p align="center">
  <img src="img/GM2L1/FP8_B_GM2L1_TRANS_KN.png" width="700">
</p>

<p align="center">
Figure 18: FP8 data type, Matrix B [k, n] input, GM -> L1, ND -> Nz
</p>

**(1) DataCopy**

When Matrix B is input without transpose on GM ([k, n]), call `DataCopy` instruction with `Nd2NzParams` structure parameter to transfer Matrix B from GM (ND) to L1 (Nz). According to interface constraints, if the input is FP4 data type, during the DataCopy instruction ND2Nz process, the instruction internally processes based on B8 type, and parameter configuration follows B8 type settings. When configuring `Nd2NzParams` structure, source operand shape is [k, n], dstNzC0Stride takes the aligned row count of Nz matrix on L1 (that is, the aligned length in k direction alignK). FP4 data type is shown in Figure 17, FP8 data type is shown in Figure 18.

            AscendC::Nd2NzParams nd2nzB1Params;
            uint16_t bColValue = isFP4 ? CeilDivision(n, 2) : n;
            nd2nzB1Params.ndNum = 1; // Number of ND matrices
            nd2nzB1Params.nValue = k; // Number of rows in source ND matrix
            nd2nzB1Params.dValue = bColValue; // Number of columns in source ND matrix, for FP4 type 2 consecutive data are merged into 1 FP8 data for transfer
            nd2nzB1Params.srcNdMatrixStride = 0; // Starting address offset between adjacent ND matrices in source operand
            nd2nzB1Params.srcDValue = bColValue; // Starting address offset between adjacent rows within the same ND matrix in source operand
            nd2nzB1Params.dstNzC0Stride = alignK; // After ND to Nz conversion, starting address interval of each segment after splitting data in the same row, unit 32B
            nd2nzB1Params.dstNzNStride = 1; // Offset in dst after row x and row x+1 of ND matrix are converted to Nz
            nd2nzB1Params.dstNzMatrixStride = 0; // Offset of starting addresses between adjacent Nz matrices in destination Nz matrix, meaningless when configured as 0
            AscendC::DataCopy(b1Local, bGM, nd2nzB1Params);

**(2) Fill Operation**

When Matrix B is input without transpose [k, n], k is in the row direction. `DataCopy` aligns the k direction to 16, while MX matrix multiplication requires the Mmad instruction to align the k direction to 64. You need to directly call `AscendC::Fill` to set the dirty data exceeding the original length in the k direction to 0. FP4 data type is shown in Figure 17, FP8 data type is shown in Figure 18:

            // Pad invalid data in row direction within [k, alignK] range.
            // Fill Matrix B L1 data with 0 as uint16 type; when dst is in B1, Fill blockNum and dstGap unit is 32B.
            auto padTensor = b1Local.template ReinterpretCast<uint16_t>();
            AscendC::InitConstValueParams<uint16_t> initConstValueParams;
            // repeatTimes indicates iteration count; iterate in col direction.
            initConstValueParams.repeatTimes = CeilDivision(n, FP8_C0SIZE);
            // blockNum indicates the number of data blocks (32B) initialized in each iteration; here fill the number of invalid rows in row direction tail each time.
            initConstValueParams.blockNum = alignK - k;
            // dstGap indicates the distance from previous iteration end address to next iteration start address; skip valid data in row direction.
            initConstValueParams.dstGap = k;
            // initValue indicates initialization value; fill invalid data with 0 to prevent participating in Mmad computation.
            initConstValueParams.initValue = 0;
            // Starting address locates to the first fractal in row direction that needs to be padded with 0.
            AscendC::Fill(padTensor[k * fractalShape[0]], initConstValueParams);

##### 2.2.2. Matrix B GM input as [n, k]

<p align="center">
  <img src="img/GM2L1/FP4_B_GM2L1_NK.png" width="800">
</p>

<p align="center">
Figure 19: FP4 data type, Matrix B [n, k] input, GM -> L1, ND -> Nz
</p>

<p align="center">
  <img src="img/GM2L1/FP8_B_GM2L1_NK.png" width="800">
</p>

<p align="center">
Figure 20: FP8 data type, Matrix B [n, k] input, GM -> L1, ND -> Nz
</p>

**(1) DataCopy**

When Matrix B is input with transpose on GM ([n, k]), call `DataCopy` instruction with `Nd2NzParams` structure parameter to transfer Matrix B from GM (ND) to L1 (Nz). According to interface constraints, if the input is FP4 data type, during the DataCopy instruction ND2Nz process, the instruction internally processes based on B8 type, and parameter configuration follows B8 type settings. When configuring `Nd2NzParams` structure, source operand shape is [n, k], dstNzC0Stride takes the aligned row count of Nz matrix on L1 (that is, the aligned length in n direction nAlignL1). FP4 data type is shown in Figure 19, FP8 data type is shown in Figure 20.

            AscendC::Nd2NzParams nd2nzB1Params;
            nd2nzB1Params.ndNum = 1; // Number of ND matrices
            nd2nzB1Params.nValue = n; // Number of rows in source ND matrix
            nd2nzB1Params.dValue = packedK; // Number of columns in source ND matrix, due to instruction limitation, if input is FP4 data type, 2 consecutive data need to be merged into 1 FP8 data for transfer
            nd2nzB1Params.srcNdMatrixStride = 0; // Starting address offset between adjacent ND matrices in source operand
            nd2nzB1Params.srcDValue = packedK; // Starting address offset between adjacent rows within the same ND matrix in source operand
            nd2nzB1Params.dstNzC0Stride = nAlignL1; // After ND to Nz conversion, starting address interval of each segment after splitting data in the same row, unit 32B
            nd2nzB1Params.dstNzNStride = 1; // Offset in dst after row x and row x+1 of ND matrix are converted to Nz
            nd2nzB1Params.dstNzMatrixStride = 0; // Offset of starting addresses between adjacent Nz matrices in destination Nz matrix, meaningless when configured as 0
            AscendC::DataCopy(b1Local, bGM, nd2nzB1Params);

**(2) Fill Operation**

When Matrix B is input with transpose [n, k], k is in the col direction. For FP8 data type, `DataCopy` only aligns the k direction to 32B, while MX matrix multiplication requires the Mmad instruction to align the k direction to 64. You need to directly call `AscendC::Fill` to set the tail 1 block of data in the k direction to 0, as shown in Figure 20:

            if constexpr (AscendC::IsSameType<TB, fp8_e4m3fn_t>::value || AscendC::IsSameType<TB, fp8_e5m2_t>::value) {
                // Fill Matrix B L1 data with 0 as uint16 type; when dst is in B1, Fill blockNum unit is 32B.
                const uint32_t heightAlign = CeilAlign(n, fractalShape[0]);
                auto padTensor = b1Local.template ReinterpretCast<uint16_t>();
                AscendC::InitConstValueParams<uint16_t> initConstValueParams;
                // repeatTimes indicates iteration count; iterate in row direction, covering each row after n is aligned to 16.
                initConstValueParams.repeatTimes = heightAlign;
                // blockNum indicates the number of data blocks (32B) initialized in each iteration; here only fill the tail 1 32B data in col direction each time.
                initConstValueParams.blockNum = 1;
                // initValue indicates initialization value; fill invalid data with 0 to prevent participating in Mmad computation.
                initConstValueParams.initValue = 0;
                // dstOffset locates to the end of currently transferred data in col direction, subsequent Fill fills tail 1 32B data row by row.
                uint32_t dstOffset = heightAlign * (CeilAlign(packedK, SCALE_CEIL_NUMBER) / 2);
                AscendC::Fill(padTensor[dstOffset], initConstValueParams);
            }

#### 2.3. Matrix scaleA GM -> L1
Matrix scaleA uses fp8_e8m0_t data type. When arranged by fp8_e8m0_t actual data type, scaleA has Zz format on L1. Due to hardware constraints, scale matrices require 2-byte continuity in the K direction. During `DataCopy`, fp8_e8m0_t needs to be transferred as B16 (half) view (every 2 fp8_e8m0_t elements correspond to 1 half element), at this point L1 shows Nz layout of B16 data type. Transfer method depends on isAtranspose value:

**(1) When isAtranspose=false, Matrix scaleA GM input is [m, scaleK], use `Dn2NzParams` structure parameter (B16 view)**

<p align="center">
  <img src="img/GM2L1/scaleA_GM2L1_MK.png" width="900">
</p>

<p align="center">
Figure 21: Matrix scaleA [m, scaleK] input, GM -> L1, ND -> Zz
</p>

Matrix scaleA GM shape is [m, scaleK]. Call `DataCopy` instruction with `Dn2NzParams` structure parameter, transfer as B16 view, as shown in Figure 21:

            AscendC::GlobalTensor<half> scaleAGMB16;
            scaleAGMB16.SetGlobalBuffer((__gm__ half *)(scaleAGM.GetPhyAddr()), m * scaleK / 2);
            auto scaleA1LocalB16 = scaleA1Local.ReinterpretCast<half>();

            // When input without transpose, scaleA GM shape is [m, scaleK], use Dn2NzParams to transfer as B16 view
            AscendC::Dn2NzParams dn2nzParams;
            dn2nzParams.dnNum = 1; // Number of DN matrices in source operand
            dn2nzParams.dValue = m; // Number of rows in source DN matrix
            dn2nzParams.nValue = scaleK / 2; // Number of columns in source DN matrix, after B16 view 2 fp8_e8m0_t are merged into 1 half
            dn2nzParams.srcDnMatrixStride = 0; // Starting address offset between adjacent DN matrices in source operand
            dn2nzParams.srcDValue = scaleK / 2; // Starting address offset between adjacent rows within the same DN matrix in source operand
            dn2nzParams.dstNzC0Stride = scaleK / 2; // After DN to Nz conversion, starting address interval of each segment after splitting data in the same row, unit 32B
            dn2nzParams.dstNzNStride = 1; // Offset in dst after row x and row x+1 of DN matrix are converted to Nz
            dn2nzParams.dstNzMatrixStride = 0; // Starting address offset between adjacent Nz matrices
            AscendC::DataCopy(scaleA1LocalB16, scaleAGMB16, dn2nzParams);

**(2) When isAtranspose=true, Matrix scaleA GM input is [scaleK, m, 2], use `Nd2NzParams` structure parameter (B16 view)**

<p align="center">
  <img src="img/GM2L1/scaleA_GM2L1_KM.png" width="1000">
</p>

<p align="center">
Figure 22: Matrix scaleA [scaleK, m, 2] input, GM -> L1, ND -> Zz
</p>

Matrix scaleA GM shape is [scaleK, m, 2]. Call `DataCopy` instruction with `Nd2NzParams` structure parameter, transfer as B16 view, as shown in Figure 22:

            AscendC::GlobalTensor<half> scaleAGMB16;
            scaleAGMB16.SetGlobalBuffer((__gm__ half *)(scaleAGM.GetPhyAddr()), m * scaleK / 2);
            auto scaleA1LocalB16 = scaleA1Local.ReinterpretCast<half>();

            // When input with transpose, scaleA GM shape is [scaleK, m, 2], use Nd2NzParams to transfer as B16 view
            AscendC::Nd2NzParams nd2nzParams;
            nd2nzParams.ndNum = 1; // Number of ND matrices in source operand
            nd2nzParams.nValue = scaleK / 2; // Number of rows in source ND matrix, after B16 view 2 fp8 scale are merged into 1 half
            nd2nzParams.dValue = m; // Number of columns in source ND matrix
            nd2nzParams.srcDValue = m; // Starting address offset between adjacent rows within the same ND matrix in source operand
            nd2nzParams.dstNzC0Stride = scaleK / 2; // After ND to Nz conversion, starting address interval of each segment after splitting data in the same row, unit 32B
            nd2nzParams.dstNzNStride = 1; // Offset in dst after row x and row x+1 of ND matrix are converted to Nz
            nd2nzParams.dstNzMatrixStride = 0; // Starting address offset between adjacent Nz matrices
            AscendC::DataCopy(scaleB1LocalB16, scaleBGMB16, nd2nzParams);


**(2) When isBtranspose=true, Matrix scaleB GM input is [n, scaleK], use `Dn2NzParams` structure parameter (B16 view)**

<p align="center">
  <img src="img/GM2L1/scaleB_GM2L1_NK.png" width="1000">
</p>

<p align="center">
Figure 24: Matrix scaleB [n, scaleK] input, GM -> L1, ND -> Nn
</p>

Matrix scaleB GM shape is [n, scaleK]. Call `DataCopy` instruction with `Dn2NzParams` structure parameter, transfer as B16 view, as shown in Figure 24:

            AscendC::GlobalTensor<half> scaleBGMB16;
            scaleBGMB16.SetGlobalBuffer((__gm__ half *)(scaleBGM.GetPhyAddr()), n * scaleK / 2);
            auto scaleB1LocalB16 = scaleB1Local.ReinterpretCast<half>();

            // When input with transpose, scaleB GM shape is [n, scaleK], use Dn2NzParams to transfer as B16 view
            AscendC::Dn2NzParams dn2nzParams;
            dn2nzParams.dnNum = 1; // Number of DN matrices in source operand
            dn2nzParams.dValue = n; // Number of rows in source DN matrix
            dn2nzParams.nValue = scaleK / 2; // Number of columns in source DN matrix, after B16 view 2 fp8 scale are merged into 1 half
            dn2nzParams.srcDnMatrixStride = 0; // Starting address offset between adjacent DN matrices in source operand
            dn2nzParams.srcDValue = scaleK / 2; // Starting address offset between adjacent rows within the same DN matrix in source operand
            dn2nzParams.dstNzC0Stride = scaleK / 2; // After DN to Nz conversion, starting address interval of each segment after splitting data in the same row, unit 32B
            dn2nzParams.dstNzNStride = 1; // Offset in dst after row x and row x+1 of DN matrix are converted to Nz
            dn2nzParams.dstNzMatrixStride = 0; // Starting address offset between adjacent Nz matrices
            AscendC::DataCopy(scaleB1LocalB16, scaleBGMB16, dn2nzParams);
### 3. L1 to L0 (`LoadData`)
This section describes how to call the `LoadData` instruction when transferring A/B matrices from L1 to L0A/L0B and scaleA/scaleB matrices from L1 to L0A_MX/L0B_MX, and how to complete data transfer and format conversion through `LoadData2DParamsV2` and `LoadData2DMxParams` structure parameters.

#### `LoadData2DParamsV2` Structure Parameter Description
`LoadData2DParamsV2` structure parameter controls A/B matrix data transfer from L1 to L0A/L0B (transpose can be performed during this process), including:

- **sid**: Source matrix identifier, default is 0
- **mStartPosition**: Starting position in row direction of source matrix, unit is 16 elements
- **kStartPosition**: Starting position in col direction of source matrix, unit is 32B
- **mStep**: Transfer length in row direction of source matrix, unit is 16 elements
- **kStep**: Transfer length in col direction of source matrix, unit is 32B
- **srcStride**: Starting address interval between adjacent fractals in col direction of source matrix, unit is 512B
- **dstStride**: Starting address interval between adjacent fractals in col direction of destination matrix, unit is 512B
- **ifTranspose**: Whether to enable transpose function, transpose each fractal matrix, default is false
Note: A/B matrix data fractal size is 512B

#### `LoadData2DMxParams` Structure Parameter Description
`LoadData2DMxParams` structure parameter controls scale matrix data transfer from L1 to L0A_MX/L0B_MX (pure transfer, no layout format change), including:

- **xStartPosition**: Starting position in row direction of source matrix, unit is 1 32B fractal
- **yStartPosition**: Starting position in col direction of source matrix, unit is 32B
- **xStep**: Transfer length in row direction of source matrix, unit is 1 32B fractal
- **yStep**: Transfer length in col direction of source matrix, unit is 32B
- **srcStride**: Starting address interval between adjacent fractals (16*2) in row direction of source matrix, unit is 32B
- **dstStride**: Starting address interval between adjacent fractals (16*2) in row direction of destination matrix, unit is 32B
Note: Scale matrix data fractal size is 16*2*1=32B

Calling `LoadData` instruction once with `LoadData2DParamsV2` and `LoadData2DMxParams` structure parameters can simultaneously complete the transfer of Matrix A to L0A and corresponding scale matrix to L0A_MX. L0A_MX Buffer and L0A addresses have a fixed proportional relationship. The `LoadData` instruction automatically derives based on L0A address, users do not need to configure. Matrix B and scaleB are similar:

            AscendC::LoadData(a2Local, a1Local, scaleA1Local, loadDataParams, loadMxDataParams);
            AscendC::LoadData(b2Local, b1Local, scaleB1Local, loadDataParams, loadMxDataParams);

#### 3.1. Matrix A L1 -> L0A, Matrix scaleA L1 -> L0A_MX

##### 3.1.1. Matrix A L1 -> L0A without transpose (Scenario 1/3)

<p align="center">
  <img src="img/L12L0/FP4_A_L12L0_MK.png" width="1000">
</p>

<p align="center">
Figure 25: FP4 data type, Matrix A [m, k] input, L1 -> L0A without transpose, loadDataParams.ifTranspose = false
</p>

<p align="center">
  <img src="img/L12L0/FP8_A_L12L0_MK.png" width="1000">
</p>

<p align="center">
Figure 26: FP8 data type, Matrix A [m, k] input, L1 -> L0A without transpose, loadDataParams.ifTranspose = false
</p>

<p align="center">
  <img src="img/L12L0/scaleA_l12l0_KM.png" width="500">
</p>

<p align="center">
Figure 27: Matrix scaleA L1 -> L0A_MX
</p>

When Matrix A is input without transpose [m, k], L1 -> L0A does not require transpose, loadDataParams.ifTranspose = false. Call `LoadData` once with `LoadData2DParamsV2` and `LoadData2DMxParams` structure parameters to simultaneously transfer Matrix A to L0A (as shown in Figure 25 and 26) and Matrix scaleA to L0A_MX (as shown in Figure 27):

            AscendC::LoadData2DParamsV2 loadDataParams;
            loadDataParams.sid = 0;
            // Start transfer from row direction fractal 0 and col direction 32B block 0 of Matrix A L1 source operand
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            // Matrix A input without transpose [m, k], L1 -> L0A does not require transpose
            // mStep/kStep represent the number of fractals transferred in row direction and number of 32B blocks transferred in col direction
            loadDataParams.mStep = CeilDivision(mAlignL1, fractalShape[0]);
            loadDataParams.kStep = CeilDivision(kaAlignL1, fractalShape[1]);
            // srcStride/dstStride represent the starting address interval between adjacent fractals in col direction of source/destination matrix, unit 512B
            loadDataParams.srcStride = CeilDivision(mAlignL1, fractalShape[0]);
            loadDataParams.dstStride = CeilDivision(mAlignL0, fractalShape[0]);
            loadDataParams.ifTranspose = false;

            AscendC::LoadData2DMxParams loadMxDataParams;
            // scaleA synchronously transfers from row direction fractal 0 and col direction 32B block 0 of L1 source operand
            loadMxDataParams.xStartPosition = 0;
            loadMxDataParams.yStartPosition = 0;
            // xStep/yStep configure scaleA row/col direction transfer length; stride is configured as interval between adjacent fractals in row direction
            loadMxDataParams.xStep = CeilDivision(scaleMAlignL1, fractalShape[0]);
            loadMxDataParams.yStep = CeilDivision(packedK, SCALE_CEIL_NUMBER);
            loadMxDataParams.srcStride = scaleK;
            loadMxDataParams.dstStride = CeilDivision(packedK, SCALE_CEIL_NUMBER);

            AscendC::LoadData(a2Local, a1Local, scaleA1Local, loadDataParams, loadMxDataParams);

Note: `LoadData2DParamsV2` controls Matrix A transfer from Nz layout on L1 to Nz layout on L0A. mStep represents row direction transfer length, kStep represents col direction transfer length. srcStride represents the starting address interval between adjacent fractals in col direction on L1, dstStride represents the starting address interval between adjacent fractals in col direction on L0A. `LoadData2DMxParams` controls Matrix scaleA transfer from L1 to L0A_MX. xStep corresponds to row direction transfer length, yStep corresponds to col direction transfer length.

##### 3.1.2. Matrix A L1 -> L0A with transpose, single call (Scenario 2/4)

<p align="center">
  <img src="img/L12L0/FP4_A_L12L0_TRANS_KM.png" width="900">
</p>

<p align="center">
Figure 28: FP4 data type, Matrix A [k, m] input, L1 -> L0A with transpose, loadDataParams.ifTranspose = true, single LoadData call
</p>

<p align="center">
  <img src="img/L12L0/FP8_A_L12L0_TRANS_KM.png" width="900">
</p>

<p align="center">
Figure 29: FP8 data type, Matrix A [k, m] input, L1 -> L0A with transpose, loadDataParams.ifTranspose = true, single LoadData call
</p>

When Matrix A is input with transpose [k, m], L1 -> L0A requires transpose. Call `LoadData` once with `LoadData2DParamsV2` and `LoadData2DMxParams` structure parameters to simultaneously transfer Matrix A to L0A (as shown in Figure 28 and 29) and Matrix scaleA to L0A_MX (as shown in Figure 27). Small fractal transpose and large fractal layout format change occur, but more than 1 fractal of redundant data is transferred in Matrix A m direction:

            AscendC::LoadData2DParamsV2 loadDataParams;
            loadDataParams.sid = 0;
            // Start transfer from row direction fractal 0 and col direction 32B block 0 of Matrix A L1 source operand
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            // Matrix A input with transpose [k, m], L1 -> L0A requires transpose, small fractal transpose and large fractal layout format change will occur
            // During transpose transfer, Matrix A shape on L1 is [kaAlignL1, mAlignL1], row direction corresponds to logical k dimension, col direction corresponds to logical m dimension
            loadDataParams.mStep = CeilDivision(kaAlignL1, fractalShape[0]);
            loadDataParams.kStep = CeilDivision(mAlignL1, fractalShape[1]);
            // srcStride/dstStride represent the starting address interval between adjacent fractals in col direction of source/destination matrix, unit 512B
            loadDataParams.srcStride = CeilDivision(kaAlignL1, fractalShape[0]);
            loadDataParams.dstStride = CeilDivision(mAlignL0, fractalShape[0]);
            loadDataParams.ifTranspose = true;

            AscendC::LoadData2DMxParams loadMxDataParams;
            // scaleA synchronously transfers from row direction fractal 0 and col direction 32B block 0 of L1 source operand
            loadMxDataParams.xStartPosition = 0;
            loadMxDataParams.yStartPosition = 0;
            // xStep/yStep configure scaleA row/col direction transfer length; stride is configured as interval between adjacent fractals in row direction
            loadMxDataParams.xStep = CeilDivision(scaleMAlignL1, fractalShape[0]);
            loadMxDataParams.yStep = CeilDivision(packedK, SCALE_CEIL_NUMBER);
            loadMxDataParams.srcStride = scaleK;
            loadMxDataParams.dstStride = CeilDivision(packedK, SCALE_CEIL_NUMBER);

            AscendC::LoadData(a2Local, a1Local, scaleA1Local, loadDataParams, loadMxDataParams);

Note: When Matrix A is input with transpose [k, m], `LoadData2DParamsV2` ifTranspose is set to true, indicating transpose of each small fractal from L1 -> L0A. Since transpose transfer causes more than 1 fractal of redundant data to be transferred in m direction, subsequent `Mmad` instruction needs special handling (see Section 4). At this point, Matrix A shape on L1 is [kaAlignL1, mAlignL1], row direction corresponds to logical k dimension, col direction corresponds to logical m dimension, so mStep configures row direction transfer length, kStep configures col direction transfer length.

##### 3.1.3. Matrix A L1 -> L0A with transpose, for-loop call (Scenario 5/6)

<p align="center">
  <img src="img/L12L0/FP4_A_L12L0_for_TRANS_KM.png" width="900">
</p>

<p align="center">
Figure 30: FP4 data type, Matrix A [k, m] input, L1 -> L0A with transpose, loadDataParams.ifTranspose = true, for-loop LoadData call
</p>

<p align="center">
  <img src="img/L12L0/FP8_A_L12L0_for_TRANS_KM.png" width="900">
</p>

<p align="center">
Figure 31: FP8 data type, Matrix A [k, m] input, L1 -> L0A with transpose, loadDataParams.ifTranspose = true, for-loop LoadData call
</p>

<p align="center">
  <img src="img/L12L0/scaleA_l12l0_KM.png" width="500">
</p>

<p align="center">
Figure 32: Matrix scaleA L1 -> L0A_MX, for-loop LoadData call
</p>

When Matrix A is input with transpose [k, m], if a single `LoadData` call would transfer more than 1 fractal of dirty data in m direction, a for-loop approach can be used to avoid writing redundant dirty data fractals to L0A.
- Matrix A: Use for-loop in k direction. Each iteration transfers 2 fractals in k-axis direction * CeilDivision(mAlignL0, fractalShape[1]) fractals in m-axis direction from L1. Loop L0ALoopNum times. Each for-loop skips tail dirty data fractals in m direction on L0A, with no more than 1 fractal of redundant data transferred in m direction, as shown in the red boxes in Figure 30 and 31.
- Matrix scaleA: Matrix scaleA is transferred entirely to L0A_MX in the first for-loop. Subsequent for-loops skip transfer by configuring loadMxDataParams.xStep = 0 and loadMxDataParams.yStep = 0 parameters.

            uint16_t mStepAlign = isFP4 ? FP4_M_STEP_ALIGN : FP8_M_STEP_ALIGN;
            AscendC::LoadData2DParamsV2 loadDataParams;
            loadDataParams.sid = 0;
            // kStartPosition is fixed to 0, each loop uses mStartPosition to select the current fractal in Matrix A source row direction
            loadDataParams.kStartPosition = 0;
            // Transpose transfer requires mStep to be aligned by data type: FP4 is 4 fractals, FP8 is 2 fractals
            loadDataParams.mStep = mStepAlign;
            // kStep corresponds to the number of 32B blocks in Matrix A source col direction
            loadDataParams.kStep = CeilDivision(mAlignL0, fractalShape[1]);
            // srcStride/dstStride represent the starting address interval between adjacent fractals in col direction of source/destination matrix, unit 512B
            loadDataParams.srcStride = CeilDivision(kaAlignL1, fractalShape[0]);
            loadDataParams.dstStride = CeilDivision(mAlignL0, fractalShape[0]);
            loadDataParams.ifTranspose = true;

            AscendC::LoadData2DMxParams loadMxDataParams;
            // scaleA row direction starting fractal and col direction starting 32B block are fixed to 0
            loadMxDataParams.xStartPosition = 0;
            loadMxDataParams.yStartPosition = 0;
            // srcStride/dstStride represent the starting address interval between adjacent fractals in row direction of scaleA source/destination matrix, unit 32B
            loadMxDataParams.srcStride = scaleK;
            loadMxDataParams.dstStride = CeilDivision(packedK, SCALE_CEIL_NUMBER);

            uint32_t dstOffset = 0;
            uint16_t L0ALoopNum = CeilDivision(kaAlignL0, fractalShape[0] * fractalNum);
            for (uint16_t loopIdx = 0; loopIdx < L0ALoopNum; ++loopIdx) {
                // mStartPosition increments, Matrix A updates starting address in m direction each transfer, Matrix scaleA completes transfer in first for-loop, subsequent for-loops do not transfer
                loadDataParams.mStartPosition = mStepAlign * loopIdx;
                if (loopIdx != 0) {
                    loadMxDataParams.xStep = 0;
                    loadMxDataParams.yStep = 0;
                } else {
                    loadMxDataParams.xStep = mStepAlign;
                    loadMxDataParams.yStep = CeilDivision(packedK, SCALE_CEIL_NUMBER);
                }
                AscendC::LoadData(a2Local[dstOffset], a1Local, scaleA1Local, loadDataParams, loadMxDataParams);
                dstOffset += CeilAlign(mAlignL0, fractalShape[0]) * fractalShape[1];
            }

Note: For FP4 data type, mStepAlign=4; for FP8 data type, mStepAlign=2. For-loop count L0ALoopNum = CeilDivision(kaAlignL0, fractalShape[0] * fractalNum). Each loop transfers mStepAlign row direction fractals of Matrix A. mStartPosition increments with loopIdx, indicating the row direction starting position offset of Matrix A in each loop. Matrix scaleA completes transfer in the first for-loop, and subsequent for-loops skip transfer by setting xStep and yStep to 0. dstOffset records the destination address offset on L0A for each loop.

#### 3.2. Matrix B L1 -> L0B, Matrix scaleB L1 -> L0B_MX

Matrix B/scaleB transfer method is similar to Matrix A/scaleA, but Matrix B has Zn layout format on L0B, and Matrix scaleB has Nn layout format on L0B_MX. Call `LoadData` once with `LoadData2DParamsV2` and `LoadData2DMxParams` structure parameters to simultaneously transfer Matrix B to L0B and Matrix scaleB to L0B_MX.

##### 3.2.1. Matrix B L1 -> L0B without transpose (Scenario 1/3)

<p align="center">
  <img src="img/L12L0/FP4_B_L12L0_NK.png" width="900">
</p>

<p align="center">
Figure 33: FP4 data type, Matrix B [n, k] input, L1 -> L0B without transpose, loadDataParams.ifTranspose = false
</p>

<p align="center">
  <img src="img/L12L0/FP8_B_L12L0_NK.png" width="900">
</p>

<p align="center">
Figure 34: FP8 data type, Matrix B [n, k] input, L1 -> L0B without transpose, loadDataParams.ifTranspose = false
</p>

<p align="center">
  <img src="img/L12L0/scaleB_l12l0_KN.png" width="700">
</p>

<p align="center">
Figure 35: Matrix scaleB L1 -> L0B_MX
</p>

When Matrix B is input with transpose [n, k], L1 -> L0B does not require transpose. Call `LoadData` once with `LoadData2DParamsV2` and `LoadData2DMxParams` structure parameters to simultaneously transfer Matrix B to L0B (as shown in Figure 33 and 34) and Matrix scaleB to L0B_MX (as shown in Figure 35):

            AscendC::LoadData2DParamsV2 loadDataParams;
            loadDataParams.sid = 0;
            // Start transfer from row direction fractal 0 and col direction 32B block 0 of Matrix B L1 source operand
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            // Matrix B input with transpose [n, k], L1->L0B does not require transpose
            // mStep/kStep represent the number of fractals transferred in row direction and number of 32B blocks transferred in col direction
            loadDataParams.mStep = CeilDivision(nAlignL1, fractalShape[0]);
            loadDataParams.kStep = CeilDivision(kbAlignL1, fractalShape[1]);
            // srcStride/dstStride represent the starting address interval between adjacent fractals in col direction of source/destination matrix, unit 512B
            loadDataParams.srcStride = CeilDivision(nAlignL1, fractalShape[0]);
            loadDataParams.dstStride = CeilDivision(nAlignL0, fractalShape[0]);
            loadDataParams.ifTranspose = false;

            AscendC::LoadData2DMxParams loadMxDataParams;
            // scaleB synchronously transfers from row direction fractal 0 and col direction 32B block 0 of L1 source operand
            loadMxDataParams.xStartPosition = 0;
            loadMxDataParams.yStartPosition = 0;
            // xStep/yStep configure scaleB row/col direction transfer length; stride is configured as interval between adjacent fractals in row direction
            loadMxDataParams.xStep = CeilDivision(scaleNAlignL1, fractalShape[0]);
            loadMxDataParams.yStep = CeilDivision(packedK, SCALE_CEIL_NUMBER);
            loadMxDataParams.srcStride = scaleK;
            loadMxDataParams.dstStride = CeilDivision(packedK, SCALE_CEIL_NUMBER);

            AscendC::LoadData(b2Local, b1Local, scaleB1Local, loadDataParams, loadMxDataParams);

Note: When Matrix B is input with transpose [n, k], `LoadData2DParamsV2` ifTranspose=false. At this point, Matrix B shape on L1 is [nAlignL1, kbAlignL1], row direction corresponds to logical n dimension, col direction corresponds to logical k dimension, so mStep configures row direction transfer length, kStep configures col direction transfer length. `LoadData2DMxParams` parameter configuration is similar to Matrix A. xStep configures scaleB matrix row direction transfer length, yStep configures scaleB matrix col direction transfer length.

##### 3.2.2. Matrix B L1 -> L0B with transpose (Scenario 2/4/5/6)


<p align="center">
  <img src="img/L12L0/FP4_B_L12L0_TRANS_KN.png" width="1000">
</p>

<p align="center">
Figure 36: FP4 data type, Matrix B [k, n] input, L1 -> L0B with transpose, loadDataParams.ifTranspose = true
</p>

<p align="center">
  <img src="img/L12L0/FP8_B_L12L0_TRANS_KN.png" width="1000">
</p>

<p align="center">
Figure 37: FP8 data type, Matrix B [k, n] input, L1 -> L0B with transpose, loadDataParams.ifTranspose = true
</p>

When Matrix B is input without transpose [k, n], L1 -> L0B requires transpose. Call `LoadData` once with `LoadData2DParamsV2` and `LoadData2DMxParams` structure parameters to simultaneously transfer Matrix B to L0B (as shown in Figure 36 and 37) and Matrix scaleB to L0B_MX (as shown in Figure 35). Small fractal transpose and large fractal layout format change occur:

            AscendC::LoadData2DParamsV2 loadDataParams;
            loadDataParams.sid = 0;
            // Start transfer from row direction fractal 0 and col direction 32B block 0 of Matrix B L1 source operand
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            // Matrix B input without transpose [k, n], L1->L0B requires transpose, small fractal transpose and large fractal layout format change occur
            // During transpose transfer, Matrix B shape on L1 is [kbAlignL1, nAlignL1], row direction corresponds to logical k dimension, col direction corresponds to logical n dimension
            loadDataParams.mStep = CeilDivision(kbAlignL1, fractalShape[0]);
            loadDataParams.kStep = CeilDivision(nAlignL1, fractalShape[1]);
            // srcStride/dstStride represent the starting address interval between adjacent fractals in col direction of source/destination matrix, unit 512B
            loadDataParams.srcStride = CeilDivision(kbAlignL1, fractalShape[0]);
            loadDataParams.dstStride = CeilDivision(nAlignL0, fractalShape[0]);
            loadDataParams.ifTranspose = true;

            AscendC::LoadData2DMxParams loadMxDataParams;
            // scaleB synchronously transfers from row direction fractal 0 and col direction 32B block 0 of L1 source operand
            loadMxDataParams.xStartPosition = 0;
            loadMxDataParams.yStartPosition = 0;
            // xStep/yStep configure scaleB row/col direction transfer length; stride is configured as interval between adjacent fractals in row direction
            loadMxDataParams.xStep = CeilDivision(scaleNAlignL1, fractalShape[0]);
            loadMxDataParams.yStep = CeilDivision(packedK, SCALE_CEIL_NUMBER);
            loadMxDataParams.srcStride = scaleK;
            loadMxDataParams.dstStride = CeilDivision(packedK, SCALE_CEIL_NUMBER);

            AscendC::LoadData(b2Local, b1Local, scaleB1Local, loadDataParams, loadMxDataParams);

Note: When Matrix B is input without transpose [k, n], `LoadData2DParamsV2` ifTranspose=true, small fractal transpose and large fractal layout format change occur. At this point, Matrix B shape on L1 is [kbAlignL1, nAlignL1], row direction corresponds to logical k dimension, col direction corresponds to logical n dimension, so mStep configures row direction transfer length, kStep configures col direction transfer length. When Matrix B is input without transpose [k, n], no more than 1 fractal of redundant data is transferred in n direction. `LoadData2DMxParams` parameter configuration is similar to Matrix A. xStep configures scaleB matrix row direction transfer length, yStep configures scaleB matrix col direction transfer length.

### 4.Matrix Multiplication (`Mmad`)
This section describes how to configure the MmadParams structure members of the `Mmad` instruction.

MX matrix multiplication formula is C = (scaleA ⊗ A) * (scaleB ⊗ B). The `Mmad` instruction automatically completes the broadcast multiplication of left/right matrices with corresponding scale matrices. Every 32 elements in the K direction share one quantization factor.

Note that similar to the load_data_l12l0 sample, when the actual alignment requirements of Matrix A and Matrix B on each axis in L0A/L0B are inconsistent with the default alignment requirements of the `Mmad` instruction, it may cause incorrect reading of fractals filled entirely with invalid data while ignoring fractals containing valid data when continuously reading fractals.

For scenarios 2 and 4 (Matrix A input with transpose [k, m], single `LoadData` call), since more than 1 fractal of redundant data is transferred in the m direction during Matrix A transpose transfer process, if you still set mmadParams.m = m, the cube unit will incorrectly read fractals of invalid data while fractals of valid data are not read. At this point, you can set mmadParams.m = CeilAlign(m, fractalShape[0] * fractalNum) to let this fractal participate in computation, and skip the results from invalid fractals participating in computation during transfer out.

            AscendC::MmadParams mmadParams;
            if constexpr (scenarioNum == 2 || scenarioNum == 4) {
                // mmad defaults m-axis aligned to 16, but since A transpose process aligns m-axis to fractalShape[0]*fractalNum,
                // fractals filled entirely with invalid data are added. Set m aligned to fractalShape[0]*fractalNum,
                // let this fractal participate in computation, and skip results from invalid fractals participating in computation during transfer out
                mmadParams.m = CeilAlign(m, fractalShape[0] * fractalNum);
            } else {
                mmadParams.m = m;
            }
            mmadParams.n = n;
            mmadParams.k = alignK;
            mmadParams.cmatrixInitVal = true;
            AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams);

Note: mmadParams.k takes alignK = CeilAlign(k, 64) = 128, not the original k=70. This is because MX matrix multiplication requires the Mmad instruction to align the k direction to 64. mmadParams.cmatrixInitVal = true means initializing Matrix C.

For scenarios 5 and 6 (Matrix A input with transpose [k, m], for-loop `LoadData` call), since the for-loop approach avoids the case of transferring more than 1 fractal of redundant data, no more than 1 fractal of redundant data is transferred in the m direction. `Mmad` computation amount is CeilAlign(m, fractalShape[0]) * CeilAlign(n, fractalShape[0] * fractalNum), at this point mmadParams.m = m is sufficient.

### 5.L0C to GM (`Fixpipe`)
This section describes how to configure the FixpipeParamsArch3510 structure members of the `Fixpipe` instruction. FixpipeParamsArch3510 is a Fixpipe parameter structure specific to dav-3510 architecture. CO2Layout set to ROW_MAJOR indicates output is row-major ND format.

            AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams;
            fixpipeParams.nSize = n;
            fixpipeParams.mSize = m;
            if constexpr (scenarioNum == 2 || scenarioNum == 4){
                // Scenario 2/4: Matrix A input with transpose [k, m] single call, more than 1 fractal of redundant data transferred in m direction
                fixpipeParams.srcStride = CeilAlign(m, fractalShape[0] * fractalNum);
            } else {
                fixpipeParams.srcStride = CeilAlign(m, fractalShape[0]);
            }
            fixpipeParams.dstStride = n;
            AscendC::Fixpipe<U, U, AscendC::CFG_ROW_MAJOR>(cGM, c1Local, fixpipeParams);

Note: fixpipeParams.srcStride unit is elements. It means the starting address offset of adjacent Z layouts in source Nz matrix. For scenarios 2/4, srcStride needs to be consistent with `Mmad` m alignment value, taking CeilAlign(m, fractalShape[0] * fractalNum); other scenarios take CeilAlign(m, fractalShape[0]). fixpipeParams.dstStride = n represents the starting address offset of adjacent rows in destination ND matrix. fixpipeParams.nSize = n and fixpipeParams.mSize = m ensure only valid data is transferred out, skipping results from invalid data participating in computation.

## Build and Run
Execute the following steps in the sample root directory to build and run the sample.
- Configure environment variables
  Select the corresponding command to configure environment variables based on the [installation method](../../../../../../docs/quick_start.md#prepare&install) of the CANN development kit on the current environment.
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

- Sample execution
  ```bash
  SCENARIO=1
  mkdir -p build && cd build;      # Create and enter build directory
  cmake .. -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=$SCENARIO;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO   # Generate test input data
  ./demo                           # Execute the compiled executable program to run the sample
  python3 ../scripts/verify_result.py -scenarioNum=$SCENARIO output/output.bin output/golden.bin   # Verify output correctness to confirm algorithm logic correctness
  ```

  When using CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake .. -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=$SCENARIO;make -j;   # CPU debug mode
  cmake .. -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=$SCENARIO;make -j;   # NPU simulation mode
  ```

  > **Note:** Before switching build mode, you need to clean cmake cache. Execute `rm CMakeCache.txt` in the build directory and re-run cmake.

- Build option description

  | Parameter | Description | Available Values | Default Value |
  |------|------|--------|--------|
  | CMAKE_ASC_RUN_MODE | Run mode | npu, cpu, sim | npu |
  | CMAKE_ASC_ARCHITECTURES | NPU hardware architecture | dav-3510 | dav-3510 |
  | SCENARIO_NUM | Scenario number | 1-6 | 1 |

- Execution result

  The execution result shows that precision comparison is successful.
  ```bash
  test pass!
  ```

#### 2.4. Matrix scaleB GM -> L1
Matrix scaleB transfer method is similar to scaleA. When arranged by fp8_e8m0_t actual data type, scaleB has Nn format on L1. Due to hardware constraints, scale matrices require 2-byte continuity in the k direction, and `DataCopy` transfer needs to be done as B16 view. At this point L1 shows Nz layout of B16 data type. Transfer method depends on isBtranspose value:

**(1) When isBtranspose=false, Matrix scaleB GM input is [scaleK, n, 2], use `Nd2NzParams` structure parameter (B16 view)**

<p align="center">
  <img src="img/GM2L1/scaleB_GM2L1_KN.png" width="1000">
</p>

<p align="center">
Figure 23: Matrix scaleB [scaleK, n, 2] input, GM -> L1, ND -> Nn
</p>

Matrix scaleB GM shape is [scaleK, n, 2]. Call `DataCopy` instruction with `Nd2NzParams` structure parameter, transfer as B16 view, as shown in Figure 23:

            AscendC::GlobalTensor<half> scaleBGMB16;
            scaleBGMB16.SetGlobalBuffer((__gm__ half *)(scaleBGM.GetPhyAddr()), n * scaleK / 2);
            auto scaleB1LocalB16 = scaleB1Local.ReinterpretCast<half>();

            // When input without transpose, scaleB GM shape is [scaleK, n, 2], use Nd2NzParams to transfer as B16 view
            AscendC::Nd2NzParams nd2nzParams;
            nd2nzParams.ndNum = 1; // Number of ND matrices in source operand
            nd2nzParams.nValue = scaleK / 2; // Number of rows in source ND matrix, after B16 view 2 fp8 scale are merged into 1 half
            nd2nzParams.dValue = n; // Number of columns in source ND matrix
            nd2nzParams.srcDValue = n; // Starting address offset between adjacent rows within the same ND matrix in source operand
            nd2nzParams.dstNzC0Stride = scaleK / 2; // After ND to Nz conversion, starting address interval of each segment after splitting data in the same row, unit 32B