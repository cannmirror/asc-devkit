# load_data_2dv2_l12l0 Example

## Overview

This example introduces the method of using the `LoadData` interface with LoadData2DParamsV2 structure parameters (referred to as `Load2Dv2` interface in this example) to move matrix data from L1 to L0A/L0B in 7 matrix multiplication scenarios with ND format input, B8 / B16 / B32 input data types (specifically using int8_t / half / float as examples), and combinations of left/right matrix transposed and non-transposed.
- Uses `LoadData2DParamsV2` parameter structure
- L0A data layout is **Nz format**
- L0B data layout is **Zn format**
- Supports transpose function, controlled by `ifTranspose` parameter
- Supports int8_t, half, float three data types

## Supported Products
- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── load_data_2dv2_l12l0
│   ├── img                         // Figures
│   ├── scripts
│   │   ├── gen_data.py             // Input data and golden data generation script
│   │   └── verify_result.py        // Verification script for checking output data against golden data
│   ├── CMakeLists.txt              // Build configuration file
│   ├── data_utils.h                // Data read/write functions
│   └── load_data_2dv2_l12l0.asc         // Ascend C example implementation & invocation example
```

## Example Description

### Example Function

The overall flow of the example is as follows:

```
GM(ND) -> L1(Nz) -> L0A(Nz)/L0B(Zn) -> L0C(Nz) -> GM(ND)
        │         │                  │          │       
     DataCopy  Load2Dv2            Mmad        Fixpipe 
```

**Step Details**:

1. **GM → L1**: Use `DataCopy` to implement ND to Nz format conversion
2. **L1 → L0A/L0B**: Use `Load2Dv2` interface for movement, control transpose through `ifTranspose`
3. **Matrix Multiplication**: Use `Mmad` interface to execute matrix multiplication
4. **L0C → GM**: Use `Fixpipe` interface to move out results


All scenarios are based on the same matrix multiplication specification: [m, n, k] = [40, 50, 70], kernel function name is "KernelLoadDataL12L0Load2Dv2". The parameter scenarioNum represents the 7 scenarios described above. The meanings corresponding to different values of scenarioNum are shown in [Table 1](#table1) below:<br>

<a name="table1"></a>
<table border="2">
<caption style="font-weight: normal;">
    <span style="font-weight: bold; font-size: 1.2em;">Table 1: Meanings of Different scenarioNum Values</span>
  <tr>
    <td ><span style="font-weight: bold;">scenarioNum</span></td>
    <td><span style="font-weight: bold;">Input Data Type</span></td>
    <td><span style="font-weight: bold;">Output Data Type</span></td>
    <td><span style="font-weight: bold;">isAtranspose</span></td>
    <td><span style="font-weight: bold;">isBtranspose</span></td>
    <td><span style="font-weight: bold;">Extra Load/Compute</span></td>
    <td><span style="font-weight: bold;">L1->L0 Invocation Method</span></td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">1</span></td>
    <td rowspan="2" >int8_t</td>
    <td rowspan="2" >int32_t</td>
    <td>false</td>
    <td>true</td>
    <td>No</td>
    <td>Load2Dv2</td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">2</span></td>
    <td>true</td>
    <td>false</td>
    <td>Yes</td>
    <td>Load2Dv2</td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">3</span></td>
    <td rowspan="2" >half</td>
    <td rowspan="2" >float</td>
    <td>false</td>
    <td>true</td>
    <td>No</td>
    <td>Load2Dv2</td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">4</span></td>
    <td>true</td>
    <td>false</td>
    <td>No</td>
    <td>Load2Dv2</td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">5</span></td>
    <td rowspan="2" >float</td>
    <td rowspan="2" >float</td>
    <td>false</td>
    <td>true</td>
    <td>No</td>
    <td>Load2Dv2</td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">6</span></td>
    <td>true</td>
    <td>false</td>
    <td>No</td>
    <td>Load2Dv2</td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">7</span></td>
    <td rowspan="1" >int8_t</td>
    <td rowspan="1" >int32_t</td>
    <td>true</td>
    <td>false</td>
    <td>No</td>
    <td>for loop + Load2Dv2</td>
  </tr>
</table>

Note: When scenarioNum is 7, L1->L0A movement uses for loop + Load2Dv2, all other scenarios call Load2dV2 once.


**Scenario 1: Input int8_t data type, isAtranspose=False, isBtranspose=True**
- Input A [40, 70], int8_t type, ND format; B [50, 70], int8_t type, ND format;
- Output C [40, 50], int32_t type, ND format;
- Implementation: Use one Load2DV2 to move A matrix from L1 to L0A, use one Load2DV2 to move B matrix from L1 to L0B;
- Description: L1->L0A does not need transpose, complete A matrix L1 -> L0A movement by configuring mStep, kStep, srcStride and other parameters; similarly, L1->L0B does not need transpose, B matrix completes L1 -> L0B movement and large fractal layout format change by configuring Step, kStep, srcStride and other parameters.

**Scenario 2: Input int8_t data type, isAtranspose=True, isBtranspose=False**
- Input A [70, 40], int8_t type, ND format; B [70, 50], int8_t type, ND format;
- Output C [40, 50], int32_t type, ND format;
- Implementation: Use one Load2DV2 to move A matrix from L1 to L0A, use one Load2DV2 to move B matrix from L1 to L0B;
- Description: A matrix calls one Load2DV2 instruction, by configuring mStep, kStep, srcStride, and configuring ifTranspose as true, completes A matrix movement from L1 to L0A, the movement process accompanies transpose; B matrix calls one Load2DV2 instruction, configures mStep, kStep, srcStride, and configures ifTranspose as true, completes B matrix movement from L1 to L0B, the movement process accompanies transpose. This scenario has extra load/compute dirty data fractals in m direction, when moving L0C to GM, through fixpipeParam.mSize = m to ensure results from invalid data participating in computation are not moved out.

**Scenario 3: Input half data type, isAtranspose=False, isBtranspose=True**
- Input A [40, 70], half type, ND format; B [50, 70], half type, ND format;
- Output C [40, 50], float type, ND format;
- Implementation: Use one Load2DV2 to move A matrix from L1 to L0A, use one Load2DV2 to move B matrix from L1 to L0B;
- Description: L1->L0A does not need transpose, complete A matrix L1 -> L0A movement by configuring mStep, kStep, srcStride and other parameters; similarly, L1->L0B does not need transpose, B matrix completes L1 -> L0B movement and large fractal layout format change by configuring Step, kStep, srcStride and other parameters.

**Scenario 4: Input half data type, isAtranspose=True, isBtranspose=False**
- Input A [70, 40], half type, ND format; B [70, 50], half type, ND format;
- Output C [40, 50], float type, ND format;
- Implementation: Use one Load2DV2 to move A matrix from L1 to L0A, use one Load2DV2 to move B matrix from L1 to L0B;
- Description: A matrix calls one Load2DV2 instruction, by configuring mStep, kStep, srcStride, and configuring ifTranspose as true, completes A matrix movement from L1 to L0A, the movement process accompanies transpose; B matrix calls one Load2DV2 instruction, configures mStep, kStep, srcStride, and configures ifTranspose as true, completes B matrix movement from L1 to L0B, the movement process accompanies transpose.

**Scenario 5: Input float data type, isAtranspose=False, isBtranspose=True**
- Input A [40, 70], float type, ND format; B [50, 70], float type, ND format;
- Output C [40, 50], float type, ND format;
- Implementation: Use one Load2DV2 to move A matrix from L1 to L0A, use one Load2DV2 to move B matrix from L1 to L0B;
- Description: L1->L0A does not need transpose, complete A matrix L1 -> L0A movement by configuring mStep, kStep, srcStride and other parameters; similarly, L1->L0B does not need transpose, B matrix completes L1 -> L0B movement and large fractal layout format change by configuring Step, kStep, srcStride and other parameters.

**Scenario 6: Input float data type, isAtranspose=True, isBtranspose=False**
- Input A [70, 40], float type, ND format; B [70, 50], float type, ND format;
- Output C [40, 50], float type, ND format;
- Implementation: Use one Load2DV2 to move A matrix from L1 to L0A, use one Load2DV2 to move B matrix from L1 to L0B;
- Description: A matrix calls one Load2DV2 instruction, by configuring mStep, kStep, srcStride, and configuring ifTranspose as true, completes A matrix movement from L1 to L0A, the movement process accompanies transpose; B matrix calls one Load2DV2 instruction, configures mStep, kStep, srcStride, and configures ifTranspose as true, completes B matrix movement from L1 to L0B, the movement process accompanies transpose.

**Scenario 7: Input int8_t data type, isAtranspose=True, isBtranspose=False**
- Input A [70, 40], int8_t type, ND format; B [70, 50], int8_t type, ND format;
- Output C [40, 50], int32_t type, ND format;
- Implementation: Use for loop + Load2DV2 to move A matrix from L1 to L0A, use one Load2DV2 to move B matrix from L1 to L0B;
- Description: A matrix loops along k direction, calls Load2DV2 instruction multiple times, each time moving 2 fractals in k axis direction * CeilDivision(k, fractalShape[1]) fractals in m axis direction from L1, by configuring mStep, kStep, srcStride, and configuring ifTranspose as true, completes A matrix movement from L1 to L0A, the movement process accompanies transpose; B matrix calls one Load2DV2 instruction, configures mStep, kStep, srcStride, and configures ifTranspose as true, completes B matrix movement from L1 to L0B, the movement process accompanies transpose. This scenario through for loop with dstStride parameter skips extra read dirty data fractals in m direction when writing to L0A, making matrix computation m direction without extra dirty data fractals participating in computation.

To facilitate description, definitions are given here for commonly used concepts:
 
(1) fractalShape: The shape of a small fractal is [16, 32 / sizeof(T)], where T represents the input data type. The fractal-related information for data types involved in this example is shown in [Table 2](#table2).

(2) fractalSize: The number of elements contained in 1 small fractal, see [Table 2](#table2) for details.


(3) fractalNum: When transpose is needed from L1 -> L0A/L0B, the Load2DV2 interface internally performs small block matrix transpose. For B8 and B32 data types, fractalShape are [16,32] and [16,8] respectively, both require two consecutive small fractals to be combined into one block and then transposed. Therefore, this parameter represents how many small fractals a block contains, see [Table 2](#table2) for details.


<a name="table2"></a>
<table border="2">
<caption style="font-weight: normal;">
    <span style="font-weight: bold; font-size: 1.2em;">Table 2: Fractal-Related Information for Different Data Types</span></caption>
  <tr>
    <td></td>
    <td align="center"><span style="font-weight: bold;">fractalShape</span></td>
    <td align="center"><span style="font-weight: bold;">fractalSize</span></td>
    <td align="center"><span style="font-weight: bold;">fractalNum</span></td>
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


(4) CeilAlign: Round-up alignment operation. For example, when m=30, CeilAlign(30, 16)=32, meaning aligning m axis to 16, the aligned m axis length is 32.



      __aicore__ inline uint16_t CeilAlign(uint16_t size, uint16_t alignValue) {
          return (size + alignValue - 1) / alignValue * alignValue;
      }

(5) CeilDivision: Round-up division, generally used to calculate the loop count after round-up alignment.

(6) mAlignValue: Align m axis to mAlignValue. For example, mAlignValue=32 means m axis is aligned to 32, similarly nAlignValuem means aligning axis to nAlignValue, kaAlignValue means aligning A matrix k axis to kaAlignValue, kbAlignValue means aligning B matrix k axis to kbAlignValue.

(7) mAlignL0 and mAlignL1: The aligned value of m axis when A matrix is on L1 and L0A respectively. Similarly, there are nAlignL0, nAlignL1, kaAlignL0, kaAlignL1, kbAlignL0, kbAlignL1.


Additionally, the alignment requirements for A and B matrices on L1 and L0 in row and col directions are also different. The alignment requirements for the 6 scenarios corresponding to scenarioNum 1-6 in [Table 2](#table2) (layout format on L1 is Nz) are summarized as shown in [Table 3](#table3) and [Table 4](#table4):

<a name="table3"></a>
<table border="2">
<caption style="font-weight: normal;">
    <span style="font-weight: bold; font-size: 1.2em;">Table 3: Alignment Requirements for A and B Matrices on Each Axis on L1 (L1 Layout Format is Nz)</span></caption>
  <tr>
    <td></td>
    <td align="center"><span style="font-weight: bold;">B8 (fractalNum=2)</span></td>
    <td align="center"><span style="font-weight: bold;">B16 (fractalNum=1)</span></td>
    <td align="center"><span style="font-weight: bold;">B32 (fractalNum=2)</span></td>
  </tr>
  <tr>
    <td rowspan="2" align="center"><span style="font-weight: bold;">A Matrix Non-Transposed Input [m, k]</span></td>
    <td colspan="3" align="center">mAlignValue = fractalShape[0]</td>
  </tr>
  <tr>
    <td colspan="3" align="center" >kaAlignValue = fractalShape[1]</td>
  </tr>
  <tr>
    <td rowspan="2" align="center"><span style="font-weight: bold;">A Matrix Transposed Input [k, m]</span></td>
    <td colspan="2" align="center">kaAlignValue = fractalShape[0] * fractalNum</td>
    <td colspan="1" align="center">kaAlignValue = fractalShape[0]</td>
  </tr>
  <tr>
    <td colspan="2" align="center" >mAlignValue = fractalShape[1]</td>
    <td colspan="1" align="center" >mAlignValue = fractalShape[1] * fractalNum</td>
  </tr>
    <tr>
    <td rowspan="2" align="center"><span style="font-weight: bold;">B Matrix Non-Transposed Input [k, n]</span></td>
    <td colspan="2" align="center">kbAlignValue = fractalShape[0] * fractalNum</td>
    <td colspan="1" align="center">kbAlignValue = fractalShape[0]</td>
  </tr>
  <tr>
    <td colspan="2" align="center" >nAlignValue = fractalShape[1]</td>
    <td colspan="1" align="center" >nAlignValue = fractalShape[1] * fractalNum</td>
  </tr>
 <tr>
    <td rowspan="2" align="center"><span style="font-weight: bold;">B Matrix Transposed Input [n, k]</span></td>
    <td colspan="3" align="center">nAlignValue = fractalShape[0]</td>
  </tr>
  <tr>
    <td colspan="3" align="center" >kbAlignValue = fractalShape[1]</td>
  </tr>
</table>


<a name="table4"></a>
<table border="2">
<caption style="font-weight: normal;">
    <span style="font-weight: bold; font-size: 1.2em;">Table 4: Alignment Requirements for A and B Matrices on Each Axis on L0</span></caption>
  <tr>
    <td></td>
    <td align="center"><span style="font-weight: bold;">B8 (fractalNum=2)</span></td>
    <td align="center"><span style="font-weight: bold;">B16 (fractalNum=1)</span></td>
    <td align="center"><span style="font-weight: bold;">B32 (fractalNum=2)</span></td>
  </tr>
  <tr>
    <td rowspan="2" align="center"><span style="font-weight: bold;">A Matrix Non-Transposed Input [m, k], L1->L0A No Transpose Needed</span></td>
    <td colspan="3" align="center">mAlignValue = fractalShape[0]</td>
  </tr>
  <tr>
    <td colspan="3" align="center" >kaAlignValue = fractalShape[1]</td>
  </tr>
  <tr>
    <td rowspan="2" align="center"><span style="font-weight: bold;">A Matrix Transposed Input [k, m], L1->L0A Transpose Needed</span></td>
    <td colspan="2" align="center">kaAlignValue = fractalShape[1]</td>
    <td >kaAlignValue = fractalShape[1] * fractalNum</td>
  </tr>
  <tr>
    <td colspan="2" align="center" >mAlignValue = fractalShape[0] * fractalNum</td>
    <td align="center" >mAlignValue = fractalShape[0]</td>
  </tr>
    <tr>
    <td rowspan="2" align="center"><span style="font-weight: bold;">B Matrix Non-Transposed Input [k, n], L1->L0B Transpose Needed</span></td>
    <td colspan="2" align="center">kbAlignValue = fractalShape[1]</td>
      <td align="center">kbAlignValue = fractalShape[1] * fractalNum</td>
  </tr>
  <tr>
    <td colspan="2" align="center">nAlignValue = fractalShape[0] * fractalNum</td>
    <td align="center" >nAlignValue = fractalShape[0]</td>
  </tr>
 <tr>
    <td rowspan="2" align="center"><span style="font-weight: bold;">B Matrix Transposed Input [n, k], L1->L0B No Transpose Needed</span></td>
    <td colspan="3" align="center">nAlignValue = fractalShape[0]</td>
  </tr>
  <tr>
    <td colspan="3" align="center" >kbAlignValue = fractalShape[1]</td>
  </tr>
</table>


Specifically, when scenarioNum=7, A matrix uses for loop + Load2DV2 to implement L1->L0A movement, L0A only writes data aligned to valid data fractals. The alignment requirements for A and B matrices on L1 and L0 in height and width directions are shown in [Table 5](#table5) and [Table 6](#table6):
<a name="table5"></a>
<table border="2">
<caption style="font-weight: normal;">
    <span style="font-weight: bold; font-size: 1.2em;">Table 5: scenarioNum=7, Alignment Requirements for A and B Matrices on Each Axis on L1</span></caption>
  <tr>
    <td align="center" ></td>
    <td align="center" ><span style="font-weight: bold;">int8_t (fractalNum=2)</span></td>
  </tr>
   <tr>
    <td rowspan="2"><span style="font-weight: bold;">A Matrix Transposed Input [k, m]</span></td>
    <td align="center" >kaAlignValue = fractalShape[0] * fractalNum</td>
  </tr>
    <tr>
    <td align="center" >mAlignValue = fractalShape[1]</td>
  </tr>
   <tr>
    <td rowspan="2"><span style="font-weight: bold;">B Matrix Non-Transposed Input [k, n]</span></td>
    <td align="center" >kbAlignValue = fractalShape[0] * fractalNum</td>
  </tr>
    <tr>
    <td align="center" >nAlignValue = fractalShape[1]</td>
  </tr>
</table>

<a name="table6"></a>
<table border="2">
<caption style="font-weight: normal;">
    <span style="font-weight: bold; font-size: 1.2em;">Table 6: scenarioNum=7, Alignment Requirements for A and B Matrices on Each Axis on L0</span></caption>
  <tr>
    <td align="center" ></td>
    <td align="center" ><span style="font-weight: bold;">int8_t (fractalNum=2)</span></td>
  </tr>
   <tr>
    <td rowspan="2"><span style="font-weight: bold;">A Matrix Transposed Input [k, m], L1->L0A Transpose Needed</span></td>
    <td align="center" >mAlignValue = fractalShape[0]</td>
  </tr>
    <tr>
    <td align="center" >kaAlignValue = fractalShape[1]</td>
  </tr>
   <tr>
    <td rowspan="2"><span style="font-weight: bold;">B Matrix Non-Transposed Input [k, n], L1->L0B Transpose Needed</span></td>
    <td align="center" >kbAlignValue = fractalShape[1]</td>
  </tr>
    <tr>
    <td align="center" >nAlignValue = fractalShape[0] * fractalNum</td>
  </tr>
</table>

### Example Implementation
The data layout format of A/B matrices on L1 is Nz, on L0A and L0B are Nz and Zn respectively. In the L1->L0 process, call the Load2DV2 interface to complete data movement and format transformation.
#### A Matrix L1->L0A Non-Transposed

When L1 -> L0A non-transposed data movement, there is no large fractal and small fractal layout format change. In this scenario, B8 / B16 / B32 these three data types use basically the same Load2DV2 interface parameter configuration, only fractalShape differs, see [Table 2](#table2). Figures are shown using int8_t as example.<br>


<p align="center">
  <img src="img/B8_A_l1_l0A_Load2dv2.png">
</p>

<p align="center">
Figure 1: int8_t data type, L1 -> L0A non-transposed, Load2DV2 data layout diagram
</p>


As shown in Figure 1, starting from the starting address of A matrix on L1, m is row, configure mstep parameter, k is col, configure kStep parameter, call one instruction to complete the movement of entire A matrix from L1->L0A.

```cpp
mAlignL1 = CeilAlign(m, fractalShape[0]);
kaAlignL1 = CeilAlign(k, fractalShape[1]);
mAlignL0 = CeilAlign(m, fractalShape[0]);
kaAlignL0 = CeilAlign(k, fractalShape[1]);
AscendC::LoadData2DParamsV2 loadDataParams;
loadDataParams.mStartPosition = 0;
loadDataParams.kStartPosition = 0;
loadDataParams.mStep = CeilDivision(mAlignL1, fractalShape[0]); // row
loadDataParams.kStep = CeilDivision(kaAlignL1, fractalShape[1]); // col
loadDataParams.srcStride = CeilDivision(mAlignL1, fractalShape[0]);
loadDataParams.dstStride = CeilDivision(mAlignL0, fractalShape[0]);
loadDataParams.ifTranspose = false;
loadDataParams.sid = 0;
AscendC::LoadData(a2Local, a1Local, loadDataParams);
```

#### A Matrix L1->L0A Transposed

When L1->L0A, there is large fractal layout format transformation, and small fractals need transpose. Using int8_t, half, float as examples, the transpose scenario diagram explanations for B8 / B16 / B32 these three different data types are introduced in sections by data type.

##### B8 Input Data Type
B8 input data type fractal is 16 * 32. When L1->L0 transpose, it will transpose by combining 2 fractals of 16 * 32 in row direction into 1 square of 32 * 32. In this example, using int8_t input data type as example, m is 40. If calling one Load2DV2 instruction to complete L1->L0A movement + transpose, then when writing to L0A, m direction will write 1 extra fractal of invalid data. Matrix computation Mmad will also compute 1 extra fractal of invalid data in m direction. When Fixpipe moves out, only move out valid data; if calling for loop + Load2DV2 instruction to complete L1->L0A movement + transpose, when writing can skip invalid data fractals in m direction. These two scenarios are introduced separately below.<br>
**Call One Load2DV2**


The diagram for calling one Load2DV2 to complete L1->L0A movement + transpose is as follows:


<p align="center">
  <img src="img/B8_A_l1_l0A_trans_load2dv2.png">
</p>

<p align="center">
Figure 2: int8_t data type, L1 -> L0A transposed, call one Load2DV2 data layout diagram
</p>


In this example m=40, as shown in Figure 2, on L1 mAlignL1 = CeilAlign(m, fractalShape[1])=64. mAlignL1 - m = 24 > 16. When calling one Load2DV2 to complete L1->L0A movement, m direction will move 1 invalid fractal, as shown in the red box. In this scenario, matrix computation Mmad will also compute 1 extra fractal of invalid data in m direction, i.e., mmadParams.m = CeilAlign(m, fractalShape[0] * fractalNum). When Fixpipe moves out, only move out valid data (fixpipeParams.mSize = m).


```cpp
kaAlignL1 = CeilAlign(k, fractalShape[0] * fractalNum);
mAlignL1 = CeilAlign(m, fractalShape[1]);
mAlignL0 = CeilAlign(m, fractalShape[0] * fractalNum);
kaAlignL0 = CeilAlign(k, fractalShape[1]);
AscendC::LoadData2DParamsV2 loadDataParams;
loadDataParams.mStep = CeilDivision(kaAlignL1, fractalShape[0]);
loadDataParams.kStep = CeilDivision(mAlignL1, fractalShape[1]);
loadDataParams.srcStride = CeilDivision(kaAlignL1, fractalShape[0]);
loadDataParams.dstStride = CeilDivision(mAlignL0, fractalShape[0]);
loadDataParams.ifTranspose = true;
loadDataParams.sid = 0;
AscendC::LoadData(a2Local, a1Local, loadDataParams);
```


**for loop + Load2Dv2**


The diagram for for loop calling multiple Load2DV2 to complete L1->L0A movement + transpose is as follows:

<p align="center">
  <img src="img/B8_A_l1_l0A_trans_for_load2dv2.png">
</p>

<p align="center">
Figure 3: int8_t data type, L1 -> L0A transposed, for loop calling multiple Load2DV2 data layout diagram
</p>


In this example m=40, as shown in Figure 3, on L1 mAlignL1 = CeilAlign(m, fractalShape[1])=64. mAlignL1 - m = 24 > 16. At this time, loop along k direction, call Load2DV2 instruction multiple times, each time moving 2 fractals in k axis direction * CeilDivision(k, fractalShape[1]) fractals in m axis direction from L1, as shown in the red box. dstStride is configured as valid data in m direction aligned to small fractal fractalShape[0], then when writing to L0A, skip extra read dirty data fractals in m direction during transpose, making matrix computation m direction without extra dirty data fractals participating in computation.


```cpp
kaAlignL1 = CeilAlign(k, fractalShape[0] * fractalNum);
mAlignL1 = CeilAlign(m, fractalShape[1]);
mAlignL0 = CeilAlign(m, fractalShape[0]);
kaAlignL0 = CeilAlign(k, fractalShape[1]);
// Input is int8 type, A matrix [k,m] transposed input, L1->L0A needs transpose
// for loop calling Load2DV2, loop along k axis, each loop moves 2 fractals in L1's k direction, skip m direction tail dirty data fractals on L0A, m direction extra moved data not exceeding 1 fractal
uint16_t L0ALoopNum = CeilDivision(kaAlignL0, fractalShape[0] * fractalNum);
loadDataParams.mStep = INT8_M_STEP_ALIGN;
loadDataParams.kStep = CeilDivision(mAlignL0, fractalShape[1]);
loadDataParams.srcStride = CeilDivision(kaAlignL1, fractalShape[0]);
loadDataParams.dstStride = CeilDivision(mAlignL0, fractalShape[0]);
loadDataParams.ifTranspose = true;
uint32_t dstOffset = 0;
for (uint16_t loopIdx = 0; loopIdx < L0ALoopNum; ++loopIdx) {
    loadDataParams.mStartPosition = INT8_M_STEP_ALIGN * loopIdx;
    AscendC::LoadData(a2Local[dstOffset], a1Local, loadDataParams);
    dstOffset += CeilAlign(mAlignL0, fractalShape[0]) * fractalShape[1];
}
```

##### B16 Input Data Type
B16 input data type fractal is 16 * 16. One fractal is one square. When L1->L0 transpose, it will transpose by small fractal. Calling one Load2DV2 can complete L1->L0A data movement and transpose. In this example using half as example, the diagram for calling one Load2DV2 to complete L1->L0A movement + transpose is as follows:


<p align="center">
  <img src="img/B16_A_l1_l0A_trans_load2dv2.png">
</p>

<p align="center">
Figure 4: half data type, L1 -> L0A transposed, call one Load2DV2 data layout diagram
</p>

As shown in Figure 4, starting from the starting address of A matrix on L1, k is row, configure mstep parameter, m is col, configure kStep parameter, with ifTranspose=true, call one instruction to complete the movement + transpose of entire A matrix from L1->L0A.

```cpp
kaAlignL1 = CeilAlign(k, fractalShape[0] * fractalNum);
mAlignL1 = CeilAlign(m, fractalShape[1]);
mAlignL0 = CeilAlign(m, fractalShape[0] * fractalNum);
kaAlignL0 = CeilAlign(k, fractalShape[1]);
AscendC::LoadData2DParamsV2 loadDataParams;
loadDataParams.mStep = CeilDivision(kaAlignL1, fractalShape[0]);
loadDataParams.kStep = CeilDivision(mAlignL1, fractalShape[1]);
loadDataParams.srcStride = CeilDivision(kaAlignL1, fractalShape[0]);
loadDataParams.dstStride = CeilDivision(mAlignL0, fractalShape[0]);
loadDataParams.ifTranspose = true;
loadDataParams.sid = 0;
AscendC::LoadData(a2Local, a1Local, loadDataParams);
```


##### B32 Input Data Type
B32 input data type fractal is 16 * 8. When L1->L0 transpose, it will transpose by combining 2 fractals of 16 * 8 in col direction into 1 square of 16 * 16. In this example using float data type as example, the diagram for calling one Load2DV2 to complete L1->L0A movement + transpose is as follows:

<p align="center">
  <img src="img/B32_A_l1_l0A_trans_load2dv2.png">
</p>

<p align="center">
Figure 5: float data type, L1 -> L0A transposed, call one Load2DV2 data layout diagram
</p>


In this example m is 40. Since transpose will combine 2 fractals in col direction into a square for transpose (kStep must be a multiple of 2), col direction (m direction) on L1 will have 1 extra invalid fractal data to meet instruction requirements. Calling one Load2DV2 instruction to complete L1->L0A movement + transpose, when writing to L0A, k direction will write 1 extra fractal of invalid data. Since L0A layout format is Nz, extra invalid fractal data in k direction is at the tail. When performing matrix computation Mmad, configure k direction mmadParams.k = k to only let valid data participate in matrix computation.


```cpp
kaAlignL1 = CeilAlign(k, fractalShape[0]);
mAlignL1 = CeilAlign(m, fractalShape[1] * fractalNum);
mAlignL0 = CeilAlign(m, fractalShape[0]);
kaAlignL0 = CeilAlign(k, fractalShape[1] * fractalNum);
AscendC::LoadData2DParamsV2 loadDataParams;
loadDataParams.mStep = CeilDivision(kaAlignL1, fractalShape[0]);
loadDataParams.kStep = CeilDivision(mAlignL1, fractalShape[1]);
loadDataParams.srcStride = CeilDivision(kaAlignL1, fractalShape[0]);
loadDataParams.dstStride = CeilDivision(mAlignL0, fractalShape[0]);
loadDataParams.ifTranspose = true;
loadDataParams.sid = 0;
AscendC::LoadData(a2Local, a1Local, loadDataParams);
```


#### B Matrix L1->L0B Non-Transposed

When L1 -> L0B non-transposed data movement, there is only large fractal format change. In this scenario, B8 / B16 / B32 these three data types use basically the same Load2DV2 interface parameter configuration, only fractalShape differs, see [Table 2](#table2). Figures are shown using float as example.<br>


<p align="center">
  <img src="img/B32_B_l1_l0B_load2dv2.png">
</p>

<p align="center">
Figure 6: float data type, L1 -> L0B non-transposed, call one Load2DV2 data layout diagram
</p>


As shown in Figure 6, starting from the starting address of B matrix on L1, n is row, configure mstep parameter, k is col, configure kStep parameter, call one instruction to complete the movement of entire B matrix from L1->L0B.

```cpp
nAlignL1 = CeilAlign(n, fractalShape[0]);
kbAlignL1 = CeilAlign(k, fractalShape[1]);
kbAlignL0 = CeilAlign(k, fractalShape[1]);
nAlignL0 = CeilAlign(n, fractalShape[0]);
AscendC::LoadData2DParamsV2 loadDataParams;
loadDataParams.mStartPosition = 0;
loadDataParams.kStartPosition = 0;
loadDataParams.mStep = CeilDivision(nAlignL1, fractalShape[0]);
loadDataParams.kStep = CeilDivision(kbAlignL1, fractalShape[1]);
loadDataParams.srcStride = CeilDivision(nAlignL1, fractalShape[0]);
loadDataParams.dstStride = CeilDivision(nAlignL0, fractalShape[0]);
loadDataParams.ifTranspose = false;
loadDataParams.sid = 0;
AscendC::LoadData(b2Local, b1Local, loadDataParams);
```

#### B Matrix L1->L0B Transposed

When L1->L0B, there is large fractal layout format transformation, and small fractals need transpose. Using int8_t, half, float as examples, the transpose scenario diagram explanations for B8 / B16 / B32 these three different data types are introduced in sections by data type.

##### B8 Input Data Type
B8 input data type fractal is 16 * 32. When L1->L0 transpose, it will transpose by combining 2 fractals of 16 * 32 in row direction into 1 square of 32 * 32. In this example using int8_t as example, calling one Load2DV2 instruction to complete L1->L0B movement + transpose, as shown in the diagram below.


<p align="center">
  <img src="img/B8_B_l1_l0B_trans_load2dv2.png">
</p>

<p align="center">
Figure 7: int8_t data type, L1 -> L0B transposed, call one Load2DV2 data layout diagram
</p>


In this example k=70. Since transpose will combine 2 fractals in row direction into a square for transpose (instruction parameter mStep must be a multiple of 2), row direction (k direction) on L1 will have 1 extra invalid fractal data to meet instruction requirements. Calling one Load2DV2 instruction to complete L1->L0B movement + transpose.

```cpp
kbAlignL1 = CeilAlign(k, fractalShape[0] * fractalNum);
nAlignL1 = CeilAlign(n, fractalShape[1]);
kbAlignL0 = CeilAlign(k, fractalShape[1]);
nAlignL0 = CeilAlign(n, fractalShape[0] * fractalNum);
AscendC::LoadData2DParamsV2 loadDataParams;
loadDataParams.mStep = CeilDivision(kbAlignL1, fractalShape[0]);
loadDataParams.kStep = CeilDivision(nAlignL1, fractalShape[1]);
loadDataParams.srcStride = CeilDivision(kbAlignL1, fractalShape[0]);
loadDataParams.dstStride = CeilDivision(nAlignL0, fractalShape[0]);
loadDataParams.ifTranspose = true;
AscendC::LoadData(b2Local, b1Local, loadDataParams);
```

##### B16 Input Data Type

B16 input data type fractal is 16*16. One fractal is one square. When L1->L0 transpose, it will transpose by small fractal. Calling one Load2DV2 can complete L1->L0B data movement and transpose. In this example using half as example, the diagram for calling one Load2DV2 to complete L1->L0B movement + transpose is as follows:


<p align="center">
  <img src="img/B16_B_l1_l0B_trans_load2dv2.png">
</p>

<p align="center">
Figure 8: half data type, L1 -> L0B transposed, call one Load2DV2 data layout diagram
</p>


As shown in Figure 8, starting from the starting address of B matrix on L1, k is row, configure mstep parameter, N is col, configure kStep parameter, with ifTranspose=true, call one instruction to complete the movement + transpose of entire B matrix from L1->L0B.


```cpp
kbAlignL1 = CeilAlign(k, fractalShape[0] * fractalNum);
nAlignL1 = CeilAlign(n, fractalShape[1]);
kbAlignL0 = CeilAlign(k, fractalShape[1]);
nAlignL0 = CeilAlign(n, fractalShape[0] * fractalNum);
AscendC::LoadData2DParamsV2 loadDataParams;
loadDataParams.mStep = CeilDivision(kbAlignL1, fractalShape[0]);
loadDataParams.kStep = CeilDivision(nAlignL1, fractalShape[1]);
loadDataParams.srcStride = CeilDivision(kbAlignL1, fractalShape[0]);
loadDataParams.dstStride = CeilDivision(nAlignL0, fractalShape[0]);
loadDataParams.ifTranspose = true;
AscendC::LoadData(b2Local, b1Local, loadDataParams);
```

##### B32 Input Data Type
B32 input data type fractal is 16 * 8. When L1->L0 transpose, it will transpose by combining 2 fractals of 16 * 8 in col direction into 1 square of 16 * 16. In this example using float as example, calling one Load2DV2 instruction to complete L1->L0B movement + transpose, as shown in the diagram below.


<p align="center">
  <img src="img/B32_B_l1_l0B_trans_load2dv2.png">
</p>

<p align="center">
Figure 9: float data type, L1 -> L0B transposed, call one Load2DV2 data layout diagram
</p>


In this example n=50. Since transpose will combine 2 fractals in col direction into a square for transpose (instruction parameter kStep must be a multiple of 2), col direction (n direction) on L1 will have 1 extra invalid fractal data to meet instruction requirements. Calling one Load2DV2 instruction to complete L1->L0B movement + transpose, when writing to L0B, k direction will write 1 extra fractal of invalid data. Since L0B layout format is Zn, extra invalid fractal data in k direction is at the tail. When performing matrix computation Mmad, configure mmadParams.k = k to only let valid data participate in matrix computation.


```cpp
kbAlignL1 = CeilAlign(k, fractalShape[0]);
nAlignL1 = CeilAlign(n, fractalShape[1] * fractalNum);
kbAlignL0 = CeilAlign(k, fractalShape[1] * fractalNum);
nAlignL0 = CeilAlign(n, fractalShape[0]);
AscendC::LoadData2DParamsV2 loadDataParams;
loadDataParams.mStep = CeilDivision(kbAlignL1, fractalShape[0]);
loadDataParams.kStep = CeilDivision(nAlignL1, fractalShape[1]);
loadDataParams.srcStride = CeilDivision(kbAlignL1, fractalShape[0]);
loadDataParams.dstStride = CeilDivision(nAlignL0, fractalShape[0]);
loadDataParams.ifTranspose = true;
AscendC::LoadData(b2Local, b1Local, loadDataParams);
```


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
  SCENARIO=1
  mkdir -p build && cd build;
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=$SCENARIO ..;make -j;
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO
  ./demo
  python3 ../scripts/verify_result.py -scenarioNum=$SCENARIO output/output.bin output/golden.bin
  ```

  When using CPU debugging or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=$SCENARIO ..;make -j;   # CPU debugging mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=$SCENARIO ..;make -j;   # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clean the cmake cache. Execute `rm CMakeCache.txt` in the build directory and then re-run cmake.

- Build Option Description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debugging, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU architecture: Ascend 950PR/Ascend 950DT |
  | `SCENARIO_NUM` | `1`-`7` | Scenario number: different data types and transpose combinations |

- Execution Result

  The execution result is as follows, indicating precision comparison succeeded.
  ```bash
  test pass!
  ```