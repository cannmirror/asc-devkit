# Gelu Performance Tuning Example

## Overview

This example uses Gelu computation to introduce RegBase vector performance tuning methods, demonstrating performance gains after enabling VF fusion and loop unrolling.

**Optimization Paths**:
- Case 0: Gelu without VF fusion enabled (baseline)
- Case 1: RegBase API and VF fusion enabled
- Case 2: RegBase API, VF fusion, and loop unrolling optimization enabled

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── gelu_high_performance
│   ├── scripts
│   │   ├── gen_data.py         // Input data and golden data generation script
│   │   └── verify_result.py    // Golden value comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read and write functions
│   ├── gelu.asc                // Ascend C example implementation (containing 2 optimization cases)
```

## Example Description

**Example Function**:

The Gelu approximation formula is calculated as:

$$
GELU(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \cdot \left(x + 0.044715 \cdot x^3\right)\right)\right) \tag{1}
$$

The tanh calculation formula is:

$$
\tanh(u) = \frac{e^{2u} - 1}{e^{2u} + 1} \tag{2}
$$

Where $u = \sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 \cdot x^3)$.

Substituting the tanh formula into the Gelu formula and simplifying:

$$
GELU(x) \approx \frac{x}{1 + e^{-2 \cdot \sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 \cdot x^3)}} \tag{3}
$$

Where $-2 \cdot \sqrt{\frac{2}{\pi}} \approx -1.595769$.

This example uses formula (3) for computation. When designing vector operators, users should consider simplifying the original computation to effectively reduce computation steps and memory usage.

## Computation Step Analysis

To illustrate the impact of formula simplification on operator performance, three different Gelu implementation methods are compared.

**Method 1:**

$$
GELU(x) \approx 0.5 \cdot x \cdot \left(1 + \frac{e^{2 \cdot \sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 \cdot x^3)} - 1}{e^{2 \cdot \sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 \cdot x^3)} + 1}\right)
$$

**Method 2:**

$$
GELU(x) \approx \frac{x}{1 + e^{-1.595769 \cdot x - 0.071405 \cdot x^3}}
$$

**Method 3:**

$$
GELU(x) \approx \frac{x}{1 + e^{-1.595769 \cdot (x + 0.044715 \cdot x^3)}}
$$

| Computation Method | Computation Instructions | Unified Buffer (UB) Memory Copies |
|:---|:---:|:---:|
| Method 1 | 13 | 5 |
| Method 2 | 8 | 3 |
| Method 3 | 8 | 2 |

Let the input UB memory be xLocal, output UB memory be yLocal, and temporary UB memory be denoted as tmp0, tmp1, tmp2, and so on.

**Method 1 Computation Step Breakdown** (13 instructions total, 5 memory copies required):

| Step | Computation Content | Computation Instruction | Memory Usage |
|:---:|:---|:---:|:---|
| 1 | yLocal = x² | Mul | xLocal, yLocal |
| 2 | yLocal = x³ | Mul | xLocal, yLocal |
| 3 | yLocal = 0.044715 * x³ | Muls | yLocal |
| 4 | yLocal = x + 0.044715 * x³ | Add | xLocal, yLocal |
| 5 | yLocal = √(2/π) * (x + 0.044715 * x³) | Muls | yLocal |
| 6 | yLocal = 2u | Muls | yLocal |
| 7 | tmp0 = e^(2u) | Exp | tmp0, yLocal |
| 8 | tmp1 = e^(2u) - 1 | Adds | tmp1, yLocal |
| 9 | tmp2 = e^(2u) + 1 | Adds | tmp2, yLocal |
| 10 | tmp1 = tanh(u) = (e^(2u) - 1) / (e^(2u) + 1) | Div | tmp1, tmp2 |
| 11 | yLocal = 1 + tanh(u) | Adds | yLocal, tmp1 |
| 12 | yLocal = x * (1 + tanh(u)) | Mul | xLocal, yLocal |
| 13 | yLocal = 0.5 * x * (1 + tanh(u)) | Muls | yLocal |

**Method 2 Computation Step Breakdown** (8 instructions total, 3 memory copies required):

| Step | Computation Content | Computation Instruction | Memory Usage |
|:---:|:---|:---:|:---|
| 1 | yLocal = x² | Mul | xLocal, yLocal |
| 2 | yLocal = x³ | Mul | xLocal, yLocal |
| 3 | yLocal = -0.071405 * x³ | Muls | yLocal |
| 4 | tmp0 = -1.595769 * x | Muls | tmp0, xLocal |
| 5 | yLocal = -1.595769 * x + (-0.071405 * x³) | Add | yLocal, tmp0 |
| 6 | yLocal = e^(-1.595769 * x - 0.071405 * x³) | Exp | yLocal |
| 7 | yLocal = 1 + e^(...) | Adds | yLocal |
| 8 | yLocal = x / (1 + e^(...)) | Div | yLocal, xLocal |

**Method 3 Computation Step Breakdown** (8 instructions total, 2 memory copies required):

| Step | Computation Content | Computation Instruction | Memory Usage |
|:---:|:---|:---:|:---|
| 1 | yLocal = x² | Mul | yLocal, xLocal |
| 2 | yLocal = x³ | Mul | yLocal, xLocal |
| 3 | yLocal = 0.044715 * x³ | Muls | yLocal |
| 4 | yLocal = x + 0.044715 * x³ | Add | yLocal |
| 5 | yLocal = -1.595769 * (x + 0.044715 * x³) | Muls | yLocal |
| 6 | yLocal = e^(-1.595769 * (x + 0.044715 * x³)) | Exp | yLocal |
| 7 | yLocal = 1 + e^(...) | Adds | yLocal |
| 8 | yLocal = x / (1 + e^(...)) | Div | yLocal, xLocal |

**Comparison Summary**:

- **Computation Instructions**: Methods 2 and 3 both reduce **5 instructions** compared to Method 1
- **Memory Copies**: Method 2 reduces **1 memory copy** compared to Method 1, Method 3 reduces **3 memory copies** compared to Method 1
- **Implementation Choice**: Method 3 is the optimal implementation with the fewest computation instructions and lowest memory usage. This example uses Method 3 for implementation.

**Example Specification**:

<table border="2">
<tr><td rowspan="1" align="center">Example Type(OpType)</td><td colspan="4" align="center">Gelu</td></tr>
<tr><td rowspan="2" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">[8192, 8192]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Example Output</td><td align="center">y</td><td align="center">[8192, 8192]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">gelu_custom</td></tr>
</table>

## Example Implementation

### Performance Metrics Description

**Table 1: AI Core Performance Metrics Field Description**

| Field Name | Field Meaning |
|:---|:---|
|Task Duration(μs)|Overall task duration, including scheduling time to accelerator, execution time on accelerator, and response completion time.|
|aiv_time(μs)|Theoretical execution time of Task on AI Vector Core.|
|aiv_vec_time(μs)|vec type instruction (vector operation instruction) duration.|
|aiv_vec_ratio|The ratio of vec type instruction (vector operation instruction) cycles to total cycles.|
|aiv_scalar_time(μs)|scalar type instruction (scalar operation instruction) duration.|
|aiv_scalar_ratio|The ratio of scalar type instruction (scalar operation instruction) cycles to total cycles.|
|aiv_mte2_time(μs)|mte2 type instruction (GM->UB transfer instruction) duration.|
|aiv_mte2_ratio|The ratio of mte2 type instruction (GM->UB transfer instruction) cycles to total cycles.|
|aiv_mte3_time(μs)|mte3 type instruction (UB->GM transfer instruction) duration.|
|aiv_mte3_ratio|The ratio of mte3 type instruction (UB->GM transfer instruction) cycles to total cycles.|

### Case 0: Gelu Without VF Fusion Enabled

**Implementation Method**: Refer to `KernelGelu::GeluCompute()` function implementation

The baseline program uses Ascend C basic API to implement Gelu computation, including Mul, Muls, Add, Exp, Adds, Div, and other vector instructions.

**Key Code**:
```cpp
__aicore__ inline void GeluCompute(
        const AscendC::LocalTensor<float>& xLocal, const AscendC::LocalTensor<float>& yLocal, uint32_t n)
{
    // yLocal = x * x = x²
    AscendC::Mul(yLocal, xLocal, xLocal, n);
    AscendC::PipeBarrier<PIPE_V>();
    // yLocal = x² * x = x³
    AscendC::Mul(yLocal, yLocal, xLocal, n);
    AscendC::PipeBarrier<PIPE_V>();
    // yLocal = x³ * 0.044715 = 0.044715 * x³
    AscendC::Muls(yLocal, yLocal, COEFF_A, n);
    AscendC::PipeBarrier<PIPE_V>();
    // yLocal = x + 0.044715 * x³
    AscendC::Add(yLocal, xLocal, yLocal, n);
    AscendC::PipeBarrier<PIPE_V>();
    // yLocal = -1.595769 * (x + 0.044715 * x³)
    AscendC::Muls(yLocal, yLocal, COEFF_B, n);
    AscendC::PipeBarrier<PIPE_V>();
    // yLocal = e^(-1.595769 * (x + 0.044715 * x³))
    AscendC::Exp(yLocal, yLocal, n);
    AscendC::PipeBarrier<PIPE_V>();
    // yLocal = 1 + e^(-1.595769 * (x + 0.044715 * x³))
    AscendC::Adds(yLocal, yLocal, (float)1.0, n);
    AscendC::PipeBarrier<PIPE_V>();
    // yLocal = x / (1 + e^(-1.595769 * (x + 0.044715 * x³)))
    AscendC::Div(yLocal, xLocal, yLocal, n);
}
```

**Example Configuration**:
- Multi-core splitting: 2 parts in M direction, 32 parts in N direction, totaling 64 data parts distributed across 64 cores for computation
- `tileLen = 8192` is the number of data elements transferred and computed each time

**Performance Data**:

| Task Duration(μs) | aiv_time(μs) | aiv_vec_time(μs) | aiv_vec_ratio | aiv_scalar_time(μs) | aiv_scalar_ratio | aiv_mte2_time(μs) | aiv_mte2_ratio | aiv_mte3_time(μs) | aiv_mte3_ratio |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 351.525 | 350.64 | 147.895 | 0.422 | 9.513 | 0.027 | 320.283 | 0.913 | 303.647 | 0.866 |

**Optimization Effect Analysis**:
- End-to-end duration: **351.525μs**
- Vector instruction duration: 147.895μs, accounting for **42.2%**
- Data transfer duration: 320.283μs (read) + 303.647μs (write), transfer accounts for over **90%**

**Principle Explanation**:
- The computation flow executes according to formula (3), including 8 vector operations using basic APIs for vector computation
- Each vector computation internally performs load -> compute -> store operations, requiring data exchange between Unified Buffer (UB) and Reg for each computation

**Next Optimization Direction**:
- Enable RegBase API and VF fusion capability to reduce data exchange between UB and Reg, completing final computation result generation within Reg

---

### Case 1: RegBase API and VF Fusion Enabled

**Implementation Method**: Refer to `KernelGelu::GeluVfBasic()` function implementation

Convert basic APIs to RegBase APIs, using register-level vector computation interfaces to reduce data exchange between UB and Reg.

**Key Code**:
```cpp
__simd_vf__ inline void GeluVfBasic(__ubuf__ float* xAddr, __ubuf__ float* yAddr, uint32_t n, uint32_t loopNum)
{
    constexpr uint32_t oneRepeatSize = AscendC::GetVecLen() / sizeof(float);
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::RegTensor<float> xReg, yReg;

    for (uint16_t i = 0; i < loopNum; ++i) {
        mask = AscendC::Reg::UpdateMask<float>(n);
        AscendC::Reg::LoadAlign(xReg, xAddr + i * oneRepeatSize);
        AscendC::Reg::Mul(yReg, xReg, xReg, mask);
        AscendC::Reg::Mul(yReg, yReg, xReg, mask);
        AscendC::Reg::Muls(yReg, yReg, COEFF_A, mask);
        AscendC::Reg::Add(yReg, xReg, yReg, mask);
        AscendC::Reg::Muls(yReg, yReg, COEFF_B, mask);
        AscendC::Reg::Exp(yReg, yReg, mask);
        AscendC::Reg::Adds(yReg, yReg, 1.0f, mask);
        AscendC::Reg::Div(yReg, xReg, yReg, mask);
        AscendC::Reg::StoreAlign(yAddr + i * oneRepeatSize, yReg, mask);
    }
}
```

**Example Configuration**:
- Multi-core splitting: 2 parts in M direction, 32 parts in N direction, totaling 64 data parts distributed across 64 cores for computation
- `tileLen = 8192` is the number of data elements transferred and computed each time

**Optimization Methods**:
- **RegBase API Advantages**:
  - Register-level data access reduces intermediate data Load/Store overhead
  - Supports Hardware Loop optimization, enabling loops to be optimized as hardware-level vector loops

**Performance Data**:

| Task Duration(μs) | aiv_time(μs) | aiv_vec_time(μs) | aiv_vec_ratio | aiv_scalar_time(μs) | aiv_scalar_ratio | aiv_mte2_time(μs) | aiv_mte2_ratio | aiv_mte3_time(μs) | aiv_mte3_ratio |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 348.868 | 347.99 | 66.277 | 0.19 | 3.03 | 0.009 | 320.543 | 0.921 | 314.547 | 0.904 |

**Optimization Effect Analysis**:
- End-to-end duration: **348.868μs**, a **0.76%** reduction compared to Case 0. This example is MTE2 bound, so end-to-end improvement is not significant
- Vector instruction duration: 66.277μs, a **55.2%** reduction compared to Case 0

---

### Case 2: RegBase API, VF Fusion, and Loop Unrolling Optimization Enabled

**Implementation Method**: Refer to `KernelGelu::GeluVfBasic()` function implementation, adding `#pragma unroll 6` loop unrolling optimization

In Case 1, due to the long dependency path in Gelu computation, loop unrolling optimization is used to improve instruction-level parallelism.

**Key Code**:
```cpp
__simd_vf__ inline static void GeluVfBasic(__ubuf__ float* xAddr, __ubuf__ float* yAddr, uint32_t n, uint32_t loopNum)
{
    constexpr uint32_t oneRepeatSize = AscendC::GetVecLen() / sizeof(float);
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::RegTensor<float> xReg, yReg;
    #pragma unroll 6  // Loop unrolling optimization
    for (uint16_t i = 0; i < loopNum; ++i) {
        mask = AscendC::Reg::UpdateMask<float>(n);
        AscendC::Reg::LoadAlign(xReg, xAddr + i * oneRepeatSize);
        // ……
    }
}
```

**Example Configuration**:
- Multi-core splitting: 32 parts in M direction, 2 parts in N direction, totaling 64 data parts distributed across 64 cores for computation
- `tileLen = 8192` is the number of data elements transferred and computed each time

**Optimization Methods**:
- **Loop Unrolling Optimization Principle**:
  - Use `#pragma unroll 6` to instruct the compiler to unroll the loop, unrolling 6 iterations each time
  - Improves instruction-level parallelism, enabling more VF instructions to be issued consecutively

- **Loop Unrolling Benefit Analysis**:
  - Unroll factor selection: `unroll 6` is an empirical value that needs to be tuned for actual scenarios
  - Too much unrolling: May increase register pressure, leading to performance degradation, and code size will also increase
  - Too little unrolling: Optimization effect is not significant
  - Recommendation: Users can use an iterative trial method to find the optimal loop unrolling count

**Performance Data**:

| Task Duration(μs) | aiv_time(μs) | aiv_vec_time(μs) | aiv_vec_ratio | aiv_scalar_time(μs) | aiv_scalar_ratio | aiv_mte2_time(μs) | aiv_mte2_ratio | aiv_mte3_time(μs) | aiv_mte3_ratio |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 344.436 | 343.82 | 63.655 | 0.185 | 4.757 | 0.014 | 315.468 | 0.918 | 306.069 | 0.89 |

**Optimization Effect Analysis**:
- End-to-end duration: **344.436μs**, a **2.0%** reduction compared to Case 0, a **1.3%** reduction compared to Case 1 (MTE2 bound, end-to-end improvement is not significant)
- Vector instruction duration: 63.655μs, a **56.9%** reduction compared to Case 0, a **4.6%** reduction compared to Case 1

---

## Performance Comparison Summary

### Ascend 950PR Performance Comparison

The following table shows the performance data comparison for this example running on Ascend 950 series products:

| Case | Optimization Strategy | Core Count | tileLen | Task Duration(μs) | aiv_vec_time(μs) | Theoretical vector duration(μs) | End-to-end Duration vs Case 0 | vector Duration vs Case 0 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 0 | Gelu without VF fusion enabled (baseline) | 64 | 8192 | 351.525 | 147.895 | 139.02 | 1x | 1x |
| 1 | RegBase API and VF fusion enabled | 64 | 8192 | 348.868 | 66.277 | NA | 1.01x | 2.23x |
| 2 | RegBase API, VF fusion, and loop unrolling enabled | 64 | 8192 | 344.436 | 63.655 | NA | 1.02x | 2.32x |

> **Note:** This example is MTE2 bound, with data transfer as the performance bottleneck. The following analysis focuses on vector duration to help users analyze the performance benefits of enabling VF fusion and loop unrolling. Case 2 reduces vector duration by **4.6%** compared to Case 1.

### Theoretical Performance Analysis

The following table shows the parallelism of vector computation instructions in this example:

| No. | Instruction | Data Type | Computation Parallelism (bytes/cycle) |
|:---:|:---:|:---:|:---:|
| 1 | Mul | float | 256 |
| 2 | Mul | float | 256 |
| 3 | Muls | float | 256 |
| 4 | Add | float | 256 |
| 5 | Muls | float | 256 |
| 6 | Exp | float | 64 |
| 7 | Adds | float | 256 |
| 8 | Div | float | 64 |

**Theoretical Vector Duration Calculation**:

**Step 1: Calculate equivalent data processed per cycle for Gelu computation flow (without considering VF fusion instruction dual-issue characteristic)**

Gelu computation includes 8 vector instructions, with parallelism for each instruction as follows:
- 6 linear operation instructions (Mul×2, Muls×2, Add×1, Adds×1): parallelism 256 bytes/cycle
- 2 nonlinear operation instructions (Exp×1, Div×1): parallelism 64 bytes/cycle

Cycles needed to process 1 byte of data:
- Linear operation instructions: each requires 1/256 cycles
- Nonlinear operation instructions: each requires 1/64 cycles

Total cycles needed to process 1 byte of data:

$$
Cycles_{\text{per\_byte}} = 6 \times \frac{1}{256} + 2 \times \frac{1}{64} = \frac{6}{256} + \frac{2}{64} = \frac{14}{256}
$$

Equivalent data processed per cycle:

$$
P_{\text{bytes}} = \frac{256}{14} = \frac{128}{7} \approx 18.29 \text{ bytes/cycle}
$$

**Step 2: Calculate theoretical vector duration (without considering VF fusion instruction dual-issue characteristic)**

This example runs on Ascend 950PR series products with hardware parameters:
- Clock frequency $f = 1650 \text{ MHz} = 1.65 \times 10^9 \text{ Hz}$
- AIV core count $N_{\text{core}} = 64$
- Data shape $M = N = 8192$, total data $D = M \times N \times 4 = 268435456 \text{ bytes}$

Theoretical vector duration calculation formula:

$$
T_{\text{theory}} = \frac{D}{P_{\text{bytes}} \times f \times N_{\text{core}}}
$$

Substituting values:

$$
T_{\text{theory}} = \frac{268435456}{\frac{128}{7} \times 1.65 \times 10^9 \times 64} = \frac{268435456 \times 7}{128 \times 1.056 \times 10^{11}} = \frac{1879048192}{1.35168 \times 10^{11}} \approx 139.02 \times 10^{-6} \text{ s} = 139.02 \text{ μs}
$$

**Step 3: Calculate theoretical vector duration in VF fusion scenario**

In the VF fusion scenario, vector instructions have dual-issue capability, meaning regular computation instructions can achieve parallelism of 512 bytes/cycle. However, exp and div cannot be simply estimated at 128 bytes/cycle in dual-issue scenarios.
As shown in the following diagram, at time=0, Exp and Mul instructions are issued simultaneously. Mul only needs 1 cycle to complete execution, but Exp needs 4 cycles to complete. During Exp execution, at time=1, Mul and Muls are issued and executed in parallel, and other instructions continue to be issued in this manner.

```
Time:     0     1     2     3     4     5     6     7
          |-----|-----|-----|-----|-----|-----|-----|
Exp:      |<=====================>|
Mul:      |<--->|
Mul:            |<--->|
Muls:           |<--->|
Add:                  |<--->|
Adds:                 |<--->|
Muls:                       |<--->|
Adds:                       |<--->|
```
Therefore, in VF fusion scenarios, IPC (Instructions Per Cycle, number of instructions issued per cycle) is more commonly used to measure performance.

Case 1 scenario, single-core vector computation instruction count:

$$
N_{\text{instr}} = N_{\text{VF}} \times N_{\text{loop}} \times N_{\text{op}} = 128 \times 128 \times 8 = 131072
$$

Where:
- $N_{\text{VF}} = 128$: Number of VF function calls
- $N_{\text{loop}} = 128$: Number of loops inside VF
- $N_{\text{op}} = 8$: Number of computation instructions inside VF (Mul×2, Muls×2, Add×1, Adds×1, Exp×1, Div×1)

Vector computation cycle count:

$$
Cycles_{\text{vec}} = T_{\text{vec}} \times f = 66.277 \text{ μs} \times 1650 \text{ MHz} = 109357
$$

Where:
- $T_{\text{vec}} = 66.277 \text{ μs}$: Measured vector computation duration
- $f = 1650 \text{ MHz}$: Hardware clock frequency

IPC calculation:

$$
IPC = \frac{N_{\text{instr}}}{Cycles_{\text{vec}}} = \frac{131072}{109357} \approx 1.20
$$

Where:
- $N_{\text{instr}} = 131072$: Single-core vector computation instruction count
- $Cycles_{\text{vec}} = 109357$: Vector computation cycle count

Higher IPC is better, with theoretical limit approaching 2.

**Case 2: IPC Analysis After Loop Unrolling Optimization**

Case 2 scenario, adding loop unrolling optimization on top of VF fusion. Single-core vector computation instruction count remains unchanged:

$$
N_{\text{instr}} = N_{\text{VF}} \times N_{\text{loop}} \times N_{\text{op}} = 128 \times 128 \times 8 = 131072
$$

The parameters in the formula are the same as Case 1, loop unrolling does not affect total instruction count.

Vector computation cycle count:

$$
Cycles_{\text{vec}} = T_{\text{vec}} \times f = 63.655 \text{ μs} \times 1650 \text{ MHz} = 105031
$$

IPC calculation:

$$
IPC = \frac{N_{\text{instr}}}{Cycles_{\text{vec}}} = \frac{131072}{105031} \approx 1.25
$$

**Impact of Loop Unrolling on IPC Analysis**:
- Case 2 IPC is 1.25, a **4.2%** improvement compared to Case 1 IPC (1.20)
- Loop unrolling optimization improves vector instruction issue efficiency, enabling more instructions to execute in parallel

**IPC Optimization Recommendations**:
- In VF fusion scenarios, IPC is an important metric for measuring vector instruction issue efficiency
- Ideal IPC limit approaches 2.0. In most cases, achieving 1.4~1.5 indicates good performance
- When the instruction dependency chain within a VF function loop is too long, the execution queue contains too few loop iterations, resulting in fewer instructions that can be dual-issued per cycle and reduced performance. In this case, you can split the long loop into multiple loops. For example, you can choose to split at reduce endpoints or at endpoints of long-latency instructions (such as div, exp), splitting one loop into 2~3 loops. The example loop provided in this sample is short and not suitable for this optimization.

### Optimization Key Points Summary

| Optimization Method | Core Principle | Usage Recommendation |
|:---|:---|:---|
| Formula Simplification | Reduce computation steps and computation overhead | Prioritize formula derivation and simplification |
| RegBase API + VF Fusion | Register-level computation reduces intermediate Load/Store, leverage dual-issue capability to improve performance | Use asc_vf_call to invoke VF functions, leverage dual-issue capability to improve IPC |
| Loop Unrolling | Improve instruction issue parallelism | Use `#pragma unroll N`, N needs to be tuned based on actual conditions |

---

## Build and Run

Execute the following steps in the root directory of this example to build and run the example.

- Switch Case

  Specify the case to compile through `-DSCENARIO_NUM=N` during cmake compilation. Case descriptions:
  - `0`: Gelu without VF fusion enabled (requires setting `-DCMAKE_VF_MODE=false`)
  - `1`: RegBase API and VF fusion enabled
  - `2`: RegBase API, VF fusion, and loop unrolling optimization enabled

  > **Note:** The compiler has automatic VF fusion capability and enables VF auto-fusion by default. For performance comparison analysis in this example, VF auto-fusion needs to be disabled in case 0.

- Configure Environment Variables

  Select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on your current environment.
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

- Execute Example

  ```bash
  SCENARIO_NUM=0
  CMAKE_VF_MODE=false
  mkdir -p build && cd build;      # Create and enter build directory
  cmake .. -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_VF_MODE=$CMAKE_VF_MODE;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                           # Execute the compiled executable program to run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output result correctness and confirm algorithm logic
  ```

  When using NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake .. -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_VF_MODE=$CMAKE_VF_MODE;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching compilation mode or Case, you need to clean the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then re-run cmake.

- Compilation Options Description

  | Option　　　　　 | Available Values　　　　　　　　　　　| Description　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
  | ----------------| -----------------------------| --------------------------------------------------------------------------------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation　　　　　　　　　　　　　　　　　　　　　　　　 |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  | `SCENARIO_NUM` | `0`, `1`, `2`　　　　　| Case number: 0=Gelu without VF fusion, 1=RegBase API and VF fusion enabled, 2=RegBase API, VF fusion, and loop unrolling enabled |
  | `CMAKE_VF_MODE` | `true`, `false`　　　　　| VF fusion mode: case 0 requires setting to false to disable VF auto-fusion |

- Execution Result

  The following output indicates successful precision comparison:
  ```bash
  error ratio: 0.0000, tolerance:0.0001
  test pass!
  ```

### Performance Analysis

Use the `msprof` tool to obtain detailed performance data:

```bash
msprof ./demo   # Analyze performance
```

A folder with the PROF_ prefix will be generated in the current directory. The `mindstudio_profiler_output` directory contains the performance data summary for Host and each Device. For performance data analysis, it is recommended to view the files in this directory:

```bash
PROF_xxxx_XXXXXX
├── device_{id}
└── host
└── mindstudio_profiler_log
└── mindstudio_profiler_output    # Performance data summary for Host and each Device
    ├── msprof_*.json
    ├── xx_*.csv
    └── README.txt
```

View specific performance analysis results:
```bash
# View Task Duration and other data
cat ./PROF_*/mindstudio_profiler_output/op_summary_*.csv
```