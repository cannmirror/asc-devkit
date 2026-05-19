# Add Performance Tuning Sample

## Overview

This sample uses addition as an example to introduce performance tuning methods for static Tensor-based programming. The entire tuning process is divided into seven steps (cases 0-6), progressively demonstrating the complete optimization path from scalar to vector operations, from single-core to multi-core, and from basic implementation to deep optimization.

**Optimization Path**:
- Case 0: Single-core scalar version (baseline)
- Case 1: Single-core vector version
- Case 2: Multi-core even splitting + small block transfer
- Case 3: Multi-core even splitting + large block transfer
- Case 4: Multi-core even splitting + double buffer optimization
- Case 5: Multi-core even splitting + double buffer + L2Cache bypass
- Case 6: Multi-core even splitting + double buffer + L2Cache bypass + Bank Conflict avoidance

## Supported Products
- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── add_high_performance
│   ├── scripts
│   │   ├── gen_data.py         // Input data and golden data generation script
│   │   └── verify_result.py    // Golden data comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read and write functions
│   └── add.asc                 // Ascend C sample implementation (containing 7 optimization cases)
```

## Sample Description

**Sample Functionality**:

  The sample implements the addition of two matrices with a fixed shape of 8192×8192.

  The Add calculation formula is:

$$
  z = x + y
$$

  - x: input, shape [8192, 8192], data type half;
  - y: input, shape [8192, 8192], data type half;
  - z: output, shape [8192, 8192], data type half;

## Sample Implementation

### Performance Metrics Description

**Table 1 AI Core Performance Metrics Field Description**
| Field Name | Field Meaning |
|:---:|:---|
| Task Duration(μs)|Overall task duration, including time scheduled to accelerator, execution time on accelerator, and response completion time.|
| aiv_time|Theoretical execution time of Task on AI Vector Core, in μs.|
| aiv_vec_time(μs) | vec type instruction (vector operation instruction) duration, in μs. |
| aiv_vec_ratio | Ratio of vec type instruction (vector operation instruction) cycles to total cycles. |
| aiv_scalar_time(μs) | scalar type instruction (scalar operation instruction) duration, in μs. |
| aiv_scalar_ratio | Ratio of scalar type instruction (scalar operation instruction) cycles to total cycles. |
| aiv_mte2_time(μs) | mte2 type instruction (GM->UB transfer instruction) duration, in μs. |
| aiv_mte2_ratio | Ratio of mte2 type instruction (GM->UB transfer instruction) cycles to total cycles. |
| aiv_mte3_time(μs) | mte3 type instruction (UB->GM transfer instruction) duration, in μs. |
| aiv_mte3_ratio | Ratio of mte3 type instruction (UB->GM transfer instruction) cycles to total cycles. |

The performance data in this chapter was obtained on Atlas A2 Training Series Products.

### Case 0: Single-core Scalar Version (Baseline)

**Implementation**: Refer to `KernelAdd::ProcessScalar()` function implementation

The baseline program implements addition of two `half` type input data sets using `for` loop with `scalar` operations for computation.

**Key Code**:
```cpp
for (uint32_t i = 0; i < curLen; i++) {
      float xVal = (float)xLocal.GetValue(i);
      float yVal = (float)yLocal.GetValue(i);
      zLocal.SetValue(i, (half)(xVal + yVal));
    }
```

**Sample Configuration**:
- Single-core scalar operation
- `dataCopyLen = 4096` is the number of data elements transferred each time
- Single transfer data size is 4096 * 2B = 8192 Byte, single scalar processing data size is 4 Byte

**Performance Data**:

| Task Duration(μs) | aiv_time(μs) | aiv_vec_time(μs) | aiv_vec_ratio | aiv_scalar_time(μs) | aiv_scalar_ratio | aiv_mte2_time(μs) | aiv_mte2_ratio | aiv_mte3_time(μs) | aiv_mte3_ratio |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1239689.1 | 1239688.63 | 0.015 | 0 | 1233742.494 | 0.995 | 5916.341 | 0.005 | 2485.465 | 0.002 |

**Optimization Analysis**:
- End-to-end duration: **1239689.1μs** (approximately 1.24 seconds)
- Scalar instruction duration: 1233742.494μs, accounting for **99.5%**
- Vector instruction duration: 0.015μs, accounting for approximately 0%
- Performance bottleneck: Scalar operations execute serially, unable to utilize hardware parallelism capabilities. This scenario serves only as a performance comparison sample for Add operations. Using Scalar operations is not recommended in actual business scenarios.

**Principle Explanation**:
- Scalar operations can only process 1 data element at a time, requiring element-by-element loops
- AI Core hardware advantages lie in vector/matrix parallel computation; scalar operations cannot leverage hardware capabilities

**Performance Optimization Recommendation**:
> ⚠️ **Avoid scalar loops, use vector instructions**
> 
> In Ascend C programming, avoid using `for` loops with `GetValue/SetValue` scalar operations. Using vector instructions such as `AscendC::Add` can achieve order-of-magnitude performance improvements.

---

### Case 1: Single-core Vector Version

**Implementation**: Refer to `KernelAdd::ProcessSingle()` function implementation

Convert scalar operations to vector operations, using `AscendC::Add` vector instructions instead of scalar loops, significantly improving computation efficiency.

**Key Code**:
```cpp
AscendC::Add(zLocal, xLocal, yLocal, curLen);
```

**Sample Configuration**:
- Single-core operation
- `dataCopyLen = 4096` is the number of data elements transferred each time
- Single transfer operation `DataCopy` data size is 8192 Byte
- Single `Add` processes two input `Tensor`s, total data size processed is 16384 Byte


**Performance Data**:

| Task Duration(μs) | aiv_time(μs) | aiv_vec_time(μs) | aiv_vec_ratio | aiv_scalar_time(μs) | aiv_scalar_ratio | aiv_mte2_time(μs) | aiv_mte2_ratio | aiv_mte3_time(μs) | aiv_mte3_ratio |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 6909.6 | 6909.14 | 761.65 | 0.11 | 231.166 | 0.033 | 6208.762 | 0.873 | 2613.205 | 0.378 |

**Optimization Analysis**:
- End-to-end performance: 6909.6μs, improved by **99.4%** compared to Case 0
- Scalar instruction duration: decreased from 1233742.494μs to 231.166μs, significantly reduced scalar instructions
- Vector instruction duration: 761.65μs, accounting for 11%
- Data transfer duration: 6208.762μs, accounting for 87.3%, transfer pipeline serial improvement

**Principle Explanation**:
- Vector instructions can process multiple data elements at once (in this example, processing 4096*2 half elements at a time)
- Vector unit parallel computation capability far exceeds scalar unit
- However, data transfer becomes the bottleneck, indicating computation speed has exceeded data supply. In single-core scenarios, insufficient transfer request volume leads to underutilized bandwidth

**Performance Optimization Recommendation**:
> 💡 **Use vector instructions instead of scalar loops**
> 
> Using vector APIs such as `AscendC::Add` and `AscendC::Mul` instead of element-by-element scalar loops can fully utilize AI Core vector computation units, achieving over 100x performance improvement.

> 💡 **Using only single-core is not recommended**

**Next Optimization Direction**:
- Data transfer (MTE2) accounts for 87.3%, becoming the main bottleneck
- Need to improve bandwidth utilization through multi-core parallelism and larger transfer granularity

---

### Case 2: Multi-core Even Splitting + Small Block Transfer

**Implementation**: Refer to `KernelAdd::Process()` function implementation

Enable multi-core parallel computation, splitting the 8192×8192 matrix across multiple AIV Cores for parallel processing using an even splitting strategy.

**Sample Configuration**:
- Split 48 ways in row direction, evenly distributing data across 48 cores for computation
- `dataCopyLen = 4096` is the number of data elements transferred each time
- Single transfer operation `DataCopy` data size is 4096 * 2B = 8192 Byte
- Single `Add` processes two input `Tensor`s, total data size processed is 16384 Byte


**Key Code**:
```cpp
// Evenly split to calculate the number of rows each core processes (M direction)
uint32_t baseCoreM = totalM / splitM;
uint32_t remainderM = totalM % splitM;
if (blockIdxM < remainderM) {
    actualCoreM = baseCoreM + 1;  // Evenly distribute remainder
    startM = blockIdxM * actualCoreM;
} else {
    actualCoreM = baseCoreM;
    startM = remainderM * (baseCoreM + 1) + (blockIdxM - remainderM) * baseCoreM;
}
```

**Performance Data**:

| Task Duration(μs) | aiv_time(μs) | aiv_vec_time(μs) | aiv_vec_ratio | aiv_scalar_time(μs) | aiv_scalar_ratio | aiv_mte2_time(μs) | aiv_mte2_ratio | aiv_mte3_time(μs) | aiv_mte3_ratio |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 306.58 | 297.14 | 15.885 | 0.053 | 8.219 | 0.028 | 215.485 | 0.725 | 54.33 | 0.183 |

**Optimization Analysis**:
- End-to-end task duration is 306.58μs, reduced by **95.5%** compared to Case 1
- Data transfer MTE2 duration is 215.485μs, accounting for 72.5%

**Principle Explanation**:
- 48 AIV Cores processing in parallel can theoretically achieve 48x speedup
- Reasons why actual speedup is lower than theoretical:
  - Data transfer is still the bottleneck (mte2 accounts for 72.5%)
- Even splitting ensures load balancing across cores, avoiding load imbalance where some cores are idle while others are busy

- This sample chooses to split 48 ways in `M` direction (`splitM=48, splitN=1`), rather than splitting `M` into 8 and `N` into 6. The core purpose is to keep each core's data contiguous on GM, allowing single `DataCopy` to transfer `dataCopyLen` in the block loop

  <img src="figure/SplitCoreM.png" width="50%">

**Performance Optimization Recommendation**:
> 💡 **Fully utilize multi-core parallelism, adopt even splitting strategy**
> 
> 1. Evenly split data across multiple AIV Cores for parallel computation
> 2. Use even splitting strategy (distribute remainder to first few cores) to ensure load balancing
> 3. Split granularity needs to consider: number of cores, data volume, UB space size

**Next Optimization Direction**:
- MTE2 accounts for 72.5%, MTE3 accounts for 18.3%, transfer is still the bottleneck
- Computation only accounts for 5.3%, indicating "fast computation, slow transfer"
- Can improve bandwidth utilization by increasing single transfer data volume
---

### Case 3: Multi-core Even Splitting + Large Block Data Transfer

**Implementation**: Refer to `KernelAdd::Process()` function implementation

To fully utilize bandwidth resources, increase the data volume per transfer instruction.

**Sample Configuration**:
- Split 48 ways in row direction, evenly distributing data across 48 cores for computation
- `dataCopyLen = 16384` is the number of data elements per split (4x Case 2)
- Single transfer operation `DataCopy` data size is 32768 Byte
- Single `Add` processes two input `Tensor`s, total data size processed is 65536 Byte


**Performance Data**:

| Task Duration(μs)  | aiv_time(μs) | aiv_vec_time(μs) | aiv_vec_ratio | aiv_scalar_time(μs) | aiv_scalar_ratio | aiv_mte2_time(μs) | aiv_mte2_ratio | aiv_mte3_time(μs) | aiv_mte3_ratio |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 268.5 |261.5 | 12.845 | 0.049 | 2.833 | 0.011 | 184.369 | 0.705 | 57.354 | 0.219 |

**Optimization Analysis**:
- End-to-end performance: 268.5μs, reduced by **12.5%** compared to Case 2
- By increasing single transfer data volume, MTE2 duration decreased from 215.485μs to 184.369μs, reduced by **14.4%**

**Principle Explanation**:
- Increasing single transfer data volume can reduce transfer count
- Larger contiguous data blocks can better utilize memory bandwidth
- However, data volume is limited by UB space size (in this example, UB needs to hold x, y, z three sets of data)

**Performance Optimization Recommendation**:
> 💡 **Increase single data transfer volume, reduce transfer count**
> 
> 1. Within UB space limits, maximize `dataCopyLen`
> 2. Use contiguous large block data transfers, avoid frequent small data block transfers
> 3. Need to balance UB space usage and transfer efficiency


> ⚠️ **Note: dataCopyLen is not always better when larger**
> 
> On the basis of Case 3, if further increasing dataCopyLen (for example, from 16384 to 16512), end-to-end performance remains basically the same (268.5μs vs 267.76μs). It is recommended to consider total data volume, UB space, and alignment requirements to determine the optimal dataCopyLen value.

---

### Case 4: Double Buffer Optimization

**Implementation**: Refer to `KernelAdd::ProcessDoubleBuffer()` function implementation

Using double buffer technology to achieve pipeline parallelism between data transfer and computation, hiding memory access latency.

**Key Code**:
```cpp
// Ping-Pong double buffer addresses
static constexpr uint32_t xAddrPing = 0;
static constexpr uint32_t yAddrPing = MAX_DATA_COPY_LEN * sizeof(half);
static constexpr uint32_t zAddrPing = yAddrPing + MAX_DATA_COPY_LEN * sizeof(half);
static constexpr uint32_t xAddrPong = zAddrPing + MAX_DATA_COPY_LEN * sizeof(half);
static constexpr uint32_t yAddrPong = xAddrPong + MAX_DATA_COPY_LEN * sizeof(half);
static constexpr uint32_t zAddrPong = yAddrPong + MAX_DATA_COPY_LEN * sizeof(half);

// Double buffer pipeline: alternately use two event IDs and two sets of buffers
for (uint32_t loopIdx = 0; loopIdx < totalBlocks; loopIdx++) {
    int32_t eventID = (loopIdx % 2 == 0 ? EVENT_ID0 : EVENT_ID1);
    AscendC::LocalTensor<half> &xLocal = (loopIdx % 2 == 0 ? xPing : xPong);
    // ... data transfer and computation, use corresponding eventID for synchronization
    AscendC::Add(zLocal, xLocal, yLocal, curLen);
}
```

**Sample Configuration**:
- Split 48 ways in row direction, evenly distributing data across 48 cores for computation
- `dataCopyLen = 16384` is the number of data elements per split
- Single transfer operation `DataCopy` data size is 32768 Byte
- Single `Add` processes two input `Tensor`s, total data size processed is 65536 Byte
- Split data to be processed into two, enabling parallel execution of data transfer in/out and Vector computation

**Memory Layout**:

```
UB Memory Allocation (Double Buffer):
┌──────────────┐
│  xPing       │  0x00000
│  16384*2B    │
├──────────────┤
│  yPing       │  0x08000 (32768)
│  16384*2B    │
├──────────────┤
│  zPing       │  0x10000 (65536)
│  16384*2B    │
├──────────────┤
│  xPong       │  0x18000 (98304)
│  16384*2B    │
├──────────────┤
│  yPong       │  0x20000 (131072)
│  16384*2B    │
├──────────────┤
│  zPong       │  0x28000 (163840)
│  16384*2B    │
└──────────────┘
```

**Performance Data**:

| Task Duration(μs) | aiv_time(μs) | aiv_vec_time(μs) | aiv_vec_ratio | aiv_scalar_time(μs) | aiv_scalar_ratio | aiv_mte2_time(μs) | aiv_mte2_ratio | aiv_mte3_time(μs) | aiv_mte3_ratio |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 264.02 | 257.56 | 12.846 | 0.05 | 2.796 | 0.011 | 250.528 | 0.973 | 84.988 | 0.33 |

**Optimization Analysis**:
- End-to-end performance: 264.02μs, reduced by **1.7%** compared to Case 3
- MTE2 duration increased from 184.369μs to 250.528μs (+35.9%), MTE3 duration increased from 57.354μs to 84.988μs (+48.2%). At this point, it changes from serial pure read bandwidth to mixed read-write bandwidth, so duration increases. Users should focus more on the reduction in end-to-end duration
- Since double buffer is enabled, transfer and computation execute in parallel in the pipeline, hiding data transfer time and reducing Vector instruction wait time

**Principle Explanation**:
- **Ping-Pong Mechanism**:
  - When Ping buffer is computing, Pong buffer is transferring data
  - Alternating execution achieves pipeline parallelism between computation and transfer, as shown in the figure below
    <img src="figure/DoubleBuffer.png" width="50%">

**Performance Optimization Recommendation**:
> 💡 **Use double buffer to achieve transfer and computation parallelism**
> 
> 1. Double buffer yields maximum benefit when computation and transfer times are similar
> 2. Requires sufficient UB space (approximately 2x single buffer space)
> 3. Use independent Event IDs to manage synchronization between two sets of buffers

**Next Optimization Direction**:
- Double buffer benefit is limited, indicating the bottleneck is transfer speed itself
- Can try L2 Cache optimization to improve transfer efficiency
---

### Case 5: Double Buffer + L2 Cache Bypass

**Implementation**: Refer to `KernelAdd::ProcessDoubleBufferL2Bypass()` function implementation (sets `SetL2CacheHint(CACHE_MODE_DISABLE)` first inside this function, then calls `ProcessDoubleBuffer()`)

On top of double buffer, for data that only needs to be loaded once, L2 Cache bypass can be set to load directly from GM to UB.

**Key Code**:
```cpp
  xGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
  yGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
```

**Sample Configuration**:
- Split 48 ways in row direction, evenly distributing data across 48 cores for computation
- `dataCopyLen = 16384` is the number of data elements per split
- Single transfer operation `DataCopy` data size is 32768 Byte
- Single `Add` processes two input `Tensor`s, total data size processed is 65536 Byte
- Split data to be processed into two, enabling parallel execution of data transfer in/out and Vector computation

**L2 Cache Strategy**:
- xGm: Disable L2 Cache (one-time read)
- yGm: Disable L2 Cache (one-time read)

**Performance Data**:

| Task Duration(μs) | aiv_time(μs) | aiv_vec_time(μs) | aiv_vec_ratio | aiv_scalar_time(μs) | aiv_scalar_ratio | aiv_mte2_time(μs) | aiv_mte2_ratio | aiv_mte3_time(μs) | aiv_mte3_ratio |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 187.68 | 185.15 | 12.846 | 0.069 | 4.479 | 0.024 | 175.997 | 0.951 | 160.291 | 0.866 |

**Optimization Analysis**:
- End-to-end performance: 187.68μs, reduced by **28.9%** compared to Case 4
- MTE2 duration: decreased from 250.528μs to 175.997μs, reduced by **29.8%**
- Vector instruction duration: 12.846μs, unchanged

**Principle Explanation**:
- **L2 Cache Function**:
  - L2 Cache is the cache layer between AI Core and HBM
  - Repeatedly accessed data can be read from L2 Cache faster
- **Streaming Access Characteristics**:
  - Add input data is only read once, no data reuse exists

**Performance Optimization Recommendation**:
> 💡 **Reasonably use L2 Cache bypass**
> 
> 1. For input data that is only read once (such as x, y in this example), set `SetL2CacheHint(CACHE_MODE_DISABLE)`
> 2. For data that needs repeated access (such as convolution weights), keep L2 Cache
> 3. Users are advised to configure optimization based on actual measured data. In actual model and training scenarios, reasonable configuration needs to be combined with upstream and downstream

**Next Optimization Direction**:
- Transfer efficiency has improved, but vector instruction efficiency still has room for optimization
- Can try optimizing UB memory layout to avoid Bank Conflict
---

### Case 6: Double Buffer + L2 Cache Bypass + Bank Conflict Avoidance

**Implementation**: Refer to `KernelAdd::ProcessDoubleBufferBankConflict()` function implementation

On top of double buffer and L2 Cache bypass, optimize memory address layout to avoid UB (Unified Buffer) Bank Conflict and achieve optimal performance.

**Key Code**:
```cpp
// Set L2 Cache bypass
xGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
yGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

// Optimized address layout (avoid Bank Conflict)
static constexpr uint32_t xAddrPingBC = 0;
static constexpr uint32_t yAddrPingBC = BANK_CONFLICT_DATA_COPY_LEN * sizeof(half);
static constexpr uint32_t xAddrPongBC = MAX_DATA_COPY_LEN * sizeof(half) * 2;
static constexpr uint32_t yAddrPongBC = xAddrPongBC + BANK_CONFLICT_DATA_COPY_LEN * sizeof(half);
static constexpr uint32_t zAddrPingBC = MAX_DATA_COPY_LEN * sizeof(half) * 4;
static constexpr uint32_t zAddrPongBC = zAddrPingBC + BANK_CONFLICT_DATA_COPY_LEN * sizeof(half);
```

**Sample Configuration**:
- Split 48 ways in row direction, evenly distributing data across 48 cores for computation
- `dataCopyLen = 16256` is the number of data elements per split
- Single transfer operation `DataCopy` data size is 32512 Byte
- Single `Add` processes two input `Tensor`s, total data size processed is 65204 Byte
- Split data to be processed into two, enabling parallel execution of data transfer in/out and Vector computation

**Memory Layout Optimization**:

For Atlas A2/A3 series products, UB size is 192KB, containing 16 Bank Groups, each Bank Group contains 3 Banks, each Bank size is 4KB, composed of 128 rows, each row length is 32B.

UB Bank memory layout before optimization (that is, case5)
<img src="figure/UBBankConflict.png" width="100%">

It can be seen that there are simultaneously read-write conflicts within one bank, read-read conflicts within one bankgroup, and write-write conflicts.

Optimized UB Bank memory layout (case6)
<img src="figure/UBBankConflictResolution.png" width="100%">

Since vec instruction reads 256B of data in one beat (that is, reads 8 blocks of data simultaneously), as shown in the figure above, the starting addresses of xping and yping are offset by exactly 256B, effectively eliminating ub bank conflicts.

**Bank Conflict Details**:
- UB is divided into multiple Bank Groups, simultaneously reading and writing the same Bank Group causes conflicts
- By adjusting dataCopyLen (16384→16256) to offset data starting addresses
- Ensure data accessed by vec instruction in one beat is distributed across different Bank Groups

**Performance Data**:

| Task Duration(μs) | aiv_time(μs) | aiv_vec_time(μs) | aiv_vec_ratio | aiv_scalar_time(μs) | aiv_scalar_ratio | aiv_mte2_time(μs) | aiv_mte2_ratio | aiv_mte3_time(μs) | aiv_mte3_ratio |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 184.52 | 178.49 | 6.954 | 0.039 | 3.424 | 0.019 | 171.611 | 0.961 | 121.442 | 0.68 |

**Optimization Analysis**:
- End-to-end performance: 184.52μs, reduced by **1.7%** compared to Case 5
- Vector instruction duration: decreased from 12.846μs to 6.954μs, reduced by **45.9%**
- MTE3 duration: decreased from 160.291μs to 121.442μs, reduced by **24.2%**

**Principle Explanation**:
- **Bank Conflict Issue**:
  - UB (Unified Buffer) is divided into multiple Bank Groups
  - If data read and written by vector instruction in one operation falls in the same Bank, read-write conflicts occur
  - If data read or written by vector instruction in one operation falls in the same Bank Group, read-read conflicts or write-write conflicts occur
  - Bank Conflict causes memory access serialization, reducing vector instruction efficiency
- **Solution**:
  - Reduce dataCopyLen (16384→16256) to offset data starting addresses
  - Redesign memory layout to ensure data accessed at the same time is distributed across different Banks

**Performance Optimization Recommendation**:
> 💡 **Optimize UB memory layout to avoid Bank Conflict**
> 
> 1. When `aiv_vec_time` is abnormally high, Bank Conflict may exist
> 2. By adjusting dataCopyLen or memory layout offset, distribute data across different Bank Groups
> 3. Bank Conflict optimization provides significant benefit in vector-bound scenarios

**Final Performance Summary**:
- Compared to baseline Case 0: Performance improved by **6711x** (1239689.1μs → 184.52μs)
- Compared to single-core vector Case 1: Performance improved by **37.3x** (6909.6μs → 184.52μs)


---

## Performance Comparison Summary

### Atlas A2 Training Series Performance Comparison
The following table shows performance data comparison for this sample running on Atlas A2 Training Series Products:

| Case | Optimization Strategy | Core Count | dataCopyLen | Task Duration(μs) | aiv_vec_time(μs) | Theoretical vector duration(μs) | Read Bandwidth(TB/S) | Read-Write Mixed Bandwidth(TB/S) | End-to-end Duration vs Case 0 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | Single-core scalar (baseline) | 1 | 4096 | 1239689.1 | NA | 283.405 | 0.0454 | NA | 1x |
| 1 | Single-core vector | 1 | 4096 | 6909.6 | 761.65 | 283.405 | 0.0432 | NA | 179.4x |
| 2 | Multi-core even splitting | 48 | 4096 | 306.58 | 15.885 | 5.904 | 1.2457 | NA | 4043.6x |
| 3 | Increase transfer granularity | 48 | 16384 | 268.5 | 12.845 | 5.904 | 1.4560 | NA | 4617.1x |
| 4 | Double buffer | 48 | 16384 | 264.02 | 12.846 | 5.904 | NA | 1.6072 | 4695.4x |
| 5 | L2 Cache bypass | 48 | 16384 | 187.68 | 12.846 | 5.904 | NA | 2.2755 | 6612.2x |
| 6 | Bank Conflict optimization | 48 | 16256 | 184.52 | 6.954 | 5.904 | NA | 2.3456 | 6710.7x |

The "Theoretical vector duration" in the table represents the theoretical execution time considering only Vector computation itself under the current core count configuration. The performance data in this sample was obtained on Atlas A2 Training Series Products, where the processor processes 128 half data per cycle with a main frequency of 1.85GHz. The theoretical vector duration calculation formula is
$$
T_{\text{theory}} = \frac{M \times N}{128 \times 1.85 \times 10^9 \times \text{core count}}
$$

For example, in the 48-core scenario:
$$
T_{\text{theory}} = \frac{8192 \times 8192}{128 \times 1.85 \times 10^9 \times 48} = \frac{67108864}{1.13664 \times 10^{13}} \approx 5.904 \times 10^{-6} \text{ s} = 5.904 \text{ μs}
$$

It can be seen that Case 6's aiv_vec_time is 6.954 μs, already very close to the theoretical duration in the 48-core scenario.

Case 0-3 did not enable double buffer, data transfers execute serially, using read bandwidth to measure performance. From Case 4, double buffer is enabled, at which point mte2 utilization is high, read and write behaviors occur in parallel, therefore read-write mixed bandwidth is estimated by dividing total read-write data volume by $T_{mte2}$. The read bandwidth calculation formula is:
$$
BW_{read} = \frac{D_{read}}{T_{mte2}}
$$

The read-write mixed bandwidth calculation formula is:
$$
BW_{rw} = \frac{D_{read} + D_{write}}{T_{mte2}}
$$

Where:
- $D_{read} = M \times N \times sizeof(half) \times 2$ is the total read data volume (x and y two input matrices)
- $D_{write} = M \times N \times sizeof(half)$ is the total write data volume (z output matrix)
- $T_{mte2}$ is aiv_mte2_time (GM→UB transfer duration, μs)
- After enabling double buffer, mte2 and mte3 pipeline in parallel; from this sample's data, Case 4-6's mte2 utilization rates are 97.3%, 95.3%, 95.2% respectively, therefore $T_{mte2}$ is used here as the main path time for estimation

Taking Case 3 as an example ($M=N=8192$, $T_{mte2}=184.369\mu s$):
$$
BW_{read} = \frac{8192 \times 8192 \times 2 \times 2}{184.369 \times 10^{-6}} = \frac{268435456}{184.369 \times 10^{-6}} \approx 1.4560 \times 10^{12} \text{ B/s} \approx 1.4560 \text{ TB/s}
$$

Taking Case 6 as an example ($T_{mte2}=171.611\mu s$):
$$
BW_{rw} = \frac{8192 \times 8192 \times (2+1) \times 2}{171.611 \times 10^{-6}} = \frac{402653184}{171.611 \times 10^{-6}} \approx 2.3456 \times 10^{12} \text{ B/s} \approx 2.3456 \text{ TB/s}
$$

The reason why read-write mixed bandwidth exceeds 1.8 TB/s is that this measures not pure read bandwidth but "read + write" mixed bandwidth. After enabling double buffer, read-write pipeline is parallel; meanwhile, when z writes data it hits L2 Cache, and L2 bandwidth is high, so write bandwidth is high. Therefore, when using mte2 as read-write mixed time for estimation, the numerator counts total read-write data volume, and the final mixed bandwidth obtained will exceed 1.8 TB/s.

### Ascend 950 Series Performance Comparison

The following table shows performance data comparison for this sample running on Ascend 950 series products:

| Case | Optimization Strategy | Core Count | dataCopyLen | Task Duration(μs) | aiv_vec_time(μs) | Theoretical vector duration(μs) | Read Bandwidth(TB/S) | Read-Write Mixed Bandwidth(TB/S) | End-to-end Duration vs Case 0 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | Single-core scalar (baseline) | 1 | 4096 | 1228475.435 | NA | 317.750 | 0.034 | NA | 1x |
| 1 | Single-core vector | 1 | 4096 | 8924.021 | 943.362 | 317.750 | 0.035 | NA | 137.7x |
| 2 | Multi-core even splitting | 64 | 4096 | 310.489 | 15.465 | 4.965 | 1.184 | NA | 3956.6x |
| 3 | Increase transfer granularity | 64 | 21760 | 251.684 | 11.121 | 4.965 | 1.454 | NA | 4881.0x |
| 4 | Double buffer | 64 | 21760 | 247.665 | 10.976 | 4.965 | NA | 1.712 | 4960.2x |
| 5 | L2 Cache bypass | 64 | 21760 | 182.04 | 10.997 | 4.965 | NA | 2.256 | 6748.4x |

The main reason for increased `dataCopyLen` on Ascend 950 series is that UB capacity increased from 192KB to 256KB. In this sample, double buffer scenarios need to hold `x/y/z` Ping-Pong buffers simultaneously, totaling 6 blocks, so the approximate available space per block can be expressed as `UBSIZE/6`:
- Atlas A2/A3 series: `192KB / 6 = 32KB`, corresponding to `32KB / 2B = 16384` `half` elements
- Ascend 950 series: `256KB / 6 ≈ 42.67KB`, corresponding to approximately `21840` `half` elements

Therefore on Ascend 950 series, based on capacity limit estimation, single block `dataCopyLen` can be increased to approximately `21840` `half` data.
In actual implementation, it is recommended that users align transfer granularity and computation granularity to 512B for more stable performance, so case3-5 use `dataCopyLen=21760`, corresponding to `21760 * 2B = 43520B = 85 * 512B` bytes.

On Ascend 950 series products, BANK arrangement is different, and this sample does not need to consider Bank conflicts for now, therefore only performance data for case 0-5 is listed. This processor processes 128 half data per cycle with a main frequency of 1.65GHz and AIV core count of 64, therefore theoretical vector duration is
$$
T_{\text{theory}} = \frac{8192 \times 8192}{128 \times 1.65 \times 10^9 \times 64} = \frac{67108864}{1.35168 \times 10^{13}} \approx 4.965 \times 10^{-6} \text{ s} = 4.965 \text{ μs}
$$

Reasons why Ascend 950 series did not reach theoretical vector duration peak:

- This theoretical value only counts "pure Vector computation" time, not including data transfer, event synchronization, pipeline scheduling and other overheads; while in measured Case 5, `aiv_vec_time=10.997μs` and Task Duration is `182.04μs`, `aiv_mte2_reatio` is as high as 98.3%, limited by bandwidth constraints, indicating end-to-end bottleneck is not in pure computation itself.
- Ascend 950 series RegBase's main benefit comes from "reducing redundant Load/Store, register reuse, increasing independent concurrent instruction ratio". Current Add sample's core computation is basically single Add, computation chain is short, with limited room for fusion/dual-issue, making it difficult to fully realize RegBase advantages.

### Optimization Points Summary

| Optimization Method | Core Principle | Applicable Scenarios |
|:---|:---:|:---|
| Scalar→Vector | Vector instructions process multiple elements in parallel | All computation-intensive operators |
| Single-core→Multi-core | Multi-core parallelism, load balancing | Large data volume scenarios |
| Increase transfer granularity | Reduce transfer count, amortize startup overhead | Transfer-intensive scenarios |
| Double buffer | Pipeline parallelism between transfer and computation | Computation and transfer times are similar |
| L2 Cache bypass | Avoid Cache pollution, reduce overhead | Streaming access (read only once) |
| Bank Conflict optimization | Optimize memory layout, avoid memory access conflicts | Vector-bound scenarios |

---

## Build and Run

- Switch Case

  Specify the case to compile during cmake build with `-DSCENARIO=N`:

  ```bash
  cmake -DSCENARIO=6 -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..   # Compile case 6 (can be replaced with 0-6)
  ```

  Case descriptions:
  - `0`: Single-core scalar version
  - `1`: Single-core vector version
  - `2`: Multi-core even splitting (dataCopyLen=4096)
  - `3`: Multi-core even splitting (large block data transfer)
  - `4`: Double buffer optimization
  - `5`: Double buffer + L2Cache bypass
  - `6`: Double buffer + L2Cache bypass + Bank Conflict avoidance

- Configure environment variables  
  Select the appropriate environment variable configuration command based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package in the current environment.
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
  mkdir -p build && cd build;   # Create and enter build directory
  cmake -DSCENARIO_NUM=6 -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;  # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                           # Execute (using case specified at compile time)
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output correctness, confirm algorithm logic is correct
  ```

  When using CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:
  ```bash
  cmake -DSCENARIO_NUM=6 -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DSCENARIO_NUM=6 -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean cmake cache by executing `rm CMakeCache.txt` in the build directory, then re-run cmake.
  

- Build option description
  | Option | Possible Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  | `SCENARIO_NUM` | `0` (default), `1`, `2`, `3`, `4`, `5`, `6` | Performance optimization case number |
  
  Execution result shown below indicates accuracy comparison passed.
  ```bash
  error ratio: 0.0000, tolerance:0.0001
  test pass!
  ```

## Performance Analysis

Use `msprof` tool to obtain detailed performance data:

```bash
msprof ./demo   # Analyze performance
```

A folder with PROF_ prefix will be generated in the current directory. The `mindstudio_profiler_output` directory stores performance data summary for Host and each Device. For performance data analysis, it is recommended to view files in this directory

```bash
PROF_xxxx_XXXXXX
├── device_{id}
└── host
└── mindstudio_profiler_log
└── mindstudio_profiler_output    # Stores performance data summary for Host and each Device
    ├── msprof_*.json
    ├── xx_*.csv
    └── README.txt
```
View specific performance analysis results:
```
# View Task Duration and various data
cat ./PROF_*/mindstudio_profiler_output/op_summary_*.csv
```