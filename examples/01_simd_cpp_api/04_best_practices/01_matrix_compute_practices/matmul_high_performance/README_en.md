# Matmul Best Practices Sample

## Overview

This sample implements matrix multiplication based on Matmul high-level API, demonstrating a complete optimization path from basic implementation to high-performance optimization through 9 progressive cases, including single-core basic version, Tiling optimization, multi-core parallel splitting, MDL mode, L1Cache/L2Cache optimization, constant Tiling, UnitFlag optimization, and other optimization methods.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── matmul_high_performance
│   ├── scripts
│   │   ├── gen_data.py         // Input data and golden data generation script file
│   │   └── verify_result.py    // Golden value comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   ├── matmul.h                // Header file definitions for all optimization cases
│   └── matmul.asc              // Ascend C sample implementation
```

## Sample Description

- Computation Formula: C = A * B
  - A, B are source operands, A is the left matrix with shape [M, K]; B is the right matrix with shape [K, N]
  - C is the destination operand, storing the matrix multiplication result with shape [M, N]

- Sample Specification:

<table>
<tr><td rowspan="1" align="center">Sample Type(OpType)</td><td colspan="5" align="center">Matmul</td></tr>
<tr><td rowspan="3" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
<tr><td align="center">A</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
<tr><td align="center">B</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">ND</td><td align="center">true</td></tr>
<tr><td rowspan="1" align="center">Sample Output</td><td align="center">C</td><td align="center">[M, N]</td><td align="center">half</td><td align="center">ND</td><td align="center">-</td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_custom</td></tr>
</table>


## Sample Implementation

### Class Implementation Explanation

This sample implements different optimization strategies through three independent classes, each corresponding to specific Case versions.

| Class Name | Corresponding Case | Implementation Features | Kernel Function Used | Optimization Features |
|------|---------|---------|-------------|---------|
| **MatmulKernel** | Case 0-5 | Basic implementation, runtime Tiling | matmul_custom (isMdl=false)<br>matmul_custom_mdl (isMdl=true) | - Case 0: Single-core basic version<br>- Case 1: Single-core Tiling optimization<br>- Case 2: Multi-core splitting 2x12<br>- Case 3: Multi-core splitting 4x6<br>- Case 4: MDL mode<br>- Case 5: MDL mode with L1Cache optimization |
| **MatmulKernelL2Cache** | Case 6 | L2Cache optimization, runtime Tiling | matmul_custom_mdl_l2cache | - MDL mode<br>- L1Cache optimization<br>- L2Cache optimization (A matrix M-axis splitting) |
| **MatmulKernelMdlL2CacheConstant** | Case 7-8 | Constant Tiling, compile-time computation | matmul_custom_mdl_l2cache_constant (useUnitFlag=false, Case7)<br>matmul_custom_mdl_l2cache_constant_unitflag (useUnitFlag=true, Case8) | - MDL mode<br>- L1Cache optimization<br>- L2Cache optimization<br>- Constant Tiling (compile-time computation)<br>- Case 8: UnitFlag optimization |

#### 1. Tiling Mechanism Features

- `MatmulKernel` and `MatmulKernelL2Cache` use `TCubeTiling` type, requiring copying the complete Tiling data structure from GM memory, with Tiling parameters calculated by Scalar unit at runtime
- `MatmulKernelMdlL2CacheConstant` uses custom `MatmulProblemShape` structure, containing only shape information (M, N, K, singleCoreM, and so on), with Tiling parameters already computed through `CONSTANT_CFG` at compile time, no Scalar computation required at runtime

#### 2. Process Method Computation Flow Features
- **Computation Flow**: `MatmulKernel` single iteration, `MatmulKernelL2Cache` and `MatmulKernelMdlL2CacheConstant` loop 2 times (L2Cache optimization, A matrix M-axis splitting)

---

### Performance Metric Description

| Metric | Description |
|------|------|
| Task Duration(μs) | Total execution time of the entire task, operator execution time should be based on this parameter |
| Block Num | Number of cores used (Block count) |
| aicore_time(μs) | Average execution time of AI Core |
| aic_mac_time(μs) | Execution time of Cube computation unit |
| aic_mac_ratio | Time ratio of Cube computation unit, reflecting computation unit utilization |
| aic_scalar_time(μs) | Execution time of Scalar scalar computation unit |
| aic_scalar_ratio | Time ratio of Scalar scalar computation unit |
| aic_mte1_time(μs) | Execution time of MTE1 (L1 to L0A/L0B transfer) |
| aic_mte1_ratio | Time ratio of MTE1, reflecting L1 to L0 data transfer pressure |
| aic_mte2_time(μs) | Execution time of MTE2 (GM to L1 transfer) |
| aic_mte2_ratio | Time ratio of MTE2, reflecting GM to L1 data loading pressure |
| aic_fixpipe_time(μs) | Execution time of FixPipe (L0C to GM transfer) |
| aic_fixpipe_ratio | Time ratio of FixPipe, reflecting result write-back memory access pressure |

---

**Note**: The performance change analysis for each Case below uses A2 chip (Ascend 910B1) performance data as an example. For Ascend 950PR performance tuning data, refer to [below](#ascend-950pr-chip-performance-data).

### Case 0: Single-core Basic Version (SINGLE_CORE_BASIC)

**Sample Objective**: Implement basic Matmul functionality, ensure computation correctness

**Core Implementation**:
- Use single-core computation, **numBlocks=1**
- Basic Tiling configuration, **baseM=baseN=baseK=64**, base block is the basic block participating in one matrix multiplication instruction, baseM, baseN, baseK represent L0 M, N, K axis lengths during Matmul computation respectively, measured in elements
- Transfer strategy: Input A, B matrices on GM are transferred block by block in baseM×baseK, baseK×baseN order to L1, then from L1 to L0A/L0B, Cube unit executes one baseM×baseN×baseK size Matmul computation for each transferred base block

**Key Code**:
```cpp
tilingApi.SetShape(M, N, K);
tilingApi.SetFixSplit(64, 64, 64);
tilingData.set_baseK(64);
```

**Performance Data**:
| Task Duration(μs) | Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio |
|------------------|-----------|----------------|-----------------|---------------|-------------------|-----------------|------------------|----------------|------------------|----------------|--------------------|-------------------|
| 759363.98 | 1 | 759363.46 | 141699.459 | 0.187 | 508787.509 | 0.67 | 162104.184 | 0.213 | 582515.828 | 0.767 | 34303.063 | 0.045 |

**Analysis**:
- Single-core operation time **759363.98μs**
- Computation unit utilization aic_mac_ratio only **18.7%**, this scenario serves only as a Matmul performance comparison sample, not recommended for user use
---

### Case 1: Single-core Tiling Optimization (SINGLE_CORE_TILING)

**Optimization Objective**: Optimize Tiling base block parameters, improve single-core computation efficiency

**Core Implementation**:
- Use single-core computation, **numBlocks=1**
- Use optimized Tiling configuration, **baseM=128, baseN=256, baseK=64**
- Transfer strategy: Input A, B matrices on GM are transferred block by block in baseM×baseK, baseK×baseN order to L1, then from L1 to L0A/L0B, Cube unit executes one baseM×baseN×baseK size Matmul computation for each transferred base block

**Key Code**:
```cpp
tilingApi.SetShape(M, N, K);
tilingApi.SetFixSplit(128, 256, 64);
```

**Optimization Methods**:
- **Base Block Selection Principle**: In case0, the base block set in Tiling is [baseM, baseN, baseK] = [64, 64, 64], with high **memory-computation ratio** (meaning data amount needed per cycle for computation). For scenarios with large current shape, the base block selection principle is minimum memory-computation ratio, meaning with the same Cube computation amount, the minimum data amount needed for memory access.

- **Memory-Computation Ratio Analysis**: With fp16 type input, Cube execution unit can complete 16×16×16 multiply-add operations per cycle. Setting base block as [baseM, baseN, baseK] = [128, 256, 64] can achieve minimum memory-computation ratio while satisfying 512Byte alignment for GM address during transfer out. Cube computation cycle count is (128 × 64 × 256) / (16 × 16 × 16) = 512cycle, memory-computation ratio is (128 × 64 × 2 + 256 × 64 × 2) / 512cycle = **96(byte / cycle)**; setting base block as [baseM, baseN, baseK] = [64, 64, 64], Cube computation cycle count is (64 × 64 × 64) / (16 × 16 × 16) = 64cycle, memory-computation ratio is (64 × 64 × 2 + 64 × 64 * 2) / 64cycle = **256(byte / cycle)**. The [128, 256, 64] base block scheme has **lower memory-computation ratio, meaning for the same computation amount, less data is needed, thus lower bandwidth pressure**

- 💡**Recommended Base Block Setting**: A2/A3 chips have L0A size equal to L0B size at 64KB, L0C size at 128KB, [baseM, baseN, baseK] = [128, 256, 64] can maximize memory space utilization. `For b16 input data type, recommended base block is [baseM, baseN, baseK] = [128, 256, 64]; for b8 input data type, recommended base block is [baseM, baseN, baseK] = [128, 256, 128].`



**Performance Data**:
| Task Duration(μs) | Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio |
|------------------|-----------|----------------|-----------------|---------------|-------------------|-----------------|------------------|----------------|------------------|----------------|--------------------|-------------------|
| 249467.08 | 1 | 249466.54 | 81477.188 | 0.327 | 63608.818 | 0.255 | 52003.703 | 0.208 | 195613.432 | 0.784 | 11313.746 | 0.045 |

**Analysis**:
- Compared to Case 0, Task Duration reduced from 759363.98μs to 249467.08μs, reduced by **509896.90μs**, performance improved **3.04x**
- MTE2 data transfer time reduced from 582515.828μs to 195613.432μs, reduced by **66.42%**
- Future optimization direction: Introduce multi-core parallel computation, fully utilize multi-core resources to improve overall throughput

---

### Case 2: Multi-core Splitting 2x12 (MULTI_CORE_SPLIT_2_12)

**Optimization Objective**: Introduce multi-core parallel computation, improve overall throughput

**Core Implementation**:
- Multi-core parallel computation, split 8192×8192 matrix multiplication to 24 cores for parallel execution
- Splitting strategy: M direction 2 blocks, N direction 12 blocks, **singleM=4096, singleN=683, tail block tailN=679**
- Transfer strategy: Input A, B matrices on GM are transferred block by block in baseM×baseK, baseK×baseN order to L1, then from L1 to L0A/L0B, Cube unit executes one baseM×baseN×baseK size Matmul computation for each transferred base block
<img src="figure/2_12_split_core.png">

**Key Code**:
```cpp
tilingApi.SetDim(24);
tilingApi.SetSingleShape(4096, 683, 8192);
tilingApi.SetFixSplit(128, 256, 64);
SetL1(tilingData);
```

**Optimization Methods**:
- Balanced load distribution, try to make each core's computation amount similar

**Performance Data**:
| Task Duration(μs) | Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio |
|------------------|-----------|----------------|-----------------|---------------|-------------------|-----------------|------------------|----------------|------------------|----------------|--------------------|-------------------|
| 12541.22 | 24 | 12537.56 | 3462.802 | 0.276 | 2987.47 | 0.238 | 2260.551 | 0.18 | 10177.798 | 0.812 | 875.439 | 0.07 |

**Analysis**:
- Compared to Case 1, Task Duration reduced from 249467.08μs to 12541.22μs, performance improved **19.89x**, multi-core parallel computation significantly improved overall throughput
- aic_mte2_time is **10177.798μs**, accounting for **81.2%**, becoming the bottleneck for performance improvement
- Future optimization direction: Current multi-core splitting strategy does not satisfy 512B address alignment, and does not evenly split cores for M, N, future will optimize splitting strategy, improve address alignment, increase memory access efficiency

---

### Case 3: Multi-core Splitting 4x6 (MULTI_CORE_SPLIT_4_6)

**Optimization Objective**: Optimize multi-core splitting strategy

**Core Implementation**:
- Multi-core parallel computation, split 8192×8192 matrix multiplication to 24 cores for parallel execution
- Splitting strategy: Evenly split M, N, M direction 4 blocks, N direction 6 blocks
- Transfer strategy: Input A, B matrices on GM are transferred block by block in baseM×baseK, baseK×baseN order to L1, then from L1 to L0A/L0B, Cube unit executes one baseM×baseN×baseK size Matmul computation for each transferred base block

<img src="figure/4_6_split_core.png">

**Key Code**:
```cpp
tilingApi.SetDim(24);
tilingApi.SetSingleShape(2048, 1536, 8192);
tilingApi.SetFixSplit(128, 256, 64);
SetL1(tilingData);
```

**Optimization Methods**:
- Address satisfies **512B alignment**, set **singleM=2048, singleN=1536, tail block tailN=512**, improve effective bandwidth utilization
- **Avoid same address access**: Same address access means when multiple cores simultaneously read the same row data of A matrix, they need to access the same memory address. Hardware needs to serialize multiple accesses to the same address. The more cores with same address access, the more severe the performance degradation from serialization. Compared with the 2×12 core splitting strategy in case2, case3's 4×6 core splitting has smaller same address conflict latency
- **When doing multi-core splitting, should evenly split M, N while satisfying 512B address alignment.**


**Performance Data**:
| Task Duration(μs) | Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio |
|------------------|-----------|----------------|-----------------|---------------|-------------------|-----------------|------------------|----------------|------------------|----------------|--------------------|-------------------|
| 12283.84 | 24 | 10870.96 | 3394.884 | 0.312 | 2656.33 | 0.244 | 2166.824 | 0.199 | 8616.671 | 0.793 | 488.829 | 0.045 |

**Analysis**:
- Compared to Case 2, Task Duration reduced from 12541.22μs to 12283.84μs, reduced by 2.05%.
- aic_mte2_time reduced from 10177.798μs to 8616.671μs, reduced by **15.33%**, decrease of **1561.127μs**
- Future optimization direction: Current aicore time is still mainly MTE2, future will enable L1 multi-block cache functionality, hide data transfer latency, improve MTE2 pipeline efficiency

---

### Case 4: Multi-core Using MDL Template (MULTI_CORE_MDL)

**Optimization Objective**: Use MDL template, enable L1 multi-block cache functionality, enable "large packet" transfer, reduce MTE2 loop transfer count

**Core Implementation**:
- Use 24-core parallel computation, splitting strategy same as above (4x6)
- Use MDL template, enable "large packet" transfer functionality

**Key Code**:
```cpp
AscendC::Matmul<AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>,
                AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>,
                AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>,
                AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>, CFG_MDL>
    matmulObj;
// MDL template auto optimization, enable large packet transfer functionality
```

**Optimization Methods**:
- Enable MDL mode, support "large packet" transfer
- **Large Packet Transfer**: MTE2 transfer from GM to L1 no longer transfers only one basic block at a time, but caches multiple basic blocks in L1, which can significantly reduce GM to L1 transfer count. In this example scenario, depthA1=4 and enables L1 transfer double buffer, meaning L1 caches 4 copies of baseM × baseK data blocks, transferring two blocks into ping and pong buffers each time during transfer.
- 💡**L1 Multi-block Cache Tuning Parameters Should Satisfy:**
  - dbL0A / dbL0B=2
  - depthA1 / (stepM * stepKa)=2,
  - depthB1 / (stepN * stepKb)=2
  - **Parameter Meaning:**
    - dbL0A, dbL0B: respectively indicate whether A matrix, B matrix MTE1 enables double buffer (value 1 or 2, 2 means double buffer enabled)
    - depthA1, depthB1: respectively indicate the number of fully loaded base blocks in L1
    - stepM, stepKa: respectively indicate the multiple of baseM for A matrix cached data block in M direction in L1, and the multiple of baseK in Ka direction (A matrix K direction)
    - stepN, stepKb: respectively indicate the multiple of baseN for B matrix cached data block in N direction in L1, and the multiple of baseK in Kb direction (B matrix K direction)

**Performance Data**:
| Task Duration(μs) | Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio |
|------------------|-----------|----------------|-----------------|---------------|-------------------|-----------------|------------------|----------------|------------------|----------------|--------------------|-------------------|
| 5039.86 | 24 | 4487.89 | 3116.472 | 0.694 | 2073.97 | 0.462 | 2332.914 | 0.52 | 4051.458 | 0.903 | 44.749 | 0.01 |

**Analysis**:
- Compared to Case 3, Task Duration reduced from 12283.84μs to 5039.86μs, reduced by **58.98%**
- aic_mte2_time reduced from 8616.671μs to 4051.458μs, MTE2 transfer time reduced by **52.97%**, decrease of **4565.213μs**
- aic_mac_ratio improved from **31.2%** to **69.4%**
- Future optimization direction: MDL template auto-tuned Tiling does not fully utilize L1 space, can manually adjust Tiling parameters to further increase MTE2 transfer intensity

---

### Case 5: Multi-core MDL + L1Cache Optimization (MULTI_CORE_MDL_L1CACHE)

**Optimization Objective**: Increase MTE2 transfer intensity, fully utilize L1 cache space

**Core Implementation**:
- Use 24-core parallel computation, splitting strategy same as above (4x6)
- Use MDL template, enable "large packet" transfer functionality
- Manually optimize Tiling parameters to **depthA1=16, stepKa=8**, fully utilize L1 cache space

**Key Code**:
```cpp
AscendC::Matmul<AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>,
                AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>,
                AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>,
                AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>, CFG_MDL>
    matmulObj;
tilingData.set_depthA1(16);  // Increase depthA1, increase A matrix block count in L1 cache
tilingData.set_stepKa(8);
```

**Optimization Methods**:
- **L1 Cache Tiling Parameter Optimization**: Manually adjust depthA1 and stepKa parameters, fully utilize L1 cache
  - Current configuration: depthA1=16, depthB1=8, baseM=128, baseN=256, baseK=64
  - L1 cache occupancy calculation:

    A matrix:
      depthA1 × baseM × baseK × sizeof(half) = 16 × 128 × 64 × 2B = 262,144B = 256 KB

    B matrix:
      depthB1 × baseN × baseK × sizeof(half) = 8 × 256 × 64 × 2B = 262,144B = 256 KB

    Total 256 KB + 256 KB = 512 KB

**Performance Data**:
| Task Duration(μs) | Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio |
|------------------|-----------|----------------|-----------------|---------------|-------------------|-----------------|------------------|----------------|------------------|----------------|--------------------|-------------------|
| 4156.76 | 24 | 3925.61 | 3352.087 | 0.854 | 1464.492 | 0.373 | 2750.454 | 0.701 | 3778.555 | 0.963 | 47.853 | 0.012 |

**Analysis**:
- Compared to Case 4, Task Duration reduced from 5039.86μs to 4156.76μs, reduced by **17.52%**
- aic_mte2_time reduced from 4051.458μs to 3778.555μs, MTE2 transfer time reduced by **6.47%**
- aic_mac_ratio improved from 69.4% to 85.4%, cube utilization improved by **23.05%**
- Future optimization direction: aic_mte2_ratio reaches **96.3%**, reaching MTE2 bound, becoming performance improvement bottleneck, future will optimize L2Cache to alleviate MTE2 bound
---

### Case 6: Multi-core MDL + L1Cache + L2Cache (MULTI_CORE_MDL_L1CACHE_L2CACHE)

**Optimization Objective**: Enable L2Cache optimization, alleviate MTE2 bound

**Core Implementation**:
- Use 24-core parallel computation, splitting strategy same as above (4x6)
- Use MDL template and optimize L1 Cache parameters same as above
- Enable `L2Cache`, split A matrix M-axis

**Key Code**:
```cpp
for (int i = 0; i < 2; i++) {
    matmulObj.SetTensorA(aGlobal[offsetA + i * (M >> 1) * K], IS_TRANS_A);
    matmulObj.SetTensorB(bGlobal[offsetB], IS_TRANS_B);
    if (shapes.isBias) {
        matmulObj.SetBias(biasGlobal);
    }
    matmulObj.IterateAll(cGlobal[offsetC + i * (M >> 1) * N]);
}
```

**Optimization Methods**:
- **Enable L2Cache Optimization**:
  - L2Cache characteristics: L2Cache is AI Core shared external cache, L2Cache pure read bandwidth is approximately 3 to 4 times GM. When transferring in or out the same data amount, accessing data in L2Cache is faster than GM
  - Cache hit optimization: If data cannot hit L2Cache, causing need to access GM for read/write, bandwidth utilization efficiency is lower, causing MTE2 to possibly become the performance bottleneck in the entire sample execution process
  - **Block Computation Adapts to L2Cache**: Current L2Cache size is **192MB**, matrix computation required total data is **384MB** (A matrix 128MB + B matrix 128MB + C matrix 128MB). Since L2Cache capacity is smaller than matrix computation data total, can split A matrix into 2 parts in M-axis, where region 1 completes full computation with B matrix, then region 2 completes full computation with B matrix, improving L2Cache hit rate through splitting

Block computation adapting to L2Cache diagram:

<img src="figure/L2Cache.png">

```
Computation Process:
Step 1: C1 = A1 × B  (A1 loaded from GM to L2Cache, B loaded from GM to L2Cache and stays)
Step 2: C2 = A2 × B  (A2 loaded from GM to L2Cache, B already in L2Cache, no need to reload)

L2Cache Utilization:
- B matrix (128MB) stays in L2Cache after loaded in Step 1
- In Step 2, B matrix reads directly from L2Cache, avoiding GM access
- Single L2Cache access bandwidth is approximately 3 to 4 times GM, significantly improving memory access efficiency
```

**Performance Data**:
| Task Duration(μs) | Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio |
|------------------|-----------|----------------|-----------------|---------------|-------------------|-----------------|------------------|----------------|------------------|----------------|--------------------|-------------------|
| 4088.36 | 24 | 3786.26 | 3254.31 | 0.86 | 1753.463 | 0.463 | 2680.849 | 0.708 | 3625.42 | 0.958 | 47.398 | 0.013 |

**Analysis**:
- Compared to Case 5, Task Duration reduced from 4156.76μs to 4088.36μs, reduced by **68.40μs**
- aic_mte2_time reduced from 3778.555μs to 3625.42μs, MTE2 transfer time reduced by **4.05%**
- aic_mac_ratio improved from **85.4%** to **86%**
- Future optimization direction: Current constraint on sample performance is still MTE2 bound, developers can further optimize MTE2 by optimizing L2Cache splitting strategy. This sample will subsequently demonstrate Scalar and Fixpipe pipeline optimization

---

### Case 7: Multi-core MDL + L1Cache + L2Cache + Constants Tiling (MULTI_CORE_MDL_L1CACHE_L2CACHE_CONSTANTS)

**Optimization Objective**: Use constant Tiling, reduce runtime Scalar computation overhead

**Core Implementation**:
- Use 24-core parallel computation, splitting strategy same as above (4x6)
- Use MDL template and optimize L1 Cache parameters same as above
- L2 Cache strategy same as above
- Enable Tiling full constantization

**Key Code**:
```cpp
constexpr MatmulShapeParams shapeParams = {SINGLE_M_L2CACHE, SINGLE_N, SINGLE_K, BASE_M, BASE_N, BASE_K};

constexpr static auto CONSTANT_CFG = GetCustomConstantCFG<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, true, true>();
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CONSTANT_CFG> matmulObj;
// CONSTANT_CFG computed at compile time, reduces runtime Scalar computation
```

**Optimization Methods**:
- **Tiling Constantization**:
  - AI Core hardware advantage lies in vector/matrix parallel computation, scalar computation cannot leverage hardware capability. **Matmul API has large Scalar computation during initialization and iteration, Matmul initialization Scalar computation affects instruction header overhead, Matmul iteration Scalar computation may block MTE2 pipeline**
  - Static Tiling reduces runtime overhead: Use **MatmulApiStaticTiling** parameter instead of TCubeTiling variable parameter, moving Scalar computation to compile time, no need for Scalar unit dynamic computation at runtime, reducing Scalar performance overhead

**Performance Data**:
| Task Duration(μs) | Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio |
|------------------|-----------|----------------|-----------------|---------------|-------------------|-----------------|------------------|----------------|------------------|----------------|--------------------|-------------------|
| 4053.44 | 24 | 3665.12 | 3163.682 | 0.863 | 968.616 | 0.264 | 2609.617 | 0.712 | 3513.806 | 0.959 | 45.76 | 0.012 |

**Analysis**:
- Compared to Case 6, Task Duration reduced from 4088.36μs to 4053.44μs
- aic_scalar_time reduced from 1753.463μs to 968.616μs, reduced by **44.76%**
- Constant Tiling effectively reduced Scalar unit performance overhead. When sample has poor performance due to Scalar blocking, users can reduce Scalar time ratio through this method
- Future optimization direction: Enable UnitFlag optimization, demonstrate parallelized computation and transfer pipeline

---

### Case 8: Multi-core MDL + L1Cache + L2Cache + Constants Tiling + UnitFlag (MULTI_CORE_MDL_L1CACHE_L2CACHE_CONSTANTS_UNITFLAG)

**Optimization Objective**: Enable UnitFlag optimization, parallelize computation and transfer pipeline

**Core Implementation**:
- Use 24-core parallel computation, splitting strategy same as above (4x6)
- Use MDL template and optimize L1 Cache parameters same as above
- L2 Cache strategy same as above
- Enable Tiling full constantization same as above
- Enable UnitFlag, optimize computation and transfer parallelism

**Key Code**:
```cpp
// Use constexpr to define compile-time constants, and enable UnitFlag
MatmulConfig mmCFG = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(shapeParams);
mmCFG.enUnitFlag = true;
```

**Optimization Methods**:
- **Enable UnitFlag Optimization**:
  - Optimize computation and transfer parallelism: When UnitFlag functionality is not enabled, AIC core's MMAD computation instruction and FIXPIPE data transfer instruction are instruction-level synchronized. FIXPIPE instruction needs to wait for MMAD instruction to complete before transferring results out, MMAD and FIXPIPE pipelines are serial; after enabling UnitFlag functionality, MMAD and FIXPIPE are **512B size fine-grained synchronized**. During one MMAD instruction execution, whenever 512B data result computation completes, FIXPIPE immediately transfers out that 512B data, achieving Cube computation unit and FIXPIPE transfer unit pipeline parallelism

UnitFlag functionality diagram:

<img src="figure/unitflag_close.png">
<img src="figure/unitflag_open.png">



**Performance Data**:
| Task Duration(μs) | Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio |
|------------------|-----------|----------------|-----------------|---------------|-------------------|-----------------|------------------|----------------|------------------|----------------|--------------------|-------------------|
| 4012.44 | 24 | 3559.06 | 3076.396 | 0.864 | 1026.069 | 0.288 | 2533.512 | 0.712 | 3435.584 | 0.965 | 272.576 | 0.077 |

**Analysis**:
- Compared to Case 7, Task Duration reduced from 4053.44μs to 4012.44μs
- aic_fixpipe_time increased from 45.76μs to 272.576μs, reason is after enabling Unitflag, aic_fixpipe_time includes FIXPIPE instruction wait time, which is not the actual pipeline duration. Users can focus on whether end-to-end performance improves
- Current performance gain is small because the performance constraint is still MTE2 bound, operator's MMAD, FIXPIPE pipeline is masked by MTE2 bound. When sample has poor performance due to FIXPIPE blocking, users can enable UnitFlag functionality

---

## Performance Comparison Summary

### Atlas A2 Training Series Chip Performance Data
**Comprehensive Optimization Effect**:
- This sample cube utilization improved by **67.7%** (18.7% → 86.4%), reaching **86.4%** of chip peak computing power
- Through Case 0 to Case 8 progressive optimization, sample duration reduced by **99.47%** (759363.98μs → 4012.44μs)

| Case version | Task Duration(μs) | End-to-end Duration Relative to Case 0 | Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio |
|------|------------------|----------------------|-----------|----------------|-----------------|---------------|-------------------|-----------------|------------------|----------------|------------------|----------------|--------------------|-------------------|
| Case 0 | 759363.98 | **1x** | 1 | 759363.46 | 141699.459 | 0.187 | 508787.509 | 0.67 | 162104.184 | 0.213 | 582515.828 | 0.767 | 34303.063 | 0.045 |
| Case 1 | 249467.08 | **3.04x** | 1 | 249466.54 | 81477.188 | 0.327 | 63608.818 | 0.255 | 52003.703 | 0.208 | 195613.432 | 0.784 | 11313.746 | 0.045 |
| Case 2 | 12541.22 | **60.55x** | 24 | 12537.56 | 3462.802 | 0.276 | 2987.47 | 0.238 | 2260.551 | 0.18 | 10177.798 | 0.812 | 875.439 | 0.07 |
| Case 3 | 12283.84 | **61.82x** | 24 | 10870.96 | 3394.884 | 0.312 | 2656.33 | 0.244 | 2166.824 | 0.199 | 8616.671 | 0.793 | 488.829 | 0.045 |
| Case 4 | 5039.86 | **150.67x** | 24 | 4487.89 | 3116.472 | 0.694 | 2073.97 | 0.462 | 2332.914 | 0.52 | 4051.458 | 0.903 | 44.749 | 0.01 |
| Case 5 | 4156.76 | **182.68x** | 24 | 3925.61 | 3352.087 | 0.854 | 1464.492 | 0.373 | 2750.454 | 0.701 | 3778.555 | 0.963 | 47.853 | 0.012 |
| Case 6 | 4088.36 | **185.74x** | 24 | 3786.26 | 3254.31 | 0.86 | 1753.463 | 0.463 | 2680.849 | 0.708 | 3625.42 | 0.958 | 47.398 | 0.013 |
| Case 7 | 4053.44 | **187.34x** | 24 | 3665.12 | 3163.682 | 0.863 | 968.616 | 0.264 | 2609.617 | 0.712 | 3513.806 | 0.959 | 45.76 | 0.012 |
| Case 8 | 4012.44 | **189.25x** | 24 | 3559.06 | 3076.396 | 0.864 | 1026.069 | 0.288 | 2533.512 | 0.712 | 3435.584 | 0.965 | 272.576 | 0.077 |

**Theoretical Performance Comparison**:

Key parameters for Matmul theoretical performance evaluation: Cube computation performance and MTE2 bandwidth.

#### Cube Computation Performance Analysis

Sample parameters: M=N=K=8192, baseM=128, baseN=256, baseK=64. This sample performance data was tested on Atlas A2 Training Series Products, with computation chip main frequency of 1.85GHz, processing 16×16×16 multiply-add operations per cycle.

Cube theoretical computation time:
$$cube\_time = \frac{M \times N \times K}{16 \times 16 \times 16 \times core\_num \times cube\_freq} = \frac{8192 \times 8192 \times 8192}{16 \times 16 \times 16 \times 24 \times 1850} = 3022.92\mu s$$

Case 8 Cube computation duration error:
$$Error = \frac{aic\_mac\_time - cube\_time}{cube\_time} = \frac{{3076.396\mu s} - {3022.92\mu s}}{{3022.92\mu s}} = 1.77\%$$

Excluding startup overhead, has achieved 86% of this chip's peak computing power

#### MTE2 Bandwidth Analysis

Total data read:
$$Total data read = \left[\frac{N}{baseN} \times M \times K\right] + \left[\frac{M}{baseM} \times K \times N\right] \times dataType = (32 \times 8192 \times 8192) + (64 \times 8192 \times 8192) \times 2B = 12GB$$

Ideally assuming L2Cache capacity is large enough, first load data from HBM, subsequent data all read from L2Cache, L2Cache peak bandwidth is approximately 5TB/s, HBM bandwidth is approximately 1.8TB/s.
$$First time data read from HBM total = M \times K \times dataType + K \times N \times dataType = 256MB$$

MTE2 theoretical duration:
$$MTE2 theoretical duration =\frac{HBM read data total}{1.8TB/s} +\frac{L2Cache read data total}{5TB/s} = 2672.44\mu s$$

Case 8 MTE2 duration error:
$$MTE2 duration error = \frac{{3435.584\mu s} - {2672.44\mu s}}{{2672.44\mu s}} = 28.55\%$$

Current MTE2 duration differs significantly from theoretical value because actual chip L2Cache size is 192MB, current L2Cache splitting strategy is simple; on the other hand, when MTE2 transfer scenario is ND2NZ (GM data Layout is ND, transfer to L1 needs ND→NZ format conversion), L2Cache bandwidth decreases. Users can further optimize L2Cache splitting strategy to improve MTE2 bandwidth.



### Ascend 950PR Chip Performance Data
| Case version | Task Duration(μs) | End-to-end Duration Relative to Case 0 | Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio |
|------|------------------|----------------------|-----------|----------------|-----------------|---------------|-------------------|-----------------|------------------|----------------|------------------|----------------|--------------------|-------------------|
| Case 0 | 1096626.431 | **1x** | 1 | 1096625.66 | 198351.65 | 0.181 | 583195.222 | 0.532 | 115705.132 | 0.106 | 960571.993 | 0.876 | 28615.988 | 0.026 |
| Case 1 | 130560.475 | **8.40x** | 1 | 130559.56 | 88685.142 | 0.679 | 36462.067 | 0.279 | 22489.156 | 0.172 | 106793.342 | 0.818 | 4200.385 | 0.032 |
| Case 2 | 4294.619 | **255.35x** | 32 | 4293.9 | 2788.781 | 0.649 | 1149.24 | 0.268 | 707.592 | 0.165 | 3540.645 | 0.825 | 141.876 | 0.033 |
| Case 3 | 4332.557 | **253.11x** | 32 | 4331.82 | 2774.213 | 0.64 | 1143.276 | 0.264 | 703.896 | 0.162 | 3582.246 | 0.827 | 144.525 | 0.033 |
| Case 4 | 2668.224 | **410.99x** | 32 | 2667.49 | 2571.074 | 0.964 | 1377.736 | 0.516 | 799.378 | 0.3 | 2531.912 | 0.949 | 33.864 | 0.013 |
| Case 5 | 2591.366 | **423.18x** | 32 | 2590.51 | 2547.046 | 0.983 | 612.956 | 0.237 | 834.311 | 0.322 | 1926.358 | 0.744 | 35.44 | 0.014 |
| Case 6 | 2589.888 | **423.43x** | 32 | 2589.18 | 2547.518 | 0.984 | 765.125 | 0.296 | 826.429 | 0.319 | 1879.029 | 0.726 | 33.261 | 0.013 |
| Case 7 | 2589.09 | **423.55x** | 32 | 2588.38 | 2547.049 | 0.984 | 426.398 | 0.165 | 827.939 | 0.32 | 1895.165 | 0.732 | 33.648 | 0.013 |
| Case 8 | 2558.155 | **428.68x** | 32 | 2557.49 | 2549.657 | 0.997 | 412.29 | 0.161 | 835.579 | 0.327 | 1900.322 | 0.743 | 213.789 | 0.084 |

**Theoretical Performance Comparison**:
#### Cube Computation Performance Analysis

Sample parameters: M=N=K=8192, baseM=256, baseN=256, baseK=64. This sample performance data was tested on Ascend 950PR chip, with processor main frequency of 1.65GHz, processing 16×16×16 multiply-add operations per cycle.

Cube theoretical computation time:
$$cube\_time = \frac{M \times N \times K}{16 \times 16 \times 16 \times core\_num \times cube\_freq} = \frac{8192 \times 8192 \times 8192}{16 \times 16 \times 16 \times 32 \times 1650} = 2542\mu s$$

Case 8 Cube computation duration error:
$$Error = \frac{aic\_mac\_time - cube\_time}{cube\_time} = \frac{{2549.657\mu s} - {2542\mu s}}{{2542\mu s}} = 0.30\%$$

Has achieved 99.7% of this chip's peak computing power
#### MTE2 Bandwidth Analysis

Total data read:
$$Total data read = \left[\frac{N}{baseN} \times M \times K\right] + \left[\frac{M}{baseM} \times K \times N\right] \times dataType = (32 \times 8192 \times 8192) + (32 \times 8192 \times 8192) \times 2B = 8GB$$

Ideally assuming L2Cache capacity is large enough, first load data from HBM, subsequent data all read from L2Cache, L2Cache peak bandwidth is approximately 5TB/s, HBM bandwidth is approximately 1.6TB/s.
$$First time data read from HBM total = M \times K \times dataType + K \times N \times dataType = 256MB$$

MTE2 theoretical duration:
$$MTE2 theoretical duration =\frac{HBM read data total}{1.6TB/s} +\frac{L2Cache read data total}{5TB/s} = 1832.1\mu s$$

Case 8 MTE2 duration error:
$$MTE2 duration error = \frac{{1900.322\mu s} - {1832.1\mu s}}{{1832.1\mu s}} = 3.72\%$$
Compared to Atlas A2 Training Series chip, Ascend 950PR chip upgrade makes data transfer more efficient

## Tuning Suggestions

1. **Start from Small Scale**: First use single-core basic version to verify functionality correctness
2. **Gradual Optimization**: Introduce optimization methods step by step according to case sequence, observe performance improvement
3. **Multi-core Splitting Strategy**: Reasonably set multi-core splitting strategy, avoid same address access
4. **Utilize MDL Mode**: MDL mode provides highly optimized implementation, prioritize use
4. **L2Cache**: L2Cache can further alleviate MTE2 bound, recommended for data that needs repeated reading
6. **Constant Tiling**: For fixed shape scenarios, use constant Tiling to reduce Scalar runtime overhead
7. **UnitFlag Optimization**: Enable UnitFlag can parallelize computation and transfer pipeline

## Build and Run

Execute the following steps in the sample root directory to build and run the sample.

- Configure Environment Variables
  Select the corresponding environment variable configuration command based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package in the current environment.
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

- Sample Execution
  ```bash
  SCENARIO_NUM=0                       # Select execution scenario
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;  # Build project (default npu mode)
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                           # Execute
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output result correctness, confirm algorithm logic correctness
  ```

  When using NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;   # NPU simulation mode
  ```

  > **Note:** Before switching build mode, need to clear cmake cache. Can execute `rm CMakeCache.txt` in build directory and then re-run cmake.

- Build Option Description

  | Option | Available Values | Description |
  |------|--------|------|
  | `SCENARIO_NUM` | `0`-`8` | Sample type (0-8), default is 0 |
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `sim` | Run mode: NPU run, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

  The execution result shown below indicates the accuracy comparison succeeded.
  ```bash
  test pass!
  ```

## Performance Analysis

Use the `msprof` tool to obtain detailed performance data:

```bash
msprof ./demo   # Analyze case performance
```

A PROF_ prefixed folder will be generated in the current directory, with `mindstudio_profiler_output` directory storing Host and various Device performance data summary. Performance data analysis is recommended to view files in this directory
```bash
PROF_xxxx_XXXXXX
├── device_{id}
└── host
└── mindstudio_profiler_log
└── mindstudio_profiler_output    # Store Host and various Device performance data summary
    ├── msprof_*.json
    ├── xx_*.csv
    └── README.txt
```