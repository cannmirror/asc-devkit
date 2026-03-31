# Matmul最佳实践样例
## 概述
本样例基于Matmul 高阶API实现矩阵乘法运算，通过9个递进式的优化case展示从基础实现到高性能优化的完整调优路径，包括单核基础版本、Tiling优化、多核并行切分、MDL模式、L1Cache/L2Cache优化、常量Tiling、UnitFlag优化等多种优化手段。

## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
## 目录结构介绍
```
├── 05_matmul
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   ├── matmul.h                // 所有优化case的头文件定义
│   └── matmul.asc              // Ascend C样例实现
```

## 样例描述

- 计算公式：C = A * B
  - A、B为源操作数，A为左矩阵，形状为[M, K]；B为右矩阵，形状为[K, N]
  - C为目的操作数，存放矩阵乘结果的矩阵，形状为[M, N]

- 样例规格：
  <table border="2" align="center">
  <tr><td rowspan="1" align="center">样例类型(OpType)</td><td colspan="5" align="center">MatmulCustom</td></tr>
  <tr><td rowspan="3" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">[8192, 8192]</td><td align="center">float16</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">[8192, 8192]</td><td align="center">float16</td><td align="center">ND</td><td align="center">true</td></tr>
  <tr><td rowspan="1" align="center">样例输出</td><td align="center">c</td><td align="center">[8192, 8192]</td><td align="center">float16</td><td align="center">ND</td><td align="center">-</td></tr>
  </table>


## 样例实现

### 性能指标说明

| 指标 | 说明 |
|------|------|
| Task Duration(μs) | 整个任务执行的总时间，算子执行时间以该参数为准 |
| Block Num | 使用的核数（Block数量） |
| aicore_time(μs) | AI Core的平均执行时间 |
| aic_mac_time(μs) | Cube计算单元的执行时间 |
| aic_mac_ratio | Cube计算单元的时间占比，反映计算单元利用率 |
| aic_scalar_time(μs) | Scalar标量计算单元的执行时间 |
| aic_scalar_ratio | Scalar标量计算单元的时间占比 |
| aic_mte1_time(μs) | MTE1（L1到L0A/L0B搬运）的执行时间 |
| aic_mte1_ratio | MTE1的时间占比，反映L1到L0的数据搬运压力 |
| aic_mte2_time(μs) | MTE2（GM到L1搬运）的执行时间 |
| aic_mte2_ratio | MTE2的时间占比，反映GM到L1的数据加载压力 |
| aic_fixpipe_time(μs) | FixPipe（L0C到GM搬运）的执行时间 |
| aic_fixpipe_ratio | FixPipe的时间占比，反映结果写回的访存压力 |

---

### Case 0: 单核基础版本 (SINGLE_CORE_BASIC)

**样例目标**：实现基础的Matmul功能，确保计算正确性

**核心实现**：
- 使用单核计算，**numBlocks=1**
- 基础Tiling配置，**baseM=baseN=baseK=64**，base块为参与一次矩阵乘指令的基本块，baseM、baseN、baseK分别代表Matmul计算时L0上M、N、K轴长度，以元素为单位，
- 搬运策略：GM上输入的A、B矩阵按baseM×baseK、baseK×baseN分块依次搬运到L1中，再从L1搬运到L0A/L0B，Cube单元对每次搬运的base块执行一次baseM×baseN×baseK大小的Matmul计算

**关键代码**：
```cpp
tilingApi.SetShape(M, N, K);
tilingApi.SetFixSplit(64, 64, 64);
tilingData.set_baseK(64);
```

**性能数据**：
| Task Duration(μs) | Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio |
|------------------|-----------|----------------|-----------------|---------------|-------------------|-----------------|------------------|----------------|------------------|----------------|--------------------|-------------------|
| 759363.98 | 1 | 759363.46 | 141699.459 | 0.187 | 508787.509 | 0.67 | 162104.184 | 0.213 | 582515.828 | 0.767 | 34303.063 | 0.045 |

**分析**：
- 单核运算时间**759363.98μs**
- 计算单元利用率aic_mac_ratio仅**18.7%**，该场景仅作为Matmul运算性能对比样例，不建议用户使用
---

### Case 1: 单核Tiling优化 (SINGLE_CORE_TILING)

**优化目标**：优化Tiling中base块参数，提升单核计算效率

**核心实现**：
- 使用单核计算，**numBlocks=1**
- 采用优化的Tiling配置，**baseM=128, baseN=256, baseK=64**
- 搬运策略：GM上输入的A、B矩阵按baseM×baseK、baseK×baseN分块依次搬运到L1中，再从L1搬运到L0A/L0B，Cube单元对每次搬运的base块执行一次baseM×baseN×baseK大小的Matmul计算

**关键代码**：
```cpp
tilingApi.SetShape(M, N, K);
tilingApi.SetFixSplit(128, 256, 64);
```

**优化手段**：
- **base块选择原则**：case0中Tiling中设置的base块为 [baseM, baseN, baseK] = [64, 64, 64]，**计算访存比**（即计算量与需要数据量的比值）低。针对当前shape较大的场景，基本块的选择原则为计算访存比最大，即在Cube计算量最大的情况下，访存的数据量最小。

- **计算访存比分析**：在输入为fp16类型的情况下，Cube执行单元1 cycle能完成16 * 16 * 16次乘加运算。设置base块为[baseM, baseN, baseK] = [128, 256, 64]时，可以在满足搬出时GM地址512Byte对齐的同时，达到最大的计算访存比。Cube计算cycle数为(128 * 64 * 256) / (16 * 16 * 16) = 512cycle，访存计算比为(128 * 64 * 2 + 256 * 64 * 2) / 512cycle = **96byte / cycle**；设置base块为[baseM, baseN, baseK] = [64, 64, 64]时，Cube计算cycle数为(64 * 64 * 64) / (16 * 16 * 16) = 64cycle，计算访存比为(64 * 64 * 2 + 64 * 64 * 2) / 64cycle = **256byte / cycle**。[128, 256, 64]的base块方案的**访存计算比更低，同样的计算量，需要的数据量更小，所需带宽压力也就越低**

- 💡**推荐base块设置**：A2/A3 芯片L0A大小与L0B大小一致，为64KB，L0C大小为128KB，[baseM, baseN, baseK] = [128, 256, 64]时，能最大限度利用内存空间。`输入数据类型b16时，base块推荐[baseM, baseN, baseK] = [128, 256, 64]，输入数据类型b8时，base块推荐[baseM, baseN, baseK] = [128, 256, 128]。`



**性能数据**：
| Task Duration(μs) | Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio |
|------------------|-----------|----------------|-----------------|---------------|-------------------|-----------------|------------------|----------------|------------------|----------------|--------------------|-------------------|
| 249467.08 | 1 | 249466.54 | 81477.188 | 0.327 | 63608.818 | 0.255 | 52003.703 | 0.208 | 195613.432 | 0.784 | 11313.746 | 0.045 |

**分析**：
- 相比Case 0，Task Duration从759363.98μs降低到249467.08μs，降低了**509896.90μs**，性能提升**3.04**倍
- MTE2数据搬运耗时从582515.828μs降低到195613.432μs，降低**66.42%**
- 后续优化方向：引入多核并行计算，充分利用多核资源提升整体吞吐量

---

### Case 2: 多核切分 2x12 (MULTI_CORE_SPLIT_2_12)

**优化目标**：引入多核并行计算，提升整体吞吐量

**核心实现**：
- 多核并行计算，将8192×8192的矩阵乘法切分到24个核上并行执行
- 搬运策略：GM上输入的A、B矩阵按baseM×baseK、baseK×baseN分块依次搬运到L1中，再从L1搬运到L0A/L0B，Cube单元对每次搬运的base块执行一次baseM×baseN×baseK大小的Matmul计算

**关键代码**：
```cpp
tilingApi.SetDim(24);
tilingApi.SetSingleShape(4096, 683, 8192);
tilingApi.SetFixSplit(128, 256, 64);
SetL1(tilingData);
```

**优化手段**：
- 切分策略：M方向2块，N方向12块，**singleM=4096，singleN=683，尾块tailN=679**
- 均衡负载分配，尽量让每个核的计算量相近

**性能数据**：
| Task Duration(μs) | Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio |
|------------------|-----------|----------------|-----------------|---------------|-------------------|-----------------|------------------|----------------|------------------|----------------|--------------------|-------------------|
| 12541.22 | 24 | 12537.56 | 3462.802 | 0.276 | 2987.47 | 0.238 | 2260.551 | 0.18 | 10177.798 | 0.812 | 875.439 | 0.07 |

**分析**：
- 相比Case 1，Task Duration从249467.08μs降低到12541.22μs，性能提升**19.89倍**，多核并行计算显著提升了整体吞吐量
- aic_mte2_time耗时**10177.798μs**，占比**81.2%**，成为性能提升瓶颈
- 后续优化方向：当前多核切分策略没有满足地址512B对齐，且对M、N没有均匀分核，后续将优化切分策略，提升地址对齐，提高访存效率

---

### Case 3: 多核切分 4x6 (MULTI_CORE_SPLIT_4_6)

**优化目标**：优化多核切分策略

**核心实现**：
- 多核并行计算，将8192×8192的矩阵乘法切分到24个核上并行执行
- 切分策略：对M、N均匀切分，M方向4块，N方向6块
- 搬运策略：GM上输入的A、B矩阵按baseM×baseK、baseK×baseN分块依次搬运到L1中，再从L1搬运到L0A/L0B，Cube单元对每次搬运的base块执行一次baseM×baseN×baseK大小的Matmul计算

**关键代码**：
```cpp
tilingApi.SetDim(24);
tilingApi.SetSingleShape(2048, 1536, 8192);
tilingApi.SetFixSplit(128, 256, 64);
SetL1(tilingData);
```

**优化手段**：
- 地址满足**512B对齐**，设置**singleM=2048, singleN=1536，尾块tailN=512**，提高有效带宽利用率
- **避免同地址访问**：同地址访问指多个核同时读取A矩阵的同一行数据时，需要访问相同的内存地址，硬件对同一地址的多次访问需要串行化。同地址访问核数越多，串行导致的性能劣化越严重，相比于case2中2×12的分核策略，case3的4×6分核同地址冲突延迟更小
- **在多核切分时，应在满足地址512B对齐的情况下，对M、N进行均匀切分。**


**性能数据**：
| Task Duration(μs) | Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio |
|------------------|-----------|----------------|-----------------|---------------|-------------------|-----------------|------------------|----------------|------------------|----------------|--------------------|-------------------|
| 12283.84 | 24 | 10870.96 | 3394.884 | 0.312 | 2656.33 | 0.244 | 2166.824 | 0.199 | 8616.671 | 0.793 | 488.829 | 0.045 |

**分析**：
- 相比Case 2，Task Duration从12541.22μs降低到12283.84μs，降低了 2.05%。
- aic_mte2_time从10177.798μs降低到8616.671μs，降低**15.33%**，降幅**1561.127μs**
- 后续优化方向：当前aicore耗时仍主要为MTE2，后续将启用L1多块缓存功能，隐藏数据搬运延迟，提升MTE2流水线效率

---

### Case 4: 多核使用MDL模板 (MULTI_CORE_MDL)

**优化目标**：使用MDL模板，启用L1多块缓存功能，开启"大包"搬运，减少MTE2循环搬运次数

**核心实现**：
- 使用24核并行计算，切分策略同上（4x6）
- 使用MDL模板，启用"大包"搬运功能

**关键代码**：
```cpp
AscendC::Matmul<AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>,
                AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>,
                AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>,
                AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>, CFG_MDL>
    matmulObj;
// MDL模板自动优化，启用大包搬运功能
```

**优化手段**：
- 启用MDL模式，支持"大包"搬运
- **大包搬运**：MTE2从GM到L1的搬运不再是每次只搬运一个基本块，而是在L1中缓存多个基本块，可以显著减少GM到L1的搬运次数。本例场景中depthA1=4且开启L1搬运的double buffer，表示L1中缓存4份baseM × baseK的数据块，搬运时ping、pong缓冲区各搬入两块。
- 💡**L1多块缓存调优参数应满足：**
  - dbL0A / dbL0B=2
  - depthA1 / (stepM * stepKa)=2，
  - depthB1 / (stepN * stepKb)=2
  - **参数含义：**
    - dbL0A、dbL0B：分别表示 A 矩阵、 B 矩阵的 MTE1 是否开启 double buffer（值为 1 或 2，2 表示开启双缓冲）
    - depthA1、depthB1：分别表示 L1 中全载基本块的份数
    - stepM、stepKa：分别表示A矩阵在 L1 中缓存的数据块 M 方向上 baseM 的倍数、Ka方向即A矩阵K方向上 baseK 的倍数
    - stepN、stepKb：分别表示B矩阵在 L1 中缓存的数据块 N 方向上 baseN 的倍数、Kb方向即B矩阵K方向上 baseK 的倍数

**性能数据**：
| Task Duration(μs) | Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio |
|------------------|-----------|----------------|-----------------|---------------|-------------------|-----------------|------------------|----------------|------------------|----------------|--------------------|-------------------|
| 5039.86 | 24 | 4487.89 | 3116.472 | 0.694 | 2073.97 | 0.462 | 2332.914 | 0.52 | 4051.458 | 0.903 | 44.749 | 0.01 |

**分析**：
- 相比Case 3，Task Duration从12283.84μs降低到5039.86μs，耗时降低**58.98%**
- aic_mte2_time从8616.671μs降低到4051.458μs，MTE2搬运时间降低**52.97%**，降幅**4565.213μs**
- aic_mac_ratio从**31.2%**提升到**69.4%**
- 后续优化方向：MDL模板自动调优计算的的Tiling没有完全利用L1空间，可手动调整Tiling参数，进一步提高MTE2的搬运力度

---

### Case 5: 多核MDL + L1Cache优化 (MULTI_CORE_MDL_L1CACHE)

**优化目标**：提高MTE2的搬运力度，充分利用L1缓存空间

**核心实现**：
- 使用24核并行计算，切分策略同上（4x6）
- 使用MDL模板，启用"大包"搬运功能
- 手动优化Tiling参数为**depthA1=16, stepKa=8**，充分利用L1缓存空间

**关键代码**：
```cpp
AscendC::Matmul<AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>,
                AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>,
                AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>,
                AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>, CFG_MDL>
    matmulObj;
tilingData.set_depthA1(16);  // 增大depthA1，增加L1缓存中A矩阵的块数
tilingData.set_stepKa(8);
```

**优化手段**：
- **L1 Cache Tiling参数优化**：手动调整depthA1和stepKa参数，充分利用L1缓存
  - 当前配置：depthA1=16, depthB1=8, baseM=128, baseN=256, baseK=64
  - L1 缓存占用计算：
    
    A 矩阵：
      depthA1 × baseM × baseK × sizeof(float16) = 16 × 128 × 64 × 2B = 262,144B = 256 KB

    B 矩阵：
      depthB1 × baseN × baseK × sizeof(float16) = 8 × 256 × 64 × 2B = 262,144B = 256 KB

    总计256 KB + 256 KB = 512 KB

**性能数据**：
| Task Duration(μs) | Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio |
|------------------|-----------|----------------|-----------------|---------------|-------------------|-----------------|------------------|----------------|------------------|----------------|--------------------|-------------------|
| 4156.76 | 24 | 3925.61 | 3352.087 | 0.854 | 1464.492 | 0.373 | 2750.454 | 0.701 | 3778.555 | 0.963 | 47.853 | 0.012 |

**分析**：
- 相比Case 4，Task Duration从5039.86μs降低到4156.76μs，耗时降低**17.52%**
- aic_mte2_time从4051.458μs降低到3778.555μs，MTE2搬运时间降低了**6.47%**
- aic_mac_ratio从69.4%提85.4%，cube利用率提升了**23.05%**
- 后续优化方向：aic_mte2_ratio达到**96.3%**，达到MTE2 bound，成为性能提升瓶颈，后续将优化L2Cache以缓解MTE2 bound
---

### Case 6: 多核MDL + L1Cache + L2Cache (MULTI_CORE_MDL_L1CACHE_L2CACHE)

**优化目标**：启用L2Cache优化，缓解MTE2 bound

**核心实现**：
- 使用24核并行计算，切分策略同上（4x6）
- 使用MDL模板并优化L1 Cache参数同上
- 启用`L2Cache`，对A矩阵M轴进行切分

**关键代码**：
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

**优化手段**：
- **启用L2Cache优化**：
  - L2Cache特点：L2Cache是AI Core共享的外部缓存，L2Cache的纯读带宽约为GM的3到4倍，在搬入或搬出相同数据量的情况下，访问L2Cache内的数据比GM更快
  - 缓存命中优化：若数据无法命中L2Cache，导致需要访问GM进行读写，带宽利用效率较低，导致MTE2可能成为样例整个运行过程中的性能瓶颈
  - **分块计算适配L2Cache**：当前L2Cache大小为**192MB**，矩阵计算所需数据总量为**384MB**(A矩阵128MB + B矩阵128MB + C矩阵128MB)，由于L2Cache容量小于矩阵计算数据总量，因此可以将A矩阵在M轴切分成2份，其中区域1与B矩阵完成完整运算，区域2再与B矩阵完成完整运算，通过切分提高L2cache的命中率

分块计算适配L2Cache示意图：

<img src="figure/L2Cache.png">

```
计算过程：
Step 1: C1 = A1 × B  (A1从GM加载到L2Cache，B从GM加载到L2Cache并驻留)
Step 2: C2 = A2 × B  (A2从GM加载到L2Cache，B已在L2Cache中，无需重新加载)

L2Cache利用：
- B矩阵（128MB）在Step 1加载后驻留在L2Cache
- Step 2中B矩阵直接从L2Cache读取，避免GM访问
- 单次L2Cache访问带宽约为GM的3到4倍，显著提升访存效率
```

**性能数据**：
| Task Duration(μs) | Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio |
|------------------|-----------|----------------|-----------------|---------------|-------------------|-----------------|------------------|----------------|------------------|----------------|--------------------|-------------------|
| 4088.36 | 24 | 3786.26 | 3254.31 | 0.86 | 1753.463 | 0.463 | 2680.849 | 0.708 | 3625.42 | 0.958 | 47.398 | 0.013 |

**分析**：
- 相比Case 5，Task Duration从4156.76μs降低到4088.36μs，降低了**68.40μs**
- aic_mte2_time从3778.555μs降低到3625.42μs，MTE2搬运时间降低了**4.05%**
- aic_mac_ratio从**85.4%**提升到**86%**
- 后续优化方向：当前制约样例性能的仍为MTE2 bound，开发者可通过优化L2Cache的切分策略对MTE2进一步优化。本样例后续将展示对Scalar和Fixpipe流水的优化

---

### Case 7: 多核MDL + L1Cache + L2Cache + Constants Tiling (MULTI_CORE_MDL_L1CACHE_L2CACHE_CONSTANTS)

**优化目标**：使用常量Tiling，减少运行时Scalar计算开销

**核心实现**：
- 使用24核并行计算，切分策略同上（4x6）
- 使用MDL模板并优化L1 Cache参数同上
- L2 Cache策略同上
- 使能Tiling全量常量化

**关键代码**：
```cpp
constexpr MatmulShapeParams shapeParams = {SINGLE_M_L2CACHE, SINGLE_N, SINGLE_K, BASE_M, BASE_N, BASE_K};

constexpr static auto CONSTANT_CFG = GetCustomConstantCFG<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, true, true>();
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CONSTANT_CFG> matmulObj;
// CONSTANT_CFG在编译期计算，减少运行时Scalar计算
```

**优化手段**：
- **Tiling常量化**：
  - AI Core的硬件优势在于向量/矩阵并行计算，标量运算无法发挥硬件能力。**Matmul API在初始化和迭代过程中有大量Scalar计算，Matmul初始化时的Scalar计算影响指令头开销，Matmul迭代间的Scalar计算可能阻塞MTE2流水**
  - 静态Tiling减少运行时开销：使用**MatmulApiStaticTiling**参数替代TCubeTiling变量参数，将Scalar计算提前到编译期进行，运行时不需要通过Scalar单元动态计算，减少Scalar性能开销

**性能数据**：
| Task Duration(μs) | Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio |
|------------------|-----------|----------------|-----------------|---------------|-------------------|-----------------|------------------|----------------|------------------|----------------|--------------------|-------------------|
| 4053.44 | 24 | 3665.12 | 3163.682 | 0.863 | 968.616 | 0.264 | 2609.617 | 0.712 | 3513.806 | 0.959 | 45.76 | 0.012 |

**分析**：
- 相比Case 6，Task Duration从4088.36μs降低到4053.44μs
- aic_scalar_time从1753.463μs降低到968.616μs，降低**44.76%**
- 常量Tiling有效减少了Scalar单元的性能开销。当样例因Scalar阻塞性能较差时，用户可通过该方式降低Scalar时间占比
- 后续优化方向：启用UnitFlag优化，演示并行化计算与搬运流水

---

### Case 8: 多核MDL + L1Cache + L2Cache + Constants Tiling + UnitFlag (MULTI_CORE_MDL_L1CACHE_L2CACHE_CONSTANTS_UNITFLAG)

**优化目标**：启用UnitFlag优化，并行化计算与搬运流水

**核心实现**：
- 使用24核并行计算，切分策略同上（4x6）
- 使用MDL模板并优化L1 Cache参数同上
- L2 Cache策略同上
- 使能Tiling全量常量化同上
- 启用UnitFlag，优化计算与搬运的并行度

**关键代码**：
```cpp
// 使用constexpr定义编译期常量，并启用UnitFlag
MatmulConfig mmCFG = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(shapeParams);
mmCFG.enUnitFlag = true;
```

**优化手段**：
- **启用UnitFlag优化**：
  - 优化计算与搬运的并行度：未开启UnitFlag功能时，AIC核的MMAD计算指令和FIXPIPE数据搬运指令是指令级别的同步，FIXPIPE指令需要等MMAD指令执行完才进行结果搬出，MMAD和FIXPIPE之间流水串行；开启UnitFlag功能后，MMAD和FIXPIPE是**512B大小的细粒度同步**，在一条MMAD指令执行过程中，每当完成一个512B数据结果的计算，FIXPIPE立即搬出该512B的数据，从而实现Cube计算单元和FIXPIPE搬运单元流水并行

UnitFlag功能示意图：

<img src="figure/unitflag_close.png">
<img src="figure/unitflag_open.png">



**性能数据**：
| Task Duration(μs) | Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio |
|------------------|-----------|----------------|-----------------|---------------|-------------------|-----------------|------------------|----------------|------------------|----------------|--------------------|-------------------|
| 4012.44 | 24 | 3559.06 | 3076.396 | 0.864 | 1026.069 | 0.288 | 2533.512 | 0.712 | 3435.584 | 0.965 | 272.576 | 0.077 |

**分析**：
- 相比Case 7，Task Duration从4053.44μs降低到4012.44μs
- aic_fixpipe_time从45.76μs提高到272.576μs，原因为开启Unitflag后，aic_fixpipe_time会包含FIXPIPE指令的等待时间，并不是真正流水的耗时，用户可关注端到端性能是否提升
- 当前性能收益较小的原因为制约性能的仍为MTE2 bound，算子的MMAD、FIXPIPE流水被MTE2 bound掩盖。当样例因FIXPIPE阻塞性能较差时，用户可启用UnitFlag功能

---

## 性能对比总结

**综合优化效果**：
- 本样例cube利用率提升了**67.7%**(18.7% → 86.4%)，已达到芯片峰值算力的**86.4%**
- 通过Case 0到Case 8的递进优化，样例耗时降幅达**99.47%**(759363.98μs → 4012.44μs)


| Case version | Task Duration(μs) | Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio |
|------|------------------|-----------|----------------|-----------------|---------------|-------------------|-----------------|------------------|----------------|------------------|----------------|--------------------|-------------------|
| Case 0 | 759363.98 | 1 | 759363.46 | 141699.459 | 0.187 | 508787.509 | 0.67 | 162104.184 | 0.213 | 582515.828 | 0.767 | 34303.063 | 0.045 |
| Case 1 | 249467.08 | 1 | 249466.54 | 81477.188 | 0.327 | 63608.818 | 0.255 | 52003.703 | 0.208 | 195613.432 | 0.784 | 11313.746 | 0.045 |
| Case 2 | 12541.22 | 24 | 12537.56 | 3462.802 | 0.276 | 2987.47 | 0.238 | 2260.551 | 0.18 | 10177.798 | 0.812 | 875.439 | 0.07 |
| Case 3 | 12283.84 | 24 | 10870.96 | 3394.884 | 0.312 | 2656.33 | 0.244 | 2166.824 | 0.199 | 8616.671 | 0.793 | 488.829 | 0.045 |
| Case 4 | 5039.86 | 24 | 4487.89 | 3116.472 | 0.694 | 2073.97 | 0.462 | 2332.914 | 0.52 | 4051.458 | 0.903 | 44.749 | 0.01 |
| Case 5 | 4156.76 | 24 | 3925.61 | 3352.087 | 0.854 | 1464.492 | 0.373 | 2750.454 | 0.701 | 3778.555 | 0.963 | 47.853 | 0.012 |
| Case 6 | 4088.36 | 24 | 3786.26 | 3254.31 | 0.86 | 1753.463 | 0.463 | 2680.849 | 0.708 | 3625.42 | 0.958 | 47.398 | 0.013 |
| Case 7 | 4053.44 | 24 | 3665.12 | 3163.682 | 0.863 | 968.616 | 0.264 | 2609.617 | 0.712 | 3513.806 | 0.959 | 45.76 | 0.012 |
| Case 8 | 4012.44 | 24 | 3559.06 | 3076.396 | 0.864 | 1026.069 | 0.288 | 2533.512 | 0.712 | 3435.584 | 0.965 | 272.576 | 0.077 |

## 调优建议

1. **从小规模开始**：先使用单核基础版本验证功能正确性
2. **逐步优化**：按照case顺序逐步引入优化手段，观察性能提升
3. **多核切分策略**：合理设置多核切分策略，避免同地址访问
4. **利用MDL模式**：MDL模式提供了高度优化的实现，优先使用
4. **L2Cache**：L2Cache能进一步缓解MTE2 bound，对于需重复读取的数据推荐使用
6. **常量Tiling**：对于固定shape的场景，使用常量Tiling减少Scalar运行时开销
7. **UnitFlag优化**：启用UnitFlag可以并行化计算与搬运流水

## 编译运行
在本样例根目录下执行如下步骤，编译并执行样例。

### 配置环境变量
请根据当前环境上CANN开发套件包的[安装方式](../../../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。

- 默认路径，root用户安装CANN软件包
  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```

- 默认路径，非root用户安装CANN软件包
  ```bash
  source $HOME/Ascend/cann/set_env.sh
  ```

- 指定路径install_path，安装CANN软件包
  ```bash
  source ${install_path}/cann/set_env.sh
  ```

### 样例执行
```bash
mkdir -p build && cd build;    # 创建并进入build目录
cmake ..;make -j;              # 编译工程
python3 ../scripts/gen_data.py # 生成测试输入数据

# 执行不同的优化case（通过参数指定case编号）
./demo 0   # Case 0: 单核基础版本
./demo 1   # Case 1: 单核Tiling优化
./demo 2   # Case 2: 多核切分 2x12
./demo 3   # Case 3: 多核切分 4x6
./demo 4   # Case 4: 多核MDL版本
./demo 5   # Case 5: 多核MDL + L1Cache
./demo 6   # Case 6: 多核MDL + L1Cache + L2Cache
./demo 7   # Case 7: 多核MDL + L1Cache + L2Cache + Constants Tiling
./demo 8   # Case 8: 多核MDL + L1Cache + L2Cache + Constants Tiling + UnitFlag

python3 ../scripts/verify_result.py output/output.bin output/golden.bin # 验证输出结果是否正确
```

执行结果如下，说明精度对比成功。
```bash
test 0: Single Core Basic
test pass!
```
 ### 性能分析
    
  使用 `msprof` 工具获取详细性能数据：
    
  ```bash
  msprof ./demo 7   # 分析case 7的性能
  ```

  当前目录下会生成OPPROF_前缀的文件夹，`mindstudio_profiler_output`目录保存Host和各个Device的性能数据汇总，性能数据分析推荐查看该目录下文件
  ```bash
OPPROF_xxxx_XXXXXX
├── device_{id}
└── host
└── mindstudio_profiler_log
└── mindstudio_profiler_output    # 保存Host和各个Device的性能数据汇总
    ├── msprof_*.json
    ├── xx_*.csv
    └── README.txt
```