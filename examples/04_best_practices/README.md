# BestPractices样例介绍
## 概述
基于Ascend C的性能优化实践，聚焦于关键算子与内存访问的调优，旨在提升在Ascend平台上的运行效率。

## 算子开发样例
|  目录名称                                                   |  功能描述                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [00_group_matmul](./00_group_matmul) | 本样例介绍QuantGroupMatmul算子在NPU上高性能实现，支持分组量化矩阵乘与Gelu激活计算 |
| [01_aicpu_device_tiling](./01_aicpu_device_tiling) | 本样例介绍使用AI CPU算子进行tiling下沉计算的实现, 在device侧将AI CPU算子的计算结果传给AI Core算子，使用<<<>>>内核调用符来完成算子核函数在NPU侧运行验证的基础流程 |
| [02_add_doublebuffer](./02_add_doublebuffer) | 本样例介绍基于静态Tensor方式编程的场景下Add算子的实现方法，优化性能，使用double buffer进行流水排布，支持main函数和kernel函数在同一个cpp文件中实现，并提供<<<>>>直调方法 |
| [03_compability_cases](./03_compability_cases) | 本样例介绍910B部分不兼容算子迁移910D的实现方法，不兼容场景包括从L1 Buffer直接搬运到GM、L0A Buffer/L0B Buffer、int4b_t数据类型下的矩阵计算、L1 Buffer带边界值场景 |

## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/11/18 | 样例目录调整，新增本readme |