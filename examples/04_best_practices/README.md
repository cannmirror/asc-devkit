# BestPractices样例介绍

## 概述

基于Ascend C的性能优化实践，聚焦于关键算子与内存访问的调优，旨在提升在Ascend平台上的运行效率。

## 算子开发样例

|  目录名称                                                   |  功能描述                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [00_add_doublebuffer](./00_add_doublebuffer) | 本样例介绍基于静态Tensor方式编程的场景下Add算子的实现方法，优化性能，使用double buffer进行流水排布，支持main函数和kernel函数在同一个cpp文件中实现，并提供<<<>>>直调方法。 |
| [01_bank_conflict](./01_bank_conflict) | 基于AscendC的bank冲突性能优化样例。 |
| [03_l2_cache_bypass](./03_l2_cache_bypass) | 本样例介绍了设置L2 CacheMode的方法以及其影响场景，并提供核函数直调方法。 |
| [05_mata_address_conflict](./05_mata_address_conflict) | 本样例介绍了同地址冲突的影响以及两种解决方法，并提供核函数直调方法。 |
| [06_grouped_matmul](./06_grouped_matmul) | 本样例介绍QuantGroupMatmul算子在NPU上高性能实现，支持分组量化矩阵乘与Gelu激活计算。 |
| [10_compatibility_cases](./10_compatibility_cases) | 本样例介绍Atlas A2 训练系列产品/Atlas A2 推理系列产品部分不兼容特性迁移至950的实现样例方法，不兼容场景包括从L1 Buffer直接搬运到GM、L0A Buffer/L0B Buffer、int4b_t数据类型下的矩阵计算、L1 Buffer带边界值场景。 |
| [11_pattern_transformation](./11_pattern_transformation) | 新架构下基础的mmad样例，从L1 Buffer->L0A的通路不需要做Nz2Zz分型转换。 |
| [14_pure_simt_gather](./14_pure_simt_gather) |纯SIMT编程方式实现的算子样例，支持动态计算切分参数 |
