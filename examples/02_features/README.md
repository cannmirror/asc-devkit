# Features样例介绍
## 概述
基于Ascend C特性样例，介绍了Aclnn（ge入图）工程、LocalMemoryAllocator、Barrier单独内存申请分配等特性。

## 算子开发样例
| 目录名称 | 算子样例 | 功能描述 |
| --------- | --------- | --------- |
| [00_framework_launch](./00_framework_launch/) | [00_add_template](./00_framework_launch/00_add_template) | 本样例以Add算子为示例，展示了Tiling模板编程。Add算子实现了两个数据相加，返回相加结果的功能。本样例使用自定义算子工程，编译并部署自定义算子包到自定义算子库中，并调用执行自定义算子 |
| [01_c_api](./01_c_api) | [00_sync_add](./01_c_api/00_sync_add) | 本样例展示了使用C_API构建Add算子样例的编译流程 |


## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/11/28 | 新增add用例 |
| 2025/11/18 | 样例目录调整，新增本readme |
