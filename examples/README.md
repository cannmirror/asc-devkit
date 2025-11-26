# 样例运行验证

开发者调用Ascend C API实现自定义算子后，可通过单算子调用的方式验证算子功能。本代码仓提供部分算子实现及其调用样例，具体如下。

## 算子开发样例
|  目录名称                                                   |  功能描述                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [00_introduction](./00_introduction) | 基于Ascend C的简单的示例，通过Ascend C编程语言实现了自定义算子，并按照不同的算子调用方式分别给出了对应的<<<>>>直调实现，适合初学者 |
| [01_utilities](./01_utilities) | 样例通过Ascend C编程语言实现了自定义算子，实现printf、assert及debug功能等 |
| [02_features](./02_features) | Ascend C的特性：Aclnn（ge入图）工程，LocalMemoryAllocator、Barrier单独内存申请和分配、SIMT编程 |
| [03_libraries](./03_libraries) | 高阶API、基础API类库的使用示例，包括数学库，激活函数等 |
| [04_best_practices](./04_best_practices) | 最佳实践示例 |