# Features样例介绍

## 概述

基于Ascend C特性样例，介绍了Aclnn（ge入图）工程、LocalMemoryAllocator、Barrier单独内存申请分配等特性

## 算子开发样例

| 目录名称  | 功能描述 |
| --------- | --------- |
| [00_framework_launch](./00_framework_launch/) | 本样例以Add算子为样例，展示了Tiling模板编程。Add算子实现了两个数据相加，返回相加结果的功能。本样例使用自定义算子工程，编译并部署自定义算子包到自定义算子库中，并调用执行自定义算子 |
| [01_triple_chevron_notation](./01_triple_chevron_notation/) | 本样例展示了如何使用pybind11与torch.library注册自定义算子，并通过`<<<>>>`内核调用符调用核函数 |
| [02_c_api](./02_c_api) | 本样例展示了使用C_API构建Add算子样例的编译流程 |
| [03_simt](./03_simt/) | 本样例展示了SIMT算子实现，包括基于SIMT编程模型、SIMD与SIMT混合编程模型两种模式的实现 |
| [04_reg_compute](./04_reg_compute/) | 通过Reg矢量计算API实现自定义算子，分别给出对应的<<<>>>直调实现 |
| [06_static_tensor_programming](./06_static_tensor_programming) | 本样例介绍基于静态Tensor方式编程的场景下Add算子的实现方法，并提供核函数直调方法 |
| [07_data_movement](./07_data_movement) | 本样例路径以copy、data_copy_pad、data_copy_ub2l1等算子为示例，展示了数据搬运接口的使用。|
| [08_tiling](./08_tiling) | 本样例实现了一个支持多种数据类型的向量加法算子，其核心在于采用精细的多核并行数据切分（Tiling）策略，通过17个参数动态管理任务分配，处理核间负载不均及尾核尾块场景，以实现高效能计算。|
| [09_unalign](./09_unalign/) | 本样例路径以abs、reduce_min、whole_reduce_sum等算子为示例，展示了数据非32字节对齐场景中的处理方式，包括数据搬入，计算和搬出的处理 |
| [10_memory_management](./10_memory_management/) | 本路径下包含了与资源管理相关的多个API的样例。每个样例均基于Ascend C的<<<>>>直调方法，支持main函数和kernel函数在同一个cpp文件中实现。         |
| [11_synchronous_control](./11_synchronous_control/) | 本路径下包含了与同步控制相关的多个API的样例。每个样例均基于Ascend C的<<<>>>直调方法，支持main函数和kernel函数在同一个cpp文件中实现。         |
| [12_system_variable_access](./12_system_variable_access/) | 本路径下包含了与系统变量访问相关的多个API的样例。每个样例均基于Ascend C的<<<>>>直调方法，支持main函数和kernel函数在同一个cpp文件中实现。         |
| [13_atomic_operations](./13_atomic_operations/) | 本路径下包含了与原子操作相关的多个API的样例。每个样例均基于Ascend C的<<<>>>直调方法，支持main函数和kernel函数在同一个cpp文件中实现。         |
| [14_cube_group_management](./14_cube_group_management/) | 本路径下包含了与Cube分组管理相关的多个API的样例。每个样例均基于Ascend C的<<<>>>直调方法，支持main函数和kernel函数在同一个cpp文件中实现。 
| [15_utility_function](./15_utility_function/) | 本路径下包含了与工具函数相关的多个API的样例。每个样例均基于Ascend C的<<<>>>直调方法，支持main函数和kernel函数在同一个cpp文件中实现。         |
| [16_scalar_computation](./16_scalar_computation/) | 本路径下包含了与标量计算相关的多个API的样例。每个样例均基于Ascend C的<<<>>>直调方法，支持main函数和kernel函数在同一个cpp文件中实现。         |