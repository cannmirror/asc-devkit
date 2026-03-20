# 样例运行验证

开发者调用Ascend C API实现自定义算子后，可通过单算子调用的方式验证算子功能。本代码仓提供部分算子实现及其调用样例，具体如下。

## 算子开发样例
|  目录名称                                                   |  功能描述                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [00_introduction](./00_introduction) | 基于Ascend C的简单的示例，通过Ascend C编程语言实现了自定义算子，分别给出对应的<<<>>>直调实现，适合初学者 |
| [01_utilities](./01_utilities) | 基于Ascend C的简单样例，通过printf、assert、debug等API介绍上板打印、异常检测、CPU孪生调试等系统工具使用方法，适用于调试阶段 |
| [02_features](./02_features) | 基于Ascend C特性样例，介绍了Aclnn（ge入图）工程、LocalMemoryAllocator、Barrier单独内存申请分配等特性 |
| [03_libraries](./03_libraries) | 基于Ascend C API类库的使用样例，通过<<<>>>直调的实现方式，介绍了数学库，激活函数等API类库 |
| [04_best_practices](./04_best_practices) | 1. 基于Ascend C的性能优化实践，聚焦于关键算子与内存访问的调优，旨在提升在Ascend平台上的运行效率。 2. 针对不兼容的特性，提供兼容性样例 |

## npu-arch编译选项说明

开发者需根据实际的执行环境，修改具体样例目录下CMakeLists.txt文件中的--npu-arch编译选项，参考下表中的对应关系，修改为环境对应的npu-arch参数值。
| 产品型号 |  npu-arch参数 |
| ---- | ---- |
| Ascend 950PR/Ascend 950DT | --npu-arch=dav-3510 |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品<br>Atlas A2 训练系列产品/Atlas A2 推理系列产品 | --npu-arch=dav-2201 |
| Atlas 推理系列产品AI Core | --npu-arch=dav-2002 |