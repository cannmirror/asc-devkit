# 样例运行验证

开发者调用Ascend C API实现自定义算子后，可通过单算子调用的方式验证算子功能。本代码仓提供部分算子实现及其调用样例，具体如下。

## 算子开发样例
|  目录名称                                                   |  功能描述                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [00_introduction](./00_introduction) | 基于Ascend C的简单的示例，通过Ascend C编程语言实现了自定义算子，分别给出对应的<<<>>>直调实现 |
| [01_simd_cpp_api](./01_simd_cpp_api) | 基于Ascend C的SIMD API样例，通过<<<>>>直调的实现方式，介绍了SIMD API的使用方法 |
| [02_features](./02_features) | 基于Ascend C特性样例，介绍了Aclnn（ge入图）工程、LocalMemoryAllocator、Barrier单独内存申请分配等特性 |
| [02_simd_c_api](./02_simd_c_api) | 基于Ascend C的C_API样例，通过C_API的实现方式，介绍了C_API的使用方法 |

## npu-arch编译选项说明

开发者需根据实际的执行环境，修改具体样例目录下CMakeLists.txt文件中的--npu-arch编译选项，参考下表中的对应关系，修改为环境对应的npu-arch参数值。
| 产品型号 |  npu-arch参数 |
| ---- | ---- |
| Ascend 950PR/Ascend 950DT | --npu-arch=dav-3510 |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品<br>Atlas A2 训练系列产品/Atlas A2 推理系列产品 | --npu-arch=dav-2201 |
| Atlas 推理系列产品AI Core | --npu-arch=dav-2002 |