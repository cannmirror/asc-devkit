# ATVC
## 概述
ATVC(Ascend C Template for Vector Compute)是一个用Ascend C API搭建的C++模板头文件集合，旨在帮助用户快速开发Ascend C典型Vector算子。它将Ascend C Vector算子开发流程中的计算实现解耦成可自定义的模块, 内部封装实现了kernel数据搬入搬出等底层通用操作及通用tiling计算，实现了高效的算子开发模式。
相比传统Ascend C算子开发方式，利用ATVC搭建的Vector算子可做到开发效率提升3-5倍。用户只需选择匹配的模板并完成核心计算逻辑就完成算子kernel侧开发，ATVC还内置了每个模板库对应的通用tiling计算实现，可省去用户手写tiling的开发量就能达到不错的性能表现，极大提升算子开发效率。

<img src="./docs/images/atvc_user_case.png" style="margin: 30px 0;"><br>


请参阅[快速入门](./docs/01_quick_start.md)以快速了解ATVC的Add算子搭建流程。
请参阅[开发者文档](./docs/02_developer_guide.md)以获取ATVC框架各模板与API的使用细节，完成自定义Elementwise类算子以及Reduce类算子开发。


## 支持的产品型号
- 硬件型号支持  
Atlas A2训练系列产品/Atlas 800I A2推理产品/A200I A2 Box 异构组件。


## 目录结构说明
本代码仓目录结构如下：
``` 
├── docs        // 文档介绍
├── examples    // ATVC使用样例
├── include     // ATVC提供的头文件集合,用户使用前需将其置入其他工程的包含路径下
└── README.md   // 综述
```
[Developer Guide](./docs/02_developer_guide.md)给出了ATVC框架各模板与API的使用细节。

[Code Organization](./docs/03_code_organization.md)给出了模板库代码的组织结构。

[examples](./examples/)给出了使用ATVC模板库开发Vector算子的样例。

## 环境准备

参考[ascendc-api-adv仓通用环境准备章节](../../README.md)，完成源码下载和CANN软件包及相关依赖的安装。

## ATVC模板库算子调试方式
- ATVC是一个头文件集合，只需要包含头文件目录即可使用ATVC模板能力进行算子开发。
- [样例集合](../atvc/examples/)包含了多种模板、多种调用场景的算子样例，ops_aclnn和ops_pytorch展示了基于单算子API调用和PyTorch框架调用的算子样例，其他的均为Kernel直调场景下的样例。单算子API调用和PyTorch框架调用算子样例编译调试步骤详见对应样例路径下的README.md文档。

### Kernel直调算子样例本地编译调试
- Kernel直调算子样例可通过执行脚本快速发起算子编译与运行，运行命令如下所示， 其{op_name}是实际运行算子路径：
```bash
cd ./ops_templates/atvc/examples
bash run_examples.sh {op_name}
```
- 支持上板运行打开profiling获取性能数据， 运行命令为：bash run_examples.sh {op_name} --run-mode=profiling。
- 支持上板打印ATVC模板库提供的DFX信息，运行命令为：bash run_examples.sh {op_name} --run-mode=debug_print

## 模板选择
ATVC支持的模板和数据类型如下：
| 算子模板        | 数据类型       | 规格限制说明 |
| -------------------- | ---------------- | ---------- |
| Elementwise | int32_t、float               |  |
| Reduce | int32_t、float               | 当前只支持4维以内的reduce计算 |
| Broadcast | int32_t、float            | 当前只支持2维以内对齐场景的计算 |

### Elementwise类算子
Elementwise类算子通常是指对张量进行元素级别的操作的函数或方法，包括但不限于加、减、乘、除及指数、对数、三角函数等数学函数。这类算子的特点是会逐元素进行计算操作，而不会改变输入数据的形状。常见的Elementwise算子有Add、Sub、Exp、Log、Sin、Sqrt等。
### Reduce类算子
Reduce类算子通常是指对张量中的元素进行归约操作的算子，通常用来求和、求平均值等操作，可指定某几个维度进行归约计算，也可以将所有元素归约计算为一个标量。常见的Reduce类算子有ReduceSum(求和)、ReduceMean(求平均值)、ReduceProdcut(累乘)、ReduceMax(求最大值)、ReduceMin(求最小值)、ReduceAny(or操作)、ReduceAll(and操作)。
### Broadcast类算子
Broadcast算子用于在张量形状不一致时实现张量间的逐元素运算。
设张量 A 的 shape 为 (1, 5)，张量 B 的 shape 为 (3, 5)。为完成 C = A + B，首先需依据广播规则将 A 由 (1, 5) 扩展至 (3, 5)。该过程通过在长度为 1 的维度上复制数据，使两个张量的形状对齐，从而支持逐元素相加运算。

