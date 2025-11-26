# Introduce样例介绍
## 概述
简单的样例，适合初学者。样例通过Ascend C编程语言实现了自定义算子，给出了对应的<<<>>>直调实现。

## 算子开发样例
|  目录名称                                                   |  功能描述                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [00_helloworld](./00_helloworld) | 基于Ascend C的HelloWorld自定义算子调用结构演示样例 |
| [01_add](./01_add) | 基于Ascend C的Add自定义Vector算子的核函数直调方法，支持main函数和Kernel函数在同一个cpp文件中实现 |
| [02_matmul](./02_matmul) | 基于Ascend C的Matmul算子的核函数直调方法 |
| [03_matmulleakyrelu](./03_matmulleakyrelu) | 基于Ascend C的MatmulLeakyRelu自定义算子的核函数直调方法，支持main函数和Kernel函数在同一个cpp文件中实现 |
| [04_addn](./04_addnh) | 基于Ascend C的AddN自定义Vector算子，介绍了单算子直调方法 |
| [05_broadcast](./05_broadcast) | 基于Ascend C的Broadcast自定义Vector算子，介绍了单算子直调方法  |
| [06_reduce](./06_reduce) | 基于Ascend C的Reduce自定义Vector算子，以直调的方式调用算子核函数 |
| [07_sub](./07_sub) | 基于Ascend C的Sub自定义Vector算子,介绍了单算子直调方法 |
| [08_unaligned_abs](./08_unaligned_abs) | 基于Ascend C的DataCopyPad的非对齐Abs算子核函数直调方法 |
| [09_unaligned_reducemin](./09_unaligned_reducemin) | 基于Ascend C的无DataCopyPad的非对齐ReduceMin算子核函数直调方法 |
| [10_unaligned_wholereduces](./10_unaligned_wholereduces) | 基于Ascend C的非对齐WholeReduceSum算子的核函数直调方法 |
| [11_vectoradd](./11_vectoradd) | 基于Ascend C的Add算子的核函数直调方法，算子支持单核运行 |

## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/11/06 | 样例目录调整，新增本readme |