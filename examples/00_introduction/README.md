# Introduce样例介绍
## 概述
基于Ascend C的简单的示例，通过Ascend C编程语言实现了自定义算子，分别给出对应的<<<>>>直调实现，适合初学者

## 算子开发样例
|  目录名称                                                   |  功能描述                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [00_helloworld](./00_helloworld) | 本样例介绍了基于Ascend C的HelloWorld自定义算子调用结构演示样例，核函数内通过printf打印输出结果 |
| [01_add](./01_add) | 本样例介绍了基于Ascend C的Add自定义Vector算子的核函数直调方法，实现两个输入张量的逐元素相加，支持main函数和kernel函数在同一个cpp文件中实现 |
| [02_matmul](./02_matmul) | 本样例介绍了基于Ascend C的Matmul算子的核函数直调方法，可最大化利用AI处理器的并行计算能力，显著提升算子的执行效率，使用与高性能推理与训练场景 |
| [03_matmulleakyrelu](./03_matmulleakyrelu) | 本样例介绍了基于Ascend C的MatmulLeakyRelu自定义算子的核函数直调方法，能够完成矩阵乘加与LeakyReLU激活的融合计算，该方式将关键计算步骤在硬件层面高效协同执行，显著降低内存访问开销与计算延时 |
| [04_addn](./04_addn) | 本样例介绍了基于Ascend C的AddN自定义Vector算子，样例支持两个张量的动态相加运算，使用ListTensorDesc结构灵活处理多个输入参数，实现高效、可扩展的核函数调用 |
| [05_broadcast](./05_broadcast) | 本样例介绍了基于Ascend C的Broadcast自定义Vector算子，样例支持将输入张量按目标形状进行广播，通过直接调用核函数，免去框架调度开销，实现高效、低延时的张量扩展运算 |
| [06_reduce](./06_reduce) | 本样例介绍了基于Ascend C的Reduce自定义Vector算子，对输入张量沿最后一个维度进行求和，样例采用核函数直调的方式，规避框架调度开销，实现高效的归约计算 |
| [07_sub](./07_sub) | 本样例介绍了基于Ascend C的Sub自定义Vector算子，样例实现两个张量的逐元素相减，采用核函数直调的方式，有效降低调度开销，实现高效的算子执行 |
| [08_unaligned_abs](./08_unaligned_abs) | 本样例介绍了基于Ascend C的DataCopyPad的非对齐Abs算子核函数直调方法，有效降低调度开销，实现高效的算子执行 |
| [09_unaligned_reducemin](./09_unaligned_reducemin) | 本样例介绍了基于Ascend C的无DataCopyPad的非对齐ReduceMin算子核函数直调方法，有效降低调度开销，实现高效的算子执行 |
| [10_unaligned_wholereducesum](./10_unaligned_wholereducesum) | 本样例介绍了基于Ascend C的非对齐WholeReduceSum算子的核函数直调方法，有效降低调度开销，实现高效的算子执行 |
| [11_vectoradd](./11_vectoradd) | 本样例介绍了基于Ascend C的Add算子的核函数直调方法，算子支持单核运行，有效降低调度开销，实现高效的算子执行 |
| [12_aicpu](./12_aicpu) | 本样例介绍了基于Ascend C的AI CPU算子的核函数直调方法，演示了HelloWorld |