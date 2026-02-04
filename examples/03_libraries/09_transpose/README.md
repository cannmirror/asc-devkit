# 张量变换算子样例介绍

## 概述

本样例集介绍了张量变换算子不同特性的典型用法，给出了对应的端到端实现。

## 算子开发样例

|  目录名称                                                   |  功能描述                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [brcb](./brcb) | 本样例基于Brcb实现数据填充，可用于输入一个tensor，每一次取输入张量中的8个数填充到结果张量的8个datablock中 |
| [broadcast](./broadcast) | 本样例基于Kernel直调算子工程，介绍了调用BroadCast高阶API实现broadcast单算子，主要演示BroadCast高阶API在Kernel直调工程中的调用 |
| [duplicate](./duplicate) | 本样例基于Duplicate实现数据填充，可用于将一个变量或立即数复制多次并填充到向量中 |
| [fill](./fill) | 本样例基于Kernel直调算子工程，介绍了调用Fill高阶API实现fill单算子，主要演示Fill高阶API在Kernel直调工程中的调用 |
| [pad](./pad) | 本样例基于Kernel直调算子工程，介绍了调用Pad高阶API实现pad单算子，对height * width的二维Tensor在width方向上pad到32B对齐，如果Tensor的width已32B对齐，且全部为有效数据，则不支持调用本接口对齐 |
| [trans_data_to_5hd](./trans_data_to_5hd) | 本样例基于TransDataTo5HD实现数据格式转换，可用于NCHW格式转换成NC1HWC0格式，特别的也可以用于二维矩阵数据块的转置 |
| [transdata](./transdata) | 本样例演示了基于TransData高阶API实现的算子实现。样例将输入数据的排布格式转换为目标排布格式 |
| [transpose](./transpose) | 本样例介绍了调用Transpose高阶API实现Transpose算子，并按照核函数直调的方式分别给出了对应的端到端实现 |
| [transpose_common](./transpose_common) | 本样例基于Transpose实现普通转置，适用于对16*16的二维矩阵数据块进行转置 |
| [transpose_enhanced](./transpose_enhanced) | 本样例基于Transpose实现增强转置，适用于对16*16的二维矩阵数据块进行转置，也可用于[N,C,H,W]与[N,H,W,C]互相转换 |
| [unpad](./unpad) | 本样例基于Kernel直调算子工程，介绍了调用UnPad高阶API实现unpad单算子，对height * width的二维Tensor在width方向上unpad到32B对齐，如果Tensor的width已32B对齐，且全部为有效数据，则不支持调用本接口对齐 |