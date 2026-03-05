# Utilities样例介绍
## 概述
基于Ascend C的简单样例，通过printf、assert、debug等API介绍上板打印、异常检测、CPU孪生调试等系统工具使用方法，适用于调试阶段。

## 算子开发样例
|  目录名称                                                   |  功能描述                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [00_printf](./00_printf) | 本样例通过Ascend C编程语言实现了Matmul算子，同时在算子中添加printf调测 |
| [01_assert](./01_assert) | 本样例通过Ascend C编程语言实现了Matmul算子，同时在算子中添加assert调测，给出了对应的端到端实现 |
| [02_dumptensor](./02_dumptensor) | 本样例通过Ascend C编程语言实现了Add算子和Mmad算子，在算子中添加DumpTensor调测，给出了对应的端到端实现 |
| [03_cpudebug](./03_cpudebug) | 本样例通过Ascend C编程语言实现了Add算子的CPU Debug调测，给出了对应的端到端实现 |
| [04_clock](./04_clock) | 本样例通过Ascend C编程语言实现了Gather算子，同时在算子中添加clock调测，给出了对应的端到端实现 |