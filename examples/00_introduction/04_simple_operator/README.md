# SimpleOperator样例介绍

## 概述

样例介绍了5个基于Ascend C的算子的核函数直调样例，涵盖AddN、Broadcast、Gather、Sub以及向量Add等典型算子，展示了动态Tensor、纯SIMT编程、临时缓冲区使用等关键技术，充分体现了Ascend C在高性能算子开发中的灵活性与高效性。

## 算子开发样例

|  目录名称                                                   |  功能描述                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [add_broadcast](./add_broadcast) | 本样例介绍Add算子的核函数直调方法，多核&tiling场景下增加输入Broadcast |
| [add_dynamic](./add_dynamic) | 本样例演示基于动态Tensor编程模型的AddN算子实现，该实现采用ListTensorDesc结构处理多输入参数，结合TQue内存管理机制实现数据搬运与计算任务的协同调度 |
| [broadcast](./broadcast) | 本样例展示了一个支持多种数据类型（如bfloat，int8，float，half等）和多种形状（如(32, 1024)，(8, 1023)等）的输入张量执行逐元素加法 |
| [pure_simt_gather](./pure_simt_gather) | 样例基于Ascend C纯SIMT编程方式实现Gather算子，从输入张量中采集指定的m行数据，展示离散内存访问类算子的开发方法 |
| [sub](./sub) | 本样例演示了如何通过自定义核函数实现高性能的逐元素减法（Sub）运算。算子核心功能是完成两个形状相同的输入张量x与y的逐元素相减 |
| [tmp_buffer](./tmp_buffer) | 本样例展示了一个支持bfloat16_t数据类型的向量加法（Add）算子，并重点演示了在算子计算过程中使用临时缓冲区（TmpBuf）进行数据转换的典型方法 |
| [vector_add](./vector_add) | 本样例介绍Add算子的核函数直调方法，算子支持单核运行 |