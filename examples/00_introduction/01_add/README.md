# Add样例介绍
## 概述
基于Ascend C的Add算子的<<<>>>直调方法，支持main函数和kernel函数在同一个cpp文件中实现。

## 算子开发样例
|  目录名称                                                   |  功能描述                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [basic_api_memory_allocator_add](./basic_api_memory_allocator_add) | 基于静态Tensor方式编程的场景下Add算子的实现方法 |
| [basic_api_tque_add](./basic_api_tque_add) | 本样例以Add算子为样例，使用tque管理内存，使用静态Tensor编程方法进行Add算子的编程 |
| [micro_api_add](./micro_api_add) | 本样例介绍Add算子的核函数直调方法，通过微指令API直接对芯片中涉及Vector计算的寄存器进行操作，算子支持单核运行 |