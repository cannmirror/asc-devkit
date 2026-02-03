# Reg矢量计算API样例

## 概述

通过Reg矢量计算API实现自定义算子，分别给出对应的<<<>>>直调实现。

## 算子开发样例

|  目录名称 |  功能描述  |
| -------------------------------------------------- | ---------------------------------------------------- |
| [move_mask_reg](./move_mask_reg) | 本样例演示了SIMD场景下，基于RegBase编程范式下数据从UB到MaskReg之间的搬入搬出。 |
| [move_successive_align](./move_successive_align) | 本样例演示了SIMD场景下，基于RegBase编程范式的连续对齐搬运算子的核函数直调方法，算子支持单核运行。 |
| [move_successive_unalign](./move_successive_unalign) | 本样例演示了SIMD场景下，基于RegBase编程范式的连续非对齐搬运算子的核函数直调方法，算子支持单核运行。 |
| [move_unsuccessive_align](./move_unsuccessive_align) | 本样例演示了SIMD场景下，基于RegBase编程范式的非连续对齐搬运算子的核函数直调方法，算子支持单核运行。 |
| [read_write_sync](./read_write_sync) | 本样例演示了SIMD场景下，基于RegBase编程范式下读操作与写操作之间依赖场景下的同步，样例中使用到了寄存器保序这一关键特性，可以优化读写之间的同步指令。 |
| [vector_add](./vector_add) | 本样例演示了SIMD场景下，基于RegBase编程范式的Add算子的核函数直调方法，算子支持单核运行，不同流水线之间使用VEC_LOAD和VEC_TORE同步。 |
| [write_write_sync](./write_write_sync) | 本样例演示了SIMD场景下，基于RegBase编程范式下写操作与写操作之间依赖场景下的同步。 |