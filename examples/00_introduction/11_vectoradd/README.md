# VectorAdd样例介绍
## 概述
基于Ascend C的VectorAdd算子的<<<>>>直调方法，支持main函数和kernel函数在同一个cpp文件中实现。

## 算子开发样例
|  目录名称                                                   |  功能描述                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [add](./add) | 本样例介绍Add算子的核函数直调方法，算子支持单核运行 |
| [add_broadcast](./add_broadcast) | 本样例介绍Add算子的核函数直调方法，多核&tiling场景下增加输入Broadcast |
| [add_tbuf](./add_tbuf) | 本样例介绍Add算子的核函数直调方法，算支持的数据类型有：bfloat16_t，算子支持单核运行，算子内部使用TmpBuf |
| [add_tiling](./add_tiling) | 本样例介绍Add算子的核函数直调方法，支持的数据类型有：bfloat16_t/int8_t/float/half/int16_t/int32_t，算子支持多核运行、支持核间数据均分或不均分场景并且支持尾块处理 |
| [add_visualization](./add_visualization) | 本样例介绍Add算子的核函数直调方法，算子支持批量张量相加，通过流水线机制实现高效数据搬运与计算。算子支持单核运行 |