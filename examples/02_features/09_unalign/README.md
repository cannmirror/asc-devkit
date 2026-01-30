# 09_unalign样例介绍
## 概述
本样例路径以abs、reduce_min、whole_reduce_sum等算子为示例，展示了数据非32字节对齐场景中的处理方式，包括数据搬入，计算和搬出的处理。

## 算子开发样例
| 目录名称  | 功能描述 |
| --------- | --------- |
| [abs_duplicate](./abs_duplicate/) | 本样例介绍无DataCopyPad的非对齐abs_duplicate算子实现，采用核函数<<<>>>调用，有效降低调度开销，实现高效的算子执行 |
| [abs_gather_mask](./abs_gather_mask) | 本样例介绍无DataCopyPad的非对齐abs_gather_mask算子实现，采用核函数<<<>>>调用，有效降低调度开销，实现高效的算子执行 |
| [abs_pad](./abs_pad/) | 本样例介绍基于基础API实现abs_pad样例，展示了通过Pad一次性清零处理冗余数据 |
| [abs_unpad](./abs_unpad/) | 本样例介绍基于基础API实现abs_unpad样例，展示了通过UnPad去除冗余数据的方法 |
| [reduce_min](./reduce_min/) | 本样例介绍无DataCopyPad的非对齐ReduceMin算子核函数直调方法，采用核函数<<<>>>调用，有效降低调度开销，实现高效的算子执行 |
| [whole_reduce_sum](./whole_reduce_sum/) |本样例介绍非对齐WholeReduceSum算子的核函数直调方法，采用核函数<<<>>>调用，有效降低调度开销，实现高效的算子执行 |
