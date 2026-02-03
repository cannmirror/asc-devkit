# 归约操作算子样例介绍

## 概述

本样例集介绍了归约操作算子不同特性的典型用法，给出了对应的端到端实现。

## 算子开发样例

| 目录名称                                                                                                                 | 功能描述                                                                        | 
|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------| 
| [block_reduce_max](./block_reduce_max)                                                                               | 本样例基于BlockReduceMax实现最大值归约，可用于对每个datablock内所有元素求最大值                         |
| [get_acc_val](./get_acc_val)                                                                                         | 本样例介绍基础api GetAccVal的调用。GetAccVal的功能是获取ReduceSum接口的计算结果 |
| [get_reduce_max_min_count](./get_reduce_max_min_count)                                                               | 本样例介绍基础api GetReduceMaxMinCount的调用，GetReduceMaxMinCount的功能是获取ReduceMax、ReduceMin连续场景下的最大/最小值以及相应的索引值 |
| [mean](./mean)                                                                                                       | 本样例介绍了基于Ascend C异构混合编程模型演示Mean融合算子的核函数直调实现，根据最后一轴的方向对各元素求平均值                |
| [pair_reduce_sum](./pair_reduce_sum)                                                                                 | 本样例基于PairReduceSum实现求和归约，可用于相邻两个（奇数下标和偶数下标）元素求和                             |
| [reduce_max_computation_of_the_first_n_data_elements](./reduce_max_computation_of_the_first_n_data_elements)         | 本样例基于ReduceMax的tensor前n个数据计算接口实现最大值归约，可用于指定从输入tensor的前n个数据中计算找出最大值及最大值对应的索引位置 |
| [reduce_max_high_dimensional_tensor_sharding_computation](./reduce_max_high_dimensional_tensor_sharding_computation) | 本样例基于ReduceMax的tensor高维切分计算接口实现最大值归约，可用于从所有的输入数据中找出最大值及最大值对应的索引位置，使用mask用于控制每次迭代内参与计算的元素。           |
| [reduce_min_computation_of_the_first_n_data_elements](./reduce_min_computation_of_the_first_n_data_elements)         | 本样例基于ReduceMin的tensor前n个数据计算接口实现最小值归约，可用于指定从输入tensor的前n个数据中找出最小值及最小值对应的索引位置              |
| [reduce_min_high_dimensional_tensor_sharding_computation](./reduce_min_high_dimensional_tensor_sharding_computation) | 本样例基于ReduceMin的tensor高维切分计算接口实现最小值归约，可用于从所有的输入数据中找出最小值及最小值对应的索引位置，使用mask用于控制每次迭代内参与计算的元素。            |
| [reduce_sum_computation_of_the_first_n_data_elements](./reduce_sum_computation_of_the_first_n_data_elements)         | 本样例基于ReduceSum的tensor前n个数据计算接口实现求和归约，可用于指定对输入tensor的前n个数据求和                 |
| [reduce_sum_high_dimensional_tensor_sharding_computation](./reduce_sum_computation_of_the_first_n_data_elements)     | 本样例基于ReduceSum的tensor高维切分计算接口实现求和归约，可用于对所有的输入数据求和，使用mask用于控制每次迭代内参与计算的元素                |
| [reducemean](./reducemean)                                                                                           | 本样例演示了基于ReduceMean高阶API实现reducemean算子。样例对一个多维向量按照指定的维度求平均值                  |
| [reducemin](./reducemin)                                                                                             | 本样例演示了调用ReduceMin高阶API实现reducemin算子。样例对一个多维向量在指定的维度求最小值                     |
| [reducesum_lowlevel](./reducesum_lowlevel)                                                                           | 本样例演示了基于ReduceSum高阶API实现reducesum算子。样例对一个多维向量按照指定的维度进行数据累加                  |
| [repeat_reduce_sum](./repeat_reduce_sum)                                                                             | 本样例基于RepeatReduceSum实现求和归约，可用于每个repeat内所有数据求和                   |
| [sum](./sum)                                                                                                         | 本样例介绍了调用Sum高阶API实现sum算子，获取输入数据最后一个维度的元素总和                                   |
| [whole_reduce_max](./whole_reduce_max)                                                                               | 本样例基于WholeReduceMax实现最大值归约，可用于对每个repeat内所有数据求最大值以及其索引index，返回的索引值为每个repeat内部索引                  |
| [whole_reduce_min](./whole_reduce_min)                                                                               | 本样例基于WholeReduceMin实现获得每个repeat内所有数据的最小值及其索引index的功能，返回的索引值为每个repeat内部索引                  |
| [whole_reduce_sum](./whole_reduce_sum)                                                                               | 本样例基于WholeReduceSum实现对每个repeat内所有数据求和的功能                    |