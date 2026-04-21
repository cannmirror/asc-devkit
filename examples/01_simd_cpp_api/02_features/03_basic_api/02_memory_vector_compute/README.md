# Memory矢量计算算子样例介绍

## 概述

本样例集介绍了Memory矢量计算算子不同特性的典型用法，给出了对应的端到端实现。

## 样例列表

| 目录名称                                                                                                                 | 功能描述                                                                        | 
|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------| 
| [block_reduce_max](./block_reduce_max)                                                                               | 本样例基于BlockReduceMax实现最大值归约，可用于对每个datablock内所有元素求最大值                         |
| [compare](./compare)                                                                                             | 本样例基于Compare、Compares接口完成多场景下的数据比较功能，实现逐元素大小比较。                                              |
| [select](./select)                                                                                               | 本样例基于Select接口完成多种场景下的数据选择功能，根据掩码在两个向量或向量与标量之间选取元素写入目的向量                                                                                            |
| [create_vec_index](./create_vec_index) | 本样例介绍了调用CreateVecIndex实现创建指定起始值的向量索引的方法 |
| [brcb](./brcb) | 本样例基于Brcb实现数据填充，可用于每次取输入张量中的8个数填充到结果张量的8个datablock中 |
| [duplicate](./duplicate) | 本样例基于Duplicate实现数据填充，可用于将一个变量或立即数复制多次并填充到向量中 |
| [gather](./gather)         | 本样例基于GatherMask、Gather、Gatherb等接口完成多种场景模式下的数据选择功能，实现从源操作数中选取元素写入目的操作数。 |
| [mrg_sort](./mrg_sort) | 本样例基于Sort32和MrgSort基础API实现将已排好序的最多4条队列，合并成1条队列，结果按照score域由大到小排序 |
| [mrg_sort4](./mrg_sort4) | 本样例介绍基础api MrgSort4的调用，该api的功能：将已经排好序的最多4条Region Proposals队列，排列合并成1条队列，结果按照score域由大到小排序 |
| [proposal_concat](./proposal_concat) | 本样例介绍基础api ProposalConcat的调用，该api的功能：将连续元素合入Region Proposal内对应位置，每次迭代会将16个连续元素合入到16个Region Proposal的对应位置里 |
| [proposal_extract](./proposal_extract) | 本样例介绍基础api ProposalExtract的调用，该api的功能：与ProposalConcat相反，从Region Proposal内将相应位置的单个元素抽取后重排，每次迭代处理16个Region Proposals，抽取16个元素后连续排列 |
| [rp_sort16](./rp_sort16) | 本样例介绍基础api RpSort16的调用，该api的功能：根据Region Proposals中的score域对其进行排序（score大的排前面），每次排16个Region Proposals |
| [reduce_computation](./reduce_computation)         | 本样例基于ReduceMax/ReduceMin/ReduceSum接口实现归约计算              |
| [repeat_reduce_sum](./repeat_reduce_sum)                                                                             | 本样例基于RepeatReduceSum实现求和归约，可用于每个repeat内所有数据求和                   |
| [transpose](./transpose) | 本样例基于Transpose、TransDataTo5HD接口实现数据转置功能，包括16*16的二维矩阵数据块转置，[N,C,H,W]与[N,H,W,C]四维矩阵互相转换以及NCHW格式转换成NC1HWC0格式。 |
| [whole_reduce_min_max_sum](./whole_reduce_min_max_sum)                                                                               | 本样例介绍归约类接口在多种场景下的使用方法，包括WholeReduceMax、WholeReduceMin、WholeReduceSum、RepeatReduceSum，以及WholeReduceMax/Min配合GetReduceRepeatMaxMinSpr获取全局极值及索引的使用方法。                  |