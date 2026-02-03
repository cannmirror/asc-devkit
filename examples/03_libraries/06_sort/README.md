# 排序操作算子样例介绍

## 概述

本样例集介绍了排序操作算子不同特性的典型用法，给出了对应的端到端实现。

## 算子开发样例

| 目录名称               |  功能描述                                             |
|--------------------| --------------------------------------------------- |
| [mysort](./mysort) | 本样例基于Kernel直调算子工程，介绍了调用MySort高阶API实现mysort单算子，将已经排好序的最多4条队列，合并排列成1条队列，结果按照score域由大到小排序 |
| [mrg_sort4](./mrg_sort4) | 本样例介绍基础api MrgSort4的调用，该api的功能：将已经排好序的最多4条Region Proposals队列，排列合并成1条队列，结果按照score域由大到小排序 |
| [proposal_concat](./proposal_concat) | 本样例介绍基础api ProposalConcat的调用，该api的功能：将连续元素合入Region Proposal内对应位置，每次迭代会将16个连续元素合入到16个Region Proposal的对应位置里 |
| [proposal_extract](./proposal_extract) | 本样例介绍基础api ProposalExtract的调用，该api的功能：与ProposalConcat相反，从Region Proposal内将相应位置的单个元素抽取后重排，每次迭代处理16个Region Proposals，抽取16个元素后连续排列 |
| [rp_sort16](./rp_sort16) | 本样例介绍基础api RpSort16的调用，该api的功能：根据Region Proposals中的score域对其进行排序（score大的排前面），每次排16个Region Proposals |
| [sort](./sort)     | 本样例基于Kernel直调算子工程，介绍了调用Sort高阶API实现sort单算子，排序函数，按照数值大小进行降序排序 |
| [sort32](./sort32) | 本样例基于Sort32实现排序操作，该接口一次迭代可以完成32个数的排序 |
| [topk](./topk)     | 本样例基于Kernel直调算子工程，介绍了调用TopK高阶API实现topk单算子，获取最后一个维度的前k个最大值或最小值及其对应的索引 |