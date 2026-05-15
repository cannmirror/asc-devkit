# coalesced_group简介

在SIMT架构中，硬件层面上处理器以32个线程为一组（一个Warp）来执行线程。如果核函数代码中存在数据依赖的条件分支，则会导致Warp内的线程出现分支，那么Warp会串行执行每个分支，在执行某个分支时会禁用不在该路径上的线程。那些在路径上保持活跃的线程被称为“合并线程”。`coalesced_group`用于发现并创建一个包含所有合并线程的协作组。

> [!CAUTION]注意 
>使用coalesced_group时需关注SIMT架构[不支持独立线程调度]()。

## 需要包含的头文件

```c++
#include <simt_api/cooperative_groups.h>
```

## Public成员函数

```c++
- void sync() const
- unsigned long long num_threads() const
- unsigned long long thread_rank() const
- unsigned long long meta_group_size() const
- unsigned long long meta_group_rank() const
- T shfl(T var, unsigned int src_rank) const
- T shfl_up(T var, int delta) const
- T shfl_down(T var, int delta) const
- int any(int predicate) const
- int all(int predicate) const
- unsigned int ballot(int predicate) const
- unsigned long long size() const
```
