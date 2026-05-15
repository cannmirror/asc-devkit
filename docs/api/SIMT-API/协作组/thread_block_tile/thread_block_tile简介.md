# thread_block_tile简介

`thread_block_tile`是一个模板类，用于创建指定大小的协作组。

> [!CAUTION]注意 
>SIMT架构不支持独立线程调度，一个`warp`内的各协作组间应避免存在数据依赖，否则可能出现卡死的情况。

## 需包含的头文件

```C++
#inlcude "simt_api/cooperative_groups.h"
```

## Public成员函数

```C++
void sync() const;
unsigned long long num_threads() const;
unsigned long long thread_rank() const;
unsigned long long meta_group_size() const;
unsigned long long meta_group_rank() const;
T shfl(T var, unsigned int src_rank) const;
T shfl_up(T var, int delta) const;
T shfl_down(T var, int delta) const;
T shfl_xor(T var, int delta) const;
int any(int predicate) const;
int all(int predicate) const;
unsigned int ballot(int predicate) const;
unsigned long long size() const;
```