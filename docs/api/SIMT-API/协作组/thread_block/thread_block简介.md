# thread_block简介

协作组中引入一种新的数据类型`thread_block`用于在核函数中显式地表达线程块这一概念。

## 需要包含的头文件

```
#include <simt_api/cooperative_groups.h>
```

## Public成员函数

```c++
static void sync()
static unsigned int thread_rank()
static dim3 group_index()
static dim3 thread_index()
static dim3 dim_threads()
static unsigned int num_threads()
static unsigned int size()
static dim3 group_dim()
```
