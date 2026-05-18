# thread_block简介

`thread_block`是对线程块的抽象，它代表了核函数的启动配置（启动的线程块个数，每个线程块内的线程数量等）。

## 需要包含的头文件

```C++
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
