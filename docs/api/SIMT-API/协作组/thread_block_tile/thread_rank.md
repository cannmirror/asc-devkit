# thread_rank

## 功能说明

获取当前线程在协作组内的排名。

## 函数原型

```C++
unsigned long long thread_rank() const
```

## 参数说明

无

## 返回值说明

当前线程在协作组内的排名。

## 约束说明

无

## 调用示例

以4个线程为一组划分线程块，获取协作组内当前线程在组内的排名。

**图 1**   thread_rank接口返回值示意图  
![](../../../figures/thread_block_tile_rank.png "thread_block_tile_rank")

- SIMT编程场景：

    ```c++
    using namespace cooperative_groups;
    __global__ void simt_kernel(...)
    {
        ...
        thread_block block = this_thread_block();
        auto tile4 = tiled_partition<4>(block);
        unsigned long long rank = tile4.thread_rank();
        ...
    }
    ```

- SIMD与SIMT混合编程场景：

    ```c++
    using namespace cooperative_groups;
    __simt_vf__ inline void simt_kernel(...)
    {
        ...
        thread_block block = this_thread_block();
        auto tile4 = tiled_partition<4>(block);
        unsigned long long rank = tile4.thread_rank();
        ...
    }
    ```
