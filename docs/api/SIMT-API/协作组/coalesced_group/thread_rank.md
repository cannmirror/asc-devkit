# thread_rank

## 功能说明

获取当前线程在组内的排名。

## 函数原型

```c++
unsigned long long thread_rank() const
```

## 参数说明

无

## 返回值说明

当前线程在组内的排名。

## 约束说明

无

## 调用示例

示例代码中的条件分支将一个warp中所有线程id是偶数的线程组成`coalesced_group`协作组，组内各线程`thread_rank`接口返回结果如下图所示。

**图 1**  coalesced_group_rank  
![](../../../figures/coalesced_group_rank.png "coalesced_group_rank")

- SIMT编程场景：

    ```c++
    using namespace cooperative_groups;
    __global__ void simt_kernel(...)
    {
        ...
        if (threadIdx.x % 2 == 0) {
            coalesced_group active = coalesced_threads();
            unsigned long long rank = active.thread_rank();
        }
        ...
    }
    ```

- SIMD与SIMT混合编程场景：

    ```c++
    using namespace cooperative_groups;
    __simt_vf__ inline void simt_kernel(...)
    {
        ...
        if (threadIdx.x % 2 == 0) {
            coalesced_group active = coalesced_threads();
            unsigned long long rank = active.thread_rank();
        }
        ...
    }
    ```
