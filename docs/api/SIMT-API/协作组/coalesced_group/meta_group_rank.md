# meta_group_rank

## 功能说明

获取当前线程所在的协作组位于其父组划分出的所有子组的集合中的排名。

## 函数原型

```c++
unsigned long long meta_group_rank() const
```

## 参数说明

无

## 返回值说明

前组在从父组划分出的子组集合中的排名。

- 如果该组是通过`coalesced_threads`创建的，则meta_group_rank()的值为0。

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```c++
    using namespace cooperative_groups;
    __global__ void simt_kernel(...)
    {
        ...
        if (threadIdx.x % 2 == 0) {
            coalesced_group active = coalesced_threads();
            unsigned long long thread_num = active.meta_group_rank(); // 返回0
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
            unsigned long long thread_num = active.meta_group_rank(); // 返回0
        }
        ...
    }
    ```
