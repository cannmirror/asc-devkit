# meta_group_rank

## 功能说明

获取当前线程所在的组在从父组划分出的子组集合中的排名。

## 函数原型

```C++
unsigned long long meta_group_rank() const
```

## 参数说明

无

## 返回值说明

当前线程所在的组在从父组划分出的子组集合中的排名。

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```c++
    using namespace cooperative_groups;
    __global__ void simt_kernel(...)
    {
        ...
        thread_block block = this_thread_block();
        auto tile4 = tiled_partition<4>(block);
        unsigned long long group_rank = tile4.meta_group_rank();
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
        unsigned long long group_rank = tile4.meta_group_rank();
        ...
    }
    ```