# group_index

## 函数功能

获取当前线程块在网格（grid）中的三维索引，返回值与内置变量`blockIdx`相同。

## 函数原型

```c++
static dim3 group_index()
```

## 参数说明

无

## 返回值说明

当前线程块在网格中的三维索引。

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```c++
    using namespace cooperative_groups;
    __global__ void simt_kernel(...)
    {
        ...
        thread_block g = this_thread_block();
        dim3 block_idx = g.group_index();
        ...
    }
    ```

- SIMD与SIMT混合编程场景：

    ```c++
    using namespace cooperative_groups;
    __simt_vf__ inline void simt_kernel(...)
    {
        ...
        thread_block g = this_thread_block();
        dim3 block_idx = g.group_index();
        ...
    }
    ```