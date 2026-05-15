# thread_index

## 函数功能

获取当前线程在线程块中的三维索引。

## 函数原型

```c++
static dim3 thread_index()
```

## 参数说明

无

## 返回值说明

当前线程在线程块中的三维索引。

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
        dim3 thread_idx = g.thread_index();
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
        dim3 thread_idx = g.thread_index();
        ...
    }
    ```