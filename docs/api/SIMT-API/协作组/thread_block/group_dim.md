# group_dim

## 函数功能

获取当前线程块的线程配置，与接口`dim_threads`功能相同。

## 函数原型

```c++
static dim3 group_dim()
```

## 参数说明

无

## 返回值说明

当前线程块的线程配置。

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
        dim3 block_dim = g.group_dim();
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
        dim3 block_dim = g.group_dim();
        ...
    }
    ```