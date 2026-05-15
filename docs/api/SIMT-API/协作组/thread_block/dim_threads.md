# dim_threads

## 函数功能

获取当前线程块的线程配置，返回值与内置变量`blockDim`相同。

## 函数原型

```c++
static dim3 dim_threads()
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
        dim3 block_dim = g.dim_threads();
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
        dim3 block_dim = g.dim_threads();
        ...
    }
    ```
