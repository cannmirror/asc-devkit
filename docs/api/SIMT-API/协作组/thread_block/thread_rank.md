# thread_rank

## 函数功能

获取当前线程在协作组内的排名（threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y）。

## 函数原型

```c++
static unsigned int thread_rank()
```

## 参数说明

无

## 返回值说明

当前线程在协作组内的排名。

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
        unsigned int rank = g.thread_rank();
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
        unsigned int rank = g.thread_rank();
        ...
    }
    ```