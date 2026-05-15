# num_threads

## 函数功能

获取`thread_block`组内线程总数。

## 函数原型

```c++
static unsigned int num_threads()
```

## 参数说明

无

## 返回值说明

组内线程总数，即blockDim.x * blockDim.y * blockDim.z。

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
        unsigned int thread_num = g.num_threads();
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
        unsigned int thread_num = g.num_threads();
        ...
    }
    ```