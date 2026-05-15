# size

## 函数功能

获取组内线程总数，与接口`num_threads`功能相同。

## 函数原型

```c++
static unsigned int size()
```

## 参数说明

无

## 返回值说明

组内线程总数。

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
        unsigned int thread_num = g.size();
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
        unsigned int thread_num = g.size();
        ...
    }
    ```