# num_threads

## 功能说明

获取`thread_block_tile`组内线程的总数。

## 函数原型

```C++
unsigned long long num_threads() const
```

## 参数说明

无

## 返回值说明

组内线程的总数。

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
        unsigned long long thread_num = tile4.num_threads();    // 返回4
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
        unsigned long long thread_num = tile4.num_threads();    // 返回4
        ...
    }
    ```