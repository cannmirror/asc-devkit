# num_threads

## 功能说明

获取`coalesced_group`组内线程总数。

## 函数原型

```c++
unsigned long long num_threads() const
```

## 参数说明

无

## 返回值说明

`coalesced_group`组内线程总数。

## 约束说明

无

## 调用示例

示例代码中偶数id的线程使用`coalesced_group`进行协同，奇数线程独立执行业务，一个Warp中共有16个偶数id的线程。

- SIMT编程场景：

    ```c++
    using namespace cooperative_groups;
    __global__ void simt_kernel(...)
    {
        ...
        if (threadIdx.x % 2 == 0) {
            coalesced_group active = coalesced_threads();
            unsigned long long thread_num = active.num_threads(); // 返回16
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
            unsigned long long thread_num = active.num_threads(); // 返回16
        }
        ...
    }
    ```
