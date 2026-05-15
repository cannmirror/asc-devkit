# size

## 功能说明

获取协作组内线程总数，与接口`num_threads`功能相同。

## 函数原型

```c++
unsigned long long size() const
```

## 参数说明

无

## 返回值说明

协作组内线程总数。

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```c++
    using namespace cooperative_groups;
    __global__ void simt_kernel(...)
    {
        ...
        if (threadIdx.x % 2 == 0) {
            coalesced_group active = coalesced_threads();
            unsigned long long thread_num = active.size(); // 返回16
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
            unsigned long long thread_num = active.size(); // 返回16
        }
        ...
    }
    ```
