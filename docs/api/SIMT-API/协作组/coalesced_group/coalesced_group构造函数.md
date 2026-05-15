# coalesced_group构造函数

## 函数功能

`coalesced_group`不提供默认的构造函数，用户使用`coalesced_threads`函数创建对象。

## 函数原型

```c++
coalesced_group coalesced_threads()
```

## 参数说明

无

## 返回值说明

返回一个`coalesced_group`对象。

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
            coalesced_group active = coalesced_threads(); // 该coalesced_group中包含一个warp内所有线程id为偶数的线程
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
            coalesced_group active = coalesced_threads(); // 该coalesced_group中包含一个warp内所有线程id为偶数的线程
        }
        ...
    }
    ```
