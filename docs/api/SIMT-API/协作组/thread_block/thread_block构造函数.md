# thread_block构造函数

## 函数功能

`thread_block`不提供默认的构造函数，用户使用`this_thread_block`函数获取对象。

## 函数原型

```c++
thread_block this_thread_block()
```

## 参数说明

无

## 返回值说明

返回一个`thread_block`对象。

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
        ...
    }
    ```
