# sync

## 功能说明

同步线程块内所有线程，所有线程都执行到该同步点位置才能继续执行。

## 函数原型

```c++
static void sync()
```

## 参数说明

无

## 返回值说明

无

## 约束说明

必须保证线程块内所有线程都能调用到此接口，否则会导致卡死。

## 调用示例

- SIMT编程场景：

    ```c++
    using namespace cooperative_groups;
    __global__ void simt_kernel(...)
    {
        ...
        thread_block g = this_thread_block();
        g.sync();
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
        g.sync();
        ...
    }
    ```