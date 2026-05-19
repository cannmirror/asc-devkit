# sync

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

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

必须保证线程块内所有线程都能执行到同一个`sync()`调用，否则会导致卡死。

## 调用示例

- SIMT编程场景：

    ```c++
    using namespace cooperative_groups;
    __global__ void simt_kernel(...)
    {
        ...
        thread_block g = this_thread_block();
        g.sync(); // 同步线程块内所有线程
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
        g.sync(); // 同步线程块内所有线程
        ...
    }
    ```