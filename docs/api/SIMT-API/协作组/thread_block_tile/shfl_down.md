# shfl_down

## 功能说明

协作组内线程的数据交换，获取协作组内当前线程向后偏移delta的线程输入的var值。

## 函数原型

```C++
T shfl_down(T var, int delta) const
```

## 参数说明

**表 1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| var | 输入 | 线程用于交换的输入操作数。 |
| delta | 输入 | 期望获取的var值所在线程在组内相对当前线程的向后偏移值。 |

## 返回值说明

协作组内当前线程向后偏移delta的线程输入的var值。

## 约束说明

无

## 调用示例

以4个线程为一组划分线程块，获取协作组内当前线程向后偏移delta的线程输入的var值。

**图 1**  shfl_down接口返回结果示意图  
![](../../../figures/thread_block_tile_shfl_down.png "thread_block_tile_shfl_down")

- SIMT编程场景：

    ```c++
    using namespace cooperative_groups;
    __global__ void simt_kernel(...)
    {
        ...
        thread_block block = this_thread_block();
        auto tile4 = tiled_partition<4>(block);
        uint32_t result = tile4.shfl_down(threadIdx.x + 100, 2);
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
        uint32_t result = tile4.shfl_down(threadIdx.x + 100, 2);
        ...
    }
    ```
