# shfl_up

## 功能说明

获取`thread_block_tile`组内当前线程向前偏移`delta`的线程的数据。

## 函数原型

```c++
template <typename T>
T shfl_up(T var, unsigned int delta) const
```

## 参数说明

**表 1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| var | 输入 | 线程用于交换的输入操作数。支持的数据类型为：half、int32_t、uint32_t、float、half2、int64_t、uint64_t。 |
| delta | 输入 | 期望获取的var值所在线程在组内相对当前线程的向前偏移值。 |

## 返回值说明

协作组内当前线程向前偏移delta的线程输入的var值。

## 约束说明

无

## 调用示例

以4个线程为一组划分线程块，获取协作组内当前线程向前偏移delta的线程输入的var值。

**图 1**  shfl_up接口返回结果示意图  
![](../../../figures/thread_block_tile_shfl_up.png "thread_block_tile_shfl_up")

- SIMT编程场景：

    ```c++
    using namespace cooperative_groups;
    __global__ void simt_kernel(...)
    {
        ...
        thread_block block = this_thread_block();
        auto tile4 = tiled_partition<4>(block);
        uint32_t result = tile4.shfl_up(threadIdx.x + 100, 2);
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
        uint32_t result = tile4.shfl_up(threadIdx.x + 100, 2);
        ...
    }
    ```
