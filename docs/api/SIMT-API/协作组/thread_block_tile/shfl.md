# shfl

## 功能说明

获取协作组内指定线程的数据。

## 函数原型

```C++
T shfl(T var, unsigned int src_rank) const
```

## 参数说明


**表 1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| var | 输入 | 线程用于交换的输入操作数。 |
| src_rank | 输入 | 期望获取的var值所在的线程在组内的排名。 |

## 返回值说明

指定组内rank的线程传入的var。

## 约束说明

无

## 调用示例

以4个线程为一组划分线程块，获取组内排名为2的线程的var值。

**图 1**   shfl接口返回结果示意图  
![](../../../figures/thread_block_tile_shfl.png "thread_block_tile_shfl")

- SIMT编程场景：

    ```c++
    using namespace cooperative_groups;
    __global__ void simt_kernel(...)
    {
        ...
        thread_block block = this_thread_block();
        auto tile4 = tiled_partition<4>(block);
        uint32_t result = tile4.shfl(threadIdx.x + 100, 2);
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
        uint32_t result = tile4.shfl(threadIdx.x + 100, 2);
        ...
    }
    ```
