# shfl

## 功能说明

协作组内线程的数据交换，不通过共享内存实现直接读取组内某个线程的数据。

## 函数原型

```c++
T shfl(T var, unsigned int src_rank) const
```

## 参数说明

**表 1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| var | 输入 | 线程用于交换的输入操作数。 |
| src_rank | 输入 | 期望获取的var值所在的线程在组内的排名。 |

## 返回值说明

协作组内指定线程的var值。

## 约束说明

无

## 调用示例

示例代码中的条件分支将一个warp中所有线程id是偶数的线程组成coalesced\_group协作组，组内各线程shfl接口返回结果如下图所示。

**图 1**  shfl结果示意图  
![](../../../figures/coalesced_group_shfl.png)

- SIMT编程场景：

    ```c++
    using namespace cooperative_groups;
    __global__ void simt_kernel(...)
    {
        ...
        if (threadIdx.x % 2 == 0) {
            coalesced_group active = coalesced_threads();
            uint32_t result = active.shfl(threadIdx.x + 100, 3);
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
            uint32_t result = active.shfl(threadIdx.x + 100, 3);
        }
        ...
    }
    ```
