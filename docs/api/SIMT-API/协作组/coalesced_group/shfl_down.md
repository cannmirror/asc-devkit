# shfl_down

## 功能说明

获取协作组内当前线程向后偏移`delta`的线程的数据。

## 函数原型

```c++
T shfl_down(T var, int delta) const
```

## 参数说明

**表 1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| var | 输入 | 线程用于交换的输入操作数。 |
| delta | 输入 | 期望获取的var值所在线程在组内相对当前线程的向后偏移值。 |

## 返回值说明

协作组内指定线程的var值。

## 约束说明

无

## 调用示例

示例代码中的条件分支将一个Warp中所有线程id是偶数的线程组成`coalesced_group`协作组，组内各线程`shfl_down`接口返回结果如下图所示。

**图 1**  shfl_down结果示意图  
![](../../../figures/coalesced_group_shfl_down.png "shfl_down结果示意图")

- SIMT编程场景：

    ```c++
    using namespace cooperative_groups;
    __global__ void simt_kernel(...)
    {
        ...
        if (threadIdx.x % 2 == 0) {
            coalesced_group active = coalesced_threads();
            uint32_t result = active.shfl_down(threadIdx.x + 100, 2);
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
            uint32_t result = active.shfl_down(threadIdx.x + 100, 2);
        }
        ...
    }
    ```
