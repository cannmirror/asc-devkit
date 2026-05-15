# ballot

## 功能说明

判断组内每个活跃线程的输入是否非零。

## 函数原型

```c++
unsigned int ballot(int predicate) const
```

## 参数说明

**表 1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| predicate | 输入 | 操作数。 |

## 返回值说明

32bit的无符号整数：若组内活跃线程输入的predicate不为0，则返回值中与线程rank对应的bit位为1，否则为0。

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
            uint32_t result = active.ballot(1); // 返回0xFFFF
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
            uint32_t result = active.ballot(1); // 返回0xFFFF
        }
        ...
    }
    ```
