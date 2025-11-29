# asc_copy_gm2ub

## AI处理器支持情况

| AI处理器类型 | 是否支持  |
| :----------------------- | :------: |
| <term>Ascend 910C</term> |    √     |
| <term>Ascend 910B</term> |    √     |

## 功能说明

将数据从Global Memory (GM) 搬运到 Unified Buffer (UB)。

## 函数原型

- 前n个数据搬运

```c++
__aicore__ inline void asc_copy_gm2ub(__ubuf__ void* dst, __gm__ void* src, uint32_t size)
```

- 高维切分搬运

```c++
__aicore__ inline void asc_copy_gm2ub(__ubuf__ void* dst, __gm__ void* src, const asc_copy_config& config)
```

- 同步计算

```c++
__aicore__ inline void asc_copy_gm2ub_sync(__ubuf__ void* dst, __gm__ void* src, uint32_t size)
```

## 参数说明

表1 参数说明
| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst | 输出 | 目的GM地址。 |
| src | 输入 | 源UB地址。 |
| size | 输入 | 搬运数据大小（字节）。 |
| config | 输入 | 在高维切分场景下使用的数据搬运配置参数。详细说明请参考[asc_copy_config](../struct/asc_copy_config.md) 。|

## 返回值说明

无

## 流水类型

PIPE_TYPE_MTE2

## 约束说明

- src的起始地址要求按照对应数据类型所占字节数对齐。
- dst的起始地址要求32字节对齐。
- 如果需要执行多条asc_copy_gm2ub指令，且asc_copy_gm2ub指令的目的地址存在重叠，需要插入同步指令，保证多个asc_copy_gm2ub指令的串行化，防止出现异常数据。
- 同步计算包含同步等待。

## 调用示例

```cpp
//total_length指参与搬运的数据总长度。src是外部输入的half类型的GM内存。
uint64_t offset = 0;
dst = (__ubuf__ half*)asc_get_phy_buf_addr(offset);
asc_copy_gm2ub(dst, src, total_length * sizeof(half));
```
