# asc_select

## AI处理器支持情况

| AI处理器类型 | 是否支持  |
| :----------------------- | :------: |
| <term>Ascend 910C</term> |    √     |
| <term>Ascend 910B</term> |    √     |

## 功能说明

执行矢量选择操作，给定两个源操作数src0和src1，根据条件选择元素，得到目的操作数dst。

## 函数原型

- 前n个数据计算

```c++
__aicore__ inline void asc_select(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, uint32_t count)
__aicore__ inline void asc_select(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t count)
```

- 高维切分计算

```c++
__aicore__ inline void asc_select(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const asc_binary_config& config)
__aicore__ inline void asc_select(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const asc_binary_config& config)
```

- 同步计算

```c++
__aicore__ inline void asc_select_sync(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, uint32_t count)
__aicore__ inline void asc_select_sync(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t count)
```

## 参数说明

表1 参数说明
| 参数名       | 输入/输出 | 描述               |
| :--- | :--- | :--- |
| dst       | 输出    | 目的操作数。 |
| src、src1 | 输入    | 源操作数。 |
| count     | 输入    | 参与计算的元素个数。 |
| config | 输入    | 在高维切分场景下使用的计算配置参数。详细说明请参考[asc_binary_config](../struct/asc_binary_config.md) 。 |

## 返回值说明

无

## 流水类型

PIPE_TYPE_V

## 约束说明

- dst、src的起始地址需要32字节对齐
- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。

## 调用示例

```c++
//total_length指参与计算的数据总长度
uint64_t offset = 0;
__ubuf__ half* src0 = (__ubuf__ half*)asc_get_phy_buf_addr(0);
offset += total_length * sizeof(half);
__ubuf__ half* src1 = (__ubuf__ half*)asc_get_phy_buf_addr(offset);
offset += total_length * sizeof(half);
__ubuf__ half* dst = (__ubuf__ half*)asc_get_phy_buf_addr(offset);
asc_select(dst, src0, src1, total_length);
```
