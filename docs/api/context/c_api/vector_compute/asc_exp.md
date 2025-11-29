# asc_exp

## AI处理器支持情况

|AI处理器类型|是否支持|
| :------------ | :------------: |
| <term>Ascend 910C</term> | √ |
| <term>Ascend 910B</term> | √ |

## 功能说明

按元素取自然指数，计算公式如下。
$$
dst_i = e^{src_i}
$$

## 函数原型

- 前n个数据计算
```cpp
__aicore__ inline void asc_exp(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count)
__aicore__ inline void asc_exp(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
```

- 高维切分计算
```cpp
__aicore__ inline void asc_exp(__ubuf__ half* dst, __ubuf__ half* src, const asc_unary_config& config)
__aicore__ inline void asc_exp(__ubuf__ float* dst, __ubuf__ float* src, const asc_unary_config& config)
```

- 同步计算
```cpp
__aicore__ inline void asc_exp_sync(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count)
__aicore__ inline void asc_exp_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
```

## 参数说明

|参数名|输入/输出|描述|
| ------------ | ------------ | ------------ |
|dst|输出|目的操作数。|
|src|输入|源操作数。|
|count|输入|参与计算的元素个数。|
|config|输入|在非连续场景下使用的计算配置参数。<br/>详细说明请参考[asc_unary_config](../struct/asc_unary_config.md)。|

## 返回值说明

无

## 流水类型

PIPE_TYPE_V

## 约束说明

- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。
- dst、src的起始地址需要32字节对齐。

## 调用示例

```cpp
// total_length指参与计算的数据总长度
uint64_t offset = 0;
__ubuf__ half* src = (__ubuf__ half*)asc_get_phy_buf_addr(offset);
offset += total_length * sizeof(half);
__ubuf__ half* dst = (__ubuf__ half*)asc_get_phy_buf_addr(offset);
asc_exp(dst, src, total_length);
```