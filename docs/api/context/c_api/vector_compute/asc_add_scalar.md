# asc_add_scalar

## AI处理器支持情况

| AI处理器类型     | 是否支持 |
| ----------- | ---- |
| Ascend 910C | √    |
| Ascend 910B | √    |

## 功能说明

执行矢量加法运算。计算公式如下：

$$
dst_i = src0_i + a
$$

## 函数原型

- 连续数据计算

```cpp
__aicore__ inline void asc_add_scalar(_ubuf__ int16_t* dst, _ubuf__ int16_t* src, int16_t a, uint32_t count)
__aicore__ inline void asc_add_scalar(_ubuf__ half* dst, _ubuf__ half* src, half a, uint32_t count)
__aicore__ inline void asc_add_scalar(_ubuf__ int32_t* dst, _ubuf__ int32_t* src, int32_t a, uint32_t count)
__aicore__ inline void asc_add_scalar(_ubuf__ float* dst, _ubuf__ float* src, float a, uint32_t count)
```

- 高维切分计算

```cpp
_aicore__ inline void asc_add_scalar(ubuf__ int16_t* dst, _ubuf__ int16_t* src, int16_t a, const asc_unary_config& config)
__aicore__ inline void asc_add_scalar(_ubuf__ half* dst, _ubuf__ half* src, half a, const asc_unary_config& config)
__aicore__ inline void asc_add_scalar(_ubuf__ int32_t* dst, _ubuf__ int32_t* src, int32_t a, const asc_unary_config& config)
__aicore__ inline void asc_add_scalar(_ubuf__ float* dst, _ubuf__ float* src, float a, const asc_unary_config& config)
```

- 同步计算

```cpp
__aicore__ inline void asc_add_scalar_sync(_ubuf__ int16_t* dst, _ubuf__ int16_t* src, int16_t a, uint32_t count)
__aicore__ inline void asc_add_scalar_sync(_ubuf__ half* dst, _ubuf__ half* src, half a, uint32_t count)
__aicore__ inline void asc_add_scalar_sync(_ubuf__ int32_t* dst, _ubuf__ int32_t* src, int32_t a, uint32_t count)
__aicore__ inline void asc_add_scalar_sync(_ubuf__ float* dst, _ubuf__ float* src, float a, uint32_t count)
```

## 参数说明

表1 参数说明

| 参数名       | 输入/输出 | 描述               |
| --------- | ----- | ---------------- |
| dst       | 输出    | 目的操作数。            |
| src | 输入    | 源操作数。            |
| a | 输入    | 标量源操作数。       |
| count     | 输入    | 参与连续计算的元素个数。      |
| config    | 输入    | 在高维切分计算场景下使用的计算配置参数。<br/>详细说明请参考[asc_unary_config](../struct/asc_unary_config.md)。 |


## 返回值说明

无

## 流水类型

PIPE_TYPE_V

## 约束说明

- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。
- dst、src的起始地址需要32字节对齐。

## 调用示例

```cpp
//total_length 指参与计算的数据长度
half a = 10;
uint64_t offset = 0;
__ubuf__ half* src0 = (__ubuf__ half*)asc_get_phy_buf_addr(0);
offset += total_length * sizeof(half);
__ubuf__ half* src = (__ubuf__ half*)asc_get_phy_buf_addr(offset);
asc_add_scalar(dst, src, a, total_length);
```
