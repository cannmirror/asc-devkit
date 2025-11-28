
# 1. asc_mul_scalar

## AI处理器支持情况

| AI处理器类型         | 是否支持 |
| :-----------------------| :-----:|
| <term>Ascend 910C</term> | √ |
| <term>Ascend 910B</term> | √ |

### 功能说明

执行矢量与标量的乘法运算。计算公式如下

$$
dst_i = src_i * scalar
$$

### 函数原型

- 前n个数据计算

    ```cpp
    __aicore__ inline void asc_mul_scalar(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t a, uint32_t count)
    __aicore__ inline void asc_mul_scalar(__ubuf__ half* dst, __ubuf__ half* src, half a, uint32_t count)
    __aicore__ inline void asc_mul_scalar(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t a, uint32_t count)
    __aicore__ inline void asc_mul_scalar(__ubuf__ float* dst, __ubuf__ float* src, float a, uint32_t count)
    ```

- 高维切分计算

    ```cpp
    __aicore__ inline void asc_mul_scalar(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t a, const asc_unary_config& config)
    __aicore__ inline void asc_mul_scalar(__ubuf__ half* dst, __ubuf__ half* src, half a, const asc_unary_config& config)
    __aicore__ inline void asc_mul_scalar(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t a, const asc_unary_config& config)
    __aicore__ inline void asc_mul_scalar(__ubuf__ float* dst, __ubuf__ float* src, float a, const asc_unary_config& config);
    ```

- 同步计算

    ```cpp
    __aicore__ inline void asc_mul_scalar_sync(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t a, uint32_t count)
    __aicore__ inline void asc_mul_scalar_sync(__ubuf__ half* dst, __ubuf__ half* src, half a, uint32_t count)
    __aicore__ inline void asc_mul_scalar_sync(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t a, uint32_t count)
    __aicore__ inline void asc_mul_scalar_sync(__ubuf__ float* dst, __ubuf__ float* src, float a, uint32_t count)
    ```

### 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数地址。 |
| src | 输入 | 源操作数地址。 |
| a | 输入 | 源操作数。 |
| count | 输入 | 参与计算的元素个数。 |
| config | 输入 | 在非连续场景下使用的计算配置参数, 请参考[asc_unary_config](../数据结构/asc_unary_config.md)。 |

### 返回值说明

无

### 流水类型

PIPE_TYPE_V

### 约束说明

- dst、src的起始地址需要32字节对齐。
- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。

### 调用示例

```cpp
uint64_t offset = 0;
// total_length指参与计算的数据长度
uint64_t offset = 0;
__ubuf__ half* src = (__ubuf__ half*)asc_getphybufaddr(0);
offset += totalLength * sizeof(half);
__ubuf__ half* dst = (__ubuf__ half*)asc_getphybufaddr(offset);
asc_mul_scalar(dst, src, scalar_val, total_length);
```