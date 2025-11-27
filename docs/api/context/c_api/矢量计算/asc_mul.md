# 1. asc_mul

## AI处理器支持情况

| AI处理器类型         | 是否支持 |
| :-----------------------| :-----:|
| <term>Ascend 910C</term> | √ |
| <term>Ascend 910B</term> | √ |

### 功能说明

执行矢量乘法运算。计算公式如下

$$
dst_i = src0_i * src1_i
$$

### 函数原型

- 前n个数据计算

    ```cpp
    __aicore__ inline void asc_mul(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1, uint32_t count)
    __aicore__ inline void asc_mul(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, uint32_t count)
    __aicore__ inline void asc_mul(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1, uint32_t count)
    __aicore__ inline void asc_mul(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t count)
    ```

- 高维切分计算

    ```cpp
    __aicore__ inline void asc_mul(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1, const asc_binary_config& config)
    __aicore__ inline void asc_mul(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const asc_binary_config& config)
    __aicore__ inline void asc_mul(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1, const asc_binary_config& config)
    __aicore__ inline void asc_mul(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const asc_binary_config& config)
    ```

- 同步计算

    ```cpp
    __aicore__ inline void asc_mul_sync(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1, uint32_t count)
    __aicore__ inline void asc_mul_sync(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, uint32_t count)
    __aicore__ inline void asc_mul_sync(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1, uint32_t count)
    __aicore__ inline void asc_mul_sync(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t count)
    ```

### 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数地址。 |
| src0、src1 | 输入 | 源操作数地址。 |
| count | 输入 | 参与计算的元素个数。 |
| config | 输入 | 在非连续场景下使用的计算配置参数, 请参考[asc_binary_config](../数据结构/asc_binary_config.md)。 |

### 返回值说明

无

### 流水类型

PIPE_TYPE_V

### 约束说明

- dst、src0、src1的起始地址需要32字节对齐。
- 操作数地址重叠约束请参考[通用地址重叠约束](../通用说明和约束.md#通用地址重叠约束)。

### 调用示例

mask连续模式

```cpp
uint64_t offset = 0;
__ubuf__ half* src0 = (__ubuf__ half*)asc_getphybufaddr(0);
offset += totalLength * sizeof(half);
__ubuf__ half* src1 = (__ubuf__ half*)asc_getphybufaddr(offset);
offset += totalLength * sizeof(half);
__ubuf__ half* dst = (__ubuf__ half*)asc_getphybufaddr(offset);
asc_mul(dst, src0, src1, total_length);
```