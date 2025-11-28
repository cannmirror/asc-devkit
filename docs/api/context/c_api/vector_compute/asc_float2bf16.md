# asc_float2bf16

## AI处理器支持情况

| AI处理器类型          | 是否支持|
| :-----------------------| :-----:|
| Ascend 910C |   √   |
| Ascend 910B |   √   |

## 功能说明

将Float类型数据转换为Bfloat16类型，并支持多种舍入模式：

- RINT舍入模式：四舍六入五成双舍入
- ROUND舍入模式：四舍五入舍入
- FLOOR舍入模式：向负无穷舍入
- CEIL舍入模式：向正无穷舍入
- TRUNC舍入模式：向零舍入

## 函数原型

- 前n个数据连续计算
  - RINT舍入模式

    ```cpp
    __aicore__ inline void asc_float2bf16_r(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)
    ```

  - ROUND舍入模式

    ```cpp
    __aicore__ inline void asc_float2bf16_a(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)
    ```

  - FLOOR舍入模式

    ```cpp
    __aicore__ inline void asc_float2bf16_f(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)
    ```

  - CEIL舍入模式

    ```cpp
    __aicore__ inline void asc_float2bf16_c(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)
    ```

  - TRUNC舍入模式

    ```cpp
    __aicore__ inline void asc_float2bf16_z(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)
    ```

- 高维切分计算

  - RINT舍入模式

    ```cpp
    __aicore__ inline void asc_float2bf16_r(__ubuf__ half* dst, __ubuf__ float* src, const asc_unary_config& config)
    ```

  - ROUND舍入模式

    ```cpp
    __aicore__ inline void asc_float2bf16_a(__ubuf__ half* dst, __ubuf__ float* src, const asc_unary_config& config)
    ```

  - FLOOR舍入模式

    ```cpp
    __aicore__ inline void asc_float2bf16_f(__ubuf__ half* dst, __ubuf__ float* src, const asc_unary_config& config)
    ```

  - CEIL舍入模式

    ```cpp
    __aicore__ inline void asc_float2bf16_c(__ubuf__ half* dst, __ubuf__ float* src, const asc_unary_config& config)
    ```

  - TRUNC舍入模式

    ```cpp
    __aicore__ inline void asc_float2bf16_z(__ubuf__ half* dst, __ubuf__ float* src, const asc_unary_config& config)
    ```

- 同步计算

  - RINT舍入模式

    ```cpp
    __aicore__ inline void asc_float2bf16_r_sync(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)
    ```

  - ROUND舍入模式

    ```cpp
    __aicore__ inline void asc_float2bf16_a_sync(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)
    ```

  - FLOOR舍入模式

    ```cpp
    __aicore__ inline void asc_float2bf16_f_sync(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)
    ```

  - CEIL舍入模式

    ```cpp
    __aicore__ inline void asc_float2bf16_c_sync(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)
    ```

  - TRUNC舍入模式

    ```cpp
    __aicore__ inline void asc_float2bf16_z_sync(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)
    ```

## 参数说明

表1 参数说明

| 参数名 | 输入/输出 | 描述 |
| :----:| :-----:| :-----:|
| dst | 输出 | 目的操作数。 |
| src  | 输入 | 源操作数。|
| count | 输入 | 参与计算的元素个数。 |
| config | 输入 | 在非连续场景下使用的计算配置参数。|

## 返回值说明

无

## 流水类型

PIPE_TYPE_V

## 约束说明

- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。
- dst、src的起始地址需要32字节对齐。

## 调用示例

```cpp
// total_length指参与计算的数据长度
uint64_t offset = 0;
__ubuf__ float* src = (__ubuf__ float*)asc_get_phy_buf_addr(0);
offset += total_length * sizeof(float);
__ubuf__ bfloat16_t* dst = (__ubuf__ bfloat16_t*)asc_get_phy_buf_addr(offset);
asc_float2bf16_r(dst, src, total_length);
```
