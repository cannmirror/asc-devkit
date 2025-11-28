# asc_bf162int32

## AI处理器支持情况

| AI处理器类型 | 是否支持 |
| :-----------| :------: |
| Ascend 910C |    √    |
| Ascend 910B |    √    |

## 功能说明

将bfloat16_t类型数据转换为int32_t类型，并支持多种舍入模式：

- RINT舍入模式：四舍五入成双舍入
- ROUND舍入模式：四舍五入舍入
- FLOOR舍入模式：向负无穷舍入
- CEIL舍入模式：向正无穷舍入
- TRUNC舍入模式：向零舍入

## 函数原型

- 前n个数据计算
  ```cpp
  // RINT舍入模式
  __aicore__ inline void asc_bf162int32_r(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, uint32_t count)

  // ROUND舍入模式
  __aicore__ inline void asc_bf162int32_a(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, uint32_t count)

  // FLOOR舍入模式
  __aicore__ inline void asc_bf162int32_f(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, uint32_t count)

  // CEIL舍入模式
  __aicore__ inline void asc_bf162int32_c(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, uint32_t count)

  // TRUNC舍入模式
  __aicore__ inline void asc_bf162int32_z(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, uint32_t count)
  ```

- 高维切分计算
  ```cpp
  // RINT舍入模式
  __aicore__ inline void asc_bf162int32_r(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, const asc_unary_config& config)

  // ROUND舍入模式
  __aicore__ inline void asc_bf162int32_a(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, const asc_unary_config& config)

  // FLOOR舍入模式
  __aicore__ inline void asc_bf162int32_f(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, const asc_unary_config& config)

  // CEIL舍入模式
  __aicore__ inline void asc_bf162int32_c(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, const asc_unary_config& config)

  // TRUNC舍入模式
  __aicore__ inline void asc_bf162int32_z(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, const asc_unary_config& config)
  ```

- 同步计算
  ```cpp
  // RINT舍入模式
  __aicore__ inline void asc_bf162int32_r_sync(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, uint32_t count)

  // ROUND舍入模式
  __aicore__ inline void asc_bf162int32_a_sync(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, uint32_t count)

  // FLOOR舍入模式
  __aicore__ inline void asc_bf162int32_f_sync(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, uint32_t count)

  // CEIL舍入模式
  __aicore__ inline void asc_bf162int32_c_sync(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, uint32_t count)

  // TRUNC舍入模式
  __aicore__ inline void asc_bf162int32_z_sync(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, uint32_t count)
  ```

## 参数说明

表1 参数说明

| 参数名 | 输入/输出 | 描述 |
| :----| :-----| :-----|
| dst | 输出 | 目的操作数。 |
| src  | 输入 | 源操作数。|
| count | 输入 | 参与计算的元素个数。 |
| config  | 输入     | 在非连续场景下使用的计算配置参数。请参考[asc_unary_config](../struct/asc_unary_config.md)|

## 返回值说明

无

## 流水类型

PIPE_TYPE_V

## 约束说明

- dst、src的起始地址需要32字节对齐。
- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。

## 调用示例

```cpp
// 假设src操作数包含128个bfloat16_t类型的数据，dst操作数包含128个int32_t类型的数据。
uint64_t offset = 0;
__ubuf__ bfloat16_t* src = (__ubuf__ bfloat16_t*)asc_get_phy_buf_addr(offset);
offset += 128 * sizeof(bfloat16_t);
__ubuf__ int32_t* dst = (__ubuf__ int32_t*)asc_get_phy_buf_addr(offset);
...... // 将源操作数搬运到src0、src1.
asc_bf162int32_a(dst, src, 128);
...... // 使用dst中的数据进行后续计算或数据搬运操作。
```