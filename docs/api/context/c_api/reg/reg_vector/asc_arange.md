# asc_arange

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## 功能说明

以传入的value为起始值，生成递增/递减的索引，并将生成的索引保存在dst中。

## 函数原型

  ```cpp
  // 递增模式
  __simd_callee__ inline void asc_arange(vector_int8_t& dst, int8_t value)
  __simd_callee__ inline void asc_arange(vector_int16_t& dst, int16_t value)
  __simd_callee__ inline void asc_arange(vector_half& dst, half value)
  __simd_callee__ inline void asc_arange(vector_int32_t& dst, int32_t value)
  __simd_callee__ inline void asc_arange(vector_float& dst, float value)
  // 递减模式
  __simd_callee__ inline void asc_arange_descend(vector_int8_t& dst, int8_t value)
  __simd_callee__ inline void asc_arange_descend(vector_int16_t& dst, int16_t value)
  __simd_callee__ inline void asc_arange_descend(vector_half& dst, half value)
  __simd_callee__ inline void asc_arange_descend(vector_int32_t& dst, int32_t value)
  __simd_callee__ inline void asc_arange_descend(vector_float& dst, float value)
  ```

## 参数说明

| 参数名       | 输入/输出 | 描述                |
| --------- | ----- | ----------------- |
| dst       | 输出    | 目的操作数（矢量数据寄存器）。 |
| value     | 输入    | 源操作数（标量）。 |

矢量数据寄存器的详细说明请参见[reg数据类型定义.md](../reg数据类型定义.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

```cpp
vector_int8_t dst;
int8_t value;
asc_arange(dst, value);
```
