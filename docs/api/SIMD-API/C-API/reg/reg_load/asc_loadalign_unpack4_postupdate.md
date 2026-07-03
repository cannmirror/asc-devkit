# asc_loadalign_unpack4_postupdate

## 产品支持情况

| 产品     | 是否支持 |
| ----------- | :----: |
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

头文件路径：`"c_api/reg_compute/reg_load.h"`。

对齐数据搬运接口，从UB连续对齐搬入目的操作数，实现UNPACK4搬入模式并启用Post Update：按无符号整型u8加载VL/4长度数据并unpack到VL长度u32类型（中间位置补0），接口调用后自动更新源操作数地址。

## 函数原型

```cpp
__simd_callee__ inline void asc_loadalign_unpack4_postupdate(vector_int8_t& dst, __ubuf__ int8_t*& src, int32_t offset)
__simd_callee__ inline void asc_loadalign_unpack4_postupdate(vector_uint8_t& dst, __ubuf__ uint8_t*& src, int32_t offset)
__simd_callee__ inline void asc_loadalign_unpack4_postupdate(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t*& src, int32_t offset)
__simd_callee__ inline void asc_loadalign_unpack4_postupdate(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t*& src, int32_t offset)
__simd_callee__ inline void asc_loadalign_unpack4_postupdate(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t*& src, int32_t offset)
__simd_callee__ inline void asc_loadalign_unpack4_postupdate(vector_int4x2_t& dst, __ubuf__ int4b_t*& src, int32_t offset)
__simd_callee__ inline void asc_loadalign_unpack4_postupdate(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t*& src, int32_t offset)
__simd_callee__ inline void asc_loadalign_unpack4_postupdate(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t*& src, int32_t offset)
__simd_callee__ inline void asc_loadalign_unpack4_postupdate(vector_hifloat8_t& dst, __ubuf__ hifloat8_t*& src, int32_t offset)
```

## 参数说明

| 参数名       | 输入/输出 | 描述               |
| --------- | ----- | ---------------- |
| dst       | 输出    | 目的操作数（矢量数据寄存器）。            |
| src       | 输入/输出 | 源操作数（矢量）的起始地址，接口调用后自动更新。            |
| offset    | 输入    | 偏移量。            |

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
__ubuf__ int8_t* src;
asc_loadalign_unpack4_postupdate(dst, src, 0);
```
