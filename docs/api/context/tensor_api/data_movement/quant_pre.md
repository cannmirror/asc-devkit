# 随路量化

随路量化是矩阵计算结果从L0C Buffer搬出到Global Memory或Unified Buffer过程中的数据转换能力。L0C Buffer到Global Memory/Unified Buffer搬运根据是否传入`quant`参数自动选择不量化、直接cast、scalar量化或tensor量化模式。

## 调用形式

不传入`quant`参数时，接口根据源类型和目的类型选择不量化输出或直接cast输出：

```cpp
auto atom = MakeCopy(CopyL0C2GM{}).with(fixpipeParams);
Copy(atom, dst, src);
```

传入`uint64_t`类型的`quant`参数时，表示scalar量化。整个C矩阵共用同一个量化参数：

```cpp
uint64_t quant = 1;
auto atom = MakeCopy(CopyL0C2GM{}).with(fixpipeParams);
Copy(atom, dst, src, quant);
```

传入Tensor API张量作为`quant`参数时，表示tensor量化。量化参数通常位于L1 Buffer，元素类型为`uint64_t`，每个channel维度对应一个量化参数：

```cpp
auto quant = MakeTensor(MakeMemPtr(quantAddr),
    MakeFrameLayout<NDExtLayoutPtn>(1, n));
auto atom = MakeCopy(CopyL0C2GM{}).with(fixpipeParams);
Copy(atom, dst, src, quant);
```

## 支持模式

|源类型|目的类型|不传`quant`|传入scalar `quant`|传入tensor `quant`|
|--------|--------|--------|--------|--------|
|`int32_t`|`int8_t`、`uint8_t`|不支持|`REQ8`|`VREQ8`|
|`float`|`int8_t`、`uint8_t`|不支持|`QF322B8_PRE`|`VQF322B8_PRE`|
|`float`|`fp8_e4m3fn_t`|不支持|`QF322FP8_PRE`|`VQF322FP8_PRE`|
|`float`|`hifloat8_t`|不支持|`QF322HIF8_PRE`或`QF322HIF8_PRE_HYBRID`|`VQF322HIF8_PRE`或`VQF322HIF8_PRE_HYBRID`|
|`int32_t`|`half`|不支持|`DEQF16`|`VDEQF16`|
|`float`|`half`|`F322F16`直接cast|`QF322F16_PRE`|`VQF322F16_PRE`|
|`int32_t`|`bfloat16_t`|不支持|`QS322BF16_PRE`|`VQS322BF16_PRE`|
|`float`|`bfloat16_t`|`F322BF16`直接cast|`QF322BF16_PRE`|`VQF322BF16_PRE`|
|`int32_t`|`int32_t`|不量化输出|不支持|不支持|
|`float`|`float`|不量化输出|`QF322F32_PRE`|`VQF322F32_PRE`|

`fp8_e4m3fn_t`、`hifloat8_t`、`QF322F32_PRE`、`QF322F16_PRE`、`QF322BF16_PRE`、`QS322BF16_PRE`及对应tensor量化模式适用于Ascend 950PR/Ascend 950DT。

## 量化参数

scalar量化参数和tensor量化参数的元素类型均为`uint64_t`。

## 约束说明

- tensor量化参数张量应位于L1 Buffer，元素类型为`uint64_t`，地址要求32字节对齐。
- 量化参数不能为INF、NaN或非规格化数。
- `RoundMode::HYBRID`仅在源类型为`float`、目的类型为`hifloat8_t`的量化输出场景支持。
- 不传入`quant`时，仅支持不量化输出或F32到F16/BF16的直接cast输出。
