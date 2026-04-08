# Mmad

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
| Ascend 950PR/Ascend 950DT|√|

## 功能说明

完成矩阵乘加（C += A \* B或C = A \* B + Bias）操作。

## 函数原型

```cpp
template <const MmadTrait& trait = DEFAULT_MMAD_TRAIT, typename T, typename U, typename S, typename Params>
Mmad(const T& dst, const U& fm, const S& filter, const Params& params)

template <const MmadTrait& trait = DEFAULT_MMAD_TRAIT, typename T, typename U, typename S, typename V, typename Params>
Mmad(const T& dst, const U& fm, const S& filter, const V& bias, const Params& params)
```

## 参数说明

**表 1** 模板参数说明

| 参数名 | 描述 |
| :--- | :--- |
| trait | MmadTrait，矩阵乘加特征，默认为`DEFAULT_MMAD_TRAIT`。 |
| T | 目的操作数的数据类型。 |
| U | 左矩阵的数据类型。 |
| S | 右矩阵的数据类型。 |
| V | Bias的数据类型。 |
| Params | params的类型。 |

MmadTrait结构体定义如下：

```cpp
struct MmadTrait {
    int32_t fmOffset = 0;               // 特征图偏移
    bool kDirectionAlign = false;       // K方向对齐方式
    bool cmatrixSource = false;         // 配置C矩阵初始值是否来源于BiasTable Buff。
    bool disableGemv = false;           // 是否禁用GEMV优化。
    MmadType mmadType = MmadType::NORMAL;   // Mmad类型。可选值NORMAL或MX。
};

```

**表 2** 参数说明

| 参数名 | 说明 |
|--------|------|
| dst |输出矩阵C，位于L0C（`__cc__`）。 |
| fm | 左矩阵A，位于L0A（`__ca__`）。 |
| filter |  右矩阵B，位于L0B（`__cb__`）。 |
| bias |  可选，偏置向量，位于L0C（`__cbuf__`）或BiasTable Buff（`__biasbuf__`），带偏置时传入。 |
| params | MmadParams，矩阵乘加参数结构体。 |

**表 3** MmadParams参数说明

| 参数名  | 说明 |
|------|------|
| m | 左矩阵Height，取值范围：m∈[0, 4095] 。默认值为0。 |
| n | 右矩阵Width，取值范围：n∈[0, 4095] 。默认值为0。 |
| k | 左矩阵Width、右矩阵Height，取值范围：k∈[0, 4095] 。默认值为0。 |
| unitFlag | unitFlag是一种Mmad指令和Fixpipe指令细粒度的并行，使能该功能后，硬件每计算完一个分形，计算结果就会被搬出，该功能不适用于在L0C Buffer累加的场景。取值说明如下：<br>0：保留值；<br>2：使能unitFlag，硬件执行完指令之后，不会关闭unitFlag功能；<br>3：使能unitFlag，硬件执行完指令之后，会将unitFlag功能关闭。|
| cmatrixInitVal | 配置C矩阵初始值是否为0。默认值false。<br>true: C矩阵初始值为0；<br>false: C矩阵初始值通过cmatrixSource参数进行配置。|

## 数据类型支持

**表 4**  dst、fm、filter、bias支持的精度类型组合

| 左矩阵 (fm type) | 右矩阵 (filter type) | bias type | 结果矩阵 (dst type) |
| :--- | :--- | :--- | :--- |
| int8_t | int8_t | int32_t | int32_t |
| half | half | float | float |
| float | float | float | float |
| bfloat16_t | bfloat16_t | float | float |
| fp8_e4m3fn_t | fp8_e4m3fn_t | float | float |
| fp8_e4m3fn_t | fp8_e5m2_t | float | float |
| fp8_e5m2_t | fp8_e4m3fn_t | float | float |
| fp8_e5m2_t | fp8_e5m2_t | float | float |
| hifloat8_t | hifloat8_t | float | float |

## 返回值说明

无

## 约束说明

- 源操作数A必须位于L0A（`__ca__`），B必须位于L0B（`__cb__`）；
- 目的操作数C必须位于L0C（`__cc__`）；
- 当m、n、k中的任意一个值为0时，该指令不会被执行。

## 调用示例

```cpp
using namespace AscendC::Te;


constexpr int32_t  L0_MAX = 128*128*2;
constexpr int32_t  L0C_MAX = 128*128;

size_t m = 128;
size_t n = 64;
size_t k = 64;

auto l0ALayout = MakeZzLayout<half>(m, k);
auto l0BLayout = MakeZnLayout<half>(k, n);
auto l0CLayout = MakeL0CLayout(m, n);
auto biasLayout = MakeNDCLayout<float>(1, n);

__ca__ half l0AAddr[L0_MAX];
__cb__ half l0BAddr[L0_MAX];
__cc__ float l0CAddr[L0C_MAX];
__biasbuf__ float biasAddr[L0C_MAX];

auto l0APtr = MakeL0AmemPtr(l0AAddr);
auto l0BPtr = MakeL0BmemPtr(l0BAddr);
auto l0CPtr = MakeL0CmemPtr(l0CAddr);
auto biasPtr = MakeBiasmemPtr(biasAddr);

auto l0ATensor = MakeTensor(l0APtr, l0ALayout);
auto l0BTensor = MakeTensor(l0BPtr, l0BLayout);
auto l0CTensor = MakeTensor(l0CPtr, l0CLayout);
auto biasTensor = MakeTensor(biasPtr, biasLayout);

MmadParams para;
para.m = m;                  
para.n = n;                  
para.k = k;                  
para.cmatrixInitVal = true;  


// 下面接口为3种调用方式的示例
// 调用方式1
Mad(MmadAtom<MmaTraits<MmadOperation, MmadTraitDefault>>{}, L0CTensor, L0ATensor, L0BTensor, para);
// 调用方式2
MmadAtom<MmaTraits<MmadOperation, MmadTraitDefault>>{}.Call(L0CTensor, L0ATensor, L0BTensor, para);
// 调用方式3
auto atomMad = MakeMad(MmadOperation{}, MmadTraitDefault{});
atomMad.Call(L0CTensor, L0ATensor, L0BTensor, para);

// 带Bias的调用方式
Mad(MmadAtom<MmaTraits<MmadOperation, MmadTraitDefault>>{}, L0CTensor, L0ATensor, L0BTensor, biasTensor, para);

```

完整样例请参考[TensorAPI样例代码](../../../../../examples/01_simd_cpp_api/02_features/05_tensor_api)。
