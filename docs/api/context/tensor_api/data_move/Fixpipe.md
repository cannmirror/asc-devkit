# Fixpipe

## 产品支持情况

| 产品                                        | 是否支持 |
| ------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                   |    √     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    x     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    x     |
| Atlas 200I/500 A2 推理产品                  |    x     |
| Atlas 推理系列产品AI Core                   |    x     |
| Atlas 推理系列产品Vector Core               |    x     |
| Atlas 训练系列产品                          |    x     |
| Atlas 200/300/500 推理产品                  |    x     |


## 功能说明

Fixpipe能够在矩阵计算完成后对计算结果进行处理，并通过PIPE_FIX流水完成L0C到GM/UB的数据搬运，支持以下功能：
1. 数据搬运：将矩阵计算结果从L0C搬运到Global Memory/Unified Buffer
2. 随路格式转换：支持NZ到ND/DN的格式转换
3. 随路量化：支持float/int32_t到int8_t/uint8_t/half等类型的量化
4. 随路ReLU激活：对矩阵计算结果执行普通ReLU操作
5. 通道拆分：输出到GM、数据类型为float到float，输出格式为NZ时，支持通道拆分
6. 双目标模式：输出到UB时，支持启用双目标模式控制

## 函数原型

```cpp
// 通路L0C->GM/UB, 不使能量化功能
template <const FixpipeTrait& trait = DEFAULT_FIXPIPE_TRAIT, typename T, typename U>
__aicore__ inline typename Std::enable_if<VerifyingFixpipeTemplate<T, U>, void>::type Fixpipe(const T& dst, const U& src, const FixpipeParams& params = FixpipeParams{})

// 通路L0C->GM/UB, 使能量化功能
template <const FixpipeTrait& trait = DEFAULT_FIXPIPE_TRAIT, typename T, typename U, typename S>
__aicore__ inline typename Std::enable_if<VerifyingFixpipeQuantTemplate<T, U, S>, void>::type 
Fixpipe(const T& dst, const U& src, const S& quant, const FixpipeParams& params = FixpipeParams{});

// 通路L0C->GM/UB, 不使能量化功能，将src搬运到dst指定坐标coord右下角区域
template <const FixpipeTrait& trait = DEFAULT_FIXPIPE_TRAIT, typename T, typename U, typename Coord>
__aicore__ inline typename Std::enable_if<VerifyingFixpipeTemplateWithCoord<T, U, Coord>, void>::type
Fixpipe(const T& dst, const U& src, const Coord& coord, const FixpipeParams& params = FixpipeParams{});

// 通路L0C->GM/UB, 使能量化功能，将src搬运到dst指定坐标coord右下角区域
template <const FixpipeTrait& trait = DEFAULT_FIXPIPE_TRAIT, typename T, typename U, typename S, typename Coord>
__aicore__ inline typename Std::enable_if<VerifyingFixpipeQuantTemplateWithCoord<T, U, S, Coord>, void>::type
Fixpipe(const T& dst, const U& src, const S& quant, const Coord& coord, const FixpipeParams& params = FixpipeParams{});
```

## 参数说明

**表 1**  模板参数说明

| 参数名            | 描述                                                         |
| ----------------- | ------------------------------------------------------------ |
| T                 | 目的操作数的数据类型，通过MakeTensor构造的[GlobalTensor/LocalTensor类型](../../tensor/Tensor.md)，存储位置支持GM和UB，数据格式支持ND、DN和NZ。 |
| U                 | 源操作数的数据类型，通过MakeTensor构造的LocalTensor类型，存储位置仅支持L0C，数据格式仅支持NZ。  |
| S                 | 量化参数的数据类型，可以是Scalar或者是通过MakeTensor构造的LocalTensor类型（存储位置仅支持L1，数据格式仅支持ND）。量化系数数据类型为uint64_t。 |
| Coord             | 坐标偏移的数据类型，通过MakeCoord构造的[Coord类型](../../tensor/Coord.md)。 |
| FixpipeTrait | Fixpipe相关配置参数，类型为FixpipeTrait：<br/>struct FixpipeTrait {<br/>    RoundMode roundMode = RoundMode::DEFAULT;<br/>    bool enableRelu = false;<br/>    bool enableChannelSplit = false;<br/>    DualDstMode dualDstCtl = DualDstMode::DUAL_DST_DISABLE;<br/>};<br/>constexpr FixpipeTrait DEFAULT_FIXPIPE_TRAIT; |


**表 2**  FixpipeTrait参数说明

| 参数名   | 描述                                                         |
| -------- | :----------------------------------------------------------- |
| roundMode | RoundMode是一个枚举类型，用于控制量化场景下的舍入模式，仅针对float量化为hifloat8_t场景生效，默认值为RoundMode::DEFAULT。若需要设置为混合舍入模式（对应的量化模式为QuantMode_t::VQF322HIF8_PRE_HYBRID/QuantMode_t::QF322HIF8_PRE_HYBRID），设置为RoundMode::HYBRID。|
| enableRelu | 是否使能ReLU的开关， false：不使能ReLU功能，true：使能ReLU功能。|
| enableChannelSplit | 输出到GM场景下，是否使能通道拆分的功能。仅在目的操作数和源操作数都为float，且目的操作数数据格式为NZ时才能使能通道拆分。 |
| dualDstCtl | DualDstMode是一个枚举类型，用于输出到UB场景下双目标模式控制，默认值为DualDstMode::DUAL_DST_DISABLE（即不启用双目标模式控制）。当启用双目标模式控制时，L0C中的M×N矩阵将被分成两半，并同时写入两个子块（SUB BLOCK）的UB中，其中前半部分写入SUB BLOCK0，后半部分写入SUB BLOCK1。支持设置DualDstMode::DUAL_DST_SPLIT_M从M维度分成两半，要求M是2的倍数，设置DualDstMode::DUAL_DST_SPLIT_N从N维度分成两半，要求N是32的倍数。仅支持在普通搬运模式或者NZ2ND搬运场景下使用，不支持随路功能场景。 |


**表 3**  参数说明

| 参数名 | 输入/输出 | 含义         |
| ------ | --------- | ------------ |
| dst    | 输出      | 目的操作数。 |
| src    | 输入      | 源操作数。   |
| quant  | 输入      | 量化参数。     |
| coord  | 输入      | 目的操作数上的偏移坐标。 |
| params | 输入      | 运行时参数结构体，类型为FixpipeParams：<br/>struct FixpipeParams {<br/> uint8_t unitFlag = 0;<br/>} |


**表 4**  FixpipeParams参数说明

| 参数名 | 描述         |
| ------ |------------ |
| unitFlag  | unitFlag机制为Mmad指令和Fixpipe指令提供细粒度的并行能力，使能该功能后，硬件每计算完一个分形，计算结果就会被搬出，该功能不适合用于在L0C Buffer累加的场景。默认值为0，使能该功能时，unitFlag设置为3即可。 |


## 数据类型支持

**表 5**  L0C->GM/UB数据类型和量化类型支持

| 源操作数类型 | 目的操作数类型 | 是否支持 | 支持的量化类型 |
|------------|--------------|---------| --------- |
| float | int8_t | √ | Scalar量化、Vector量化 |
| float | uint8_t | √ | Scalar量化、Vector量化 |
| float | hifloat8_t | √ | Scalar量化、Vector量化 |
| float | fp8_e4m3fn_t | √ | Scalar量化、Vector量化 |
| float | half | √ | 直接类型转换、Scalar量化、Vector量化 |
| float | bfloat16_t | √ | 直接类型转换、Scalar量化、Vector量化 |
| float | float | √ | 不量化、Scalar量化、Vector量化 |
| int32_t | int8_t | √ | Scalar量化、Vector量化 |
| int32_t | uint8_t | √ | Scalar量化、Vector量化 |
| int32_t | half | √ | Scalar量化、Vector量化 |
| int32_t | bfloat16_t | √ | Scalar量化、Vector量化 |
| int32_t | int32_t | √ | 不量化 |


## 返回值说明

VerifyingFixpipeTemplate/VerifyingFixpipeQuantTemplate/VerifyingFixpipeTemplateWithCoord/VerifyingFixpipeQuantTemplateWithCoord表达式为模板参数校验表达式，执行效果如下：

-   校验表达式成立，表明输入参数符合Fixpipe接口入参数据类型约束，Fixpipe函数表达式返回值为void。
-   校验表达式不成立，表明输入参数不符合Fixpipe接口入参数据类型约束。上述四个模板参数表达式均不通过时，Fixpipe函数声明失效，继而会导致编译错误。

## 理论性能

当前版本暂不支持理论性能计算。

## 约束说明

-   地址重叠约束：NA。
-   地址对齐约束：源操作数起始地址要求64字节对齐。目的操作数为NZ格式时，其地址要求满足32字节对齐；目的操作数为ND/DN格式且输出到UB上时，其地址和长度要求32字节对齐。
-   参数组合约束：目的操作数位于GM上，其数据格式为NZ，且其数据类型和源操作数都为float时才能使能通道拆分，若不开启通道拆分，需要保证目的操作数的分形与L0C一致。
-   环境影响约束：接口调用不影响全局状态，无需复原。
-   特殊数据类型约束：使用float->hifloat8_t特殊类型时，需注意FixpipeTrait量化舍入参数roundMode配置。
-   性能约束：无。
-   异常和边界值处理：输入为inf或nan时，输出量化结果为inf或nan；输入为0时，输出为0；输入超过MAX/MIN范围时，输出为MAX/MIN。
-   Tensor Layout相关约束：
    -   Shape、Stride只支持四维，针对不同的物理存储位置，四个维度的配置均有不同的约束，部分维度为固定值，不可配置。详见[层次化表达法](../../tensor/Layout和层次化表述法.md)。
    -   Shape、Stride具体维度的数据，仅支持基础整数类型和Std::Int类型。
    -   Fixpipe Vector量化场景要求quant参数数据格式是ND，存储位置是L1，地址要求32字节对齐。

## 流水类型

PIPE_FIX

## 特性说明

### 量化特性
Fixpipe支持对矩阵计算结果进行量化操作，可将float或int32_t类型量化为int8_t、uint8_t、hifloat8_t、fp8_e4m3fn_t等低精度类型，支持Scalar量化和Vector量化。

### ReLU特性
通过FixpipeTrait中的enableRelu参数，可以控制是否随路对矩阵计算结果执行ReLU操作。

### 通道拆分特性
当源操作数和目的操作数都为float，目的操作数位于GM上且为NZ格式时，可以通过FixpipeTrait中的enableChannelSplit参数，控制是否使能通道拆分功能。

### 双目标模式
当目的操作数位于UB上，可以通过FixpipeTrait中的dualDstCtl参数，开启使能双目标模式。

## 调用示例

```cpp
...
// L0C -> GM

size_t mLength = 128;
size_t nLength = 64;

// 其中DstT为输出的数据类型，支持int8_t、uint8_t、hifloat8_t、fp8_e4m3fn_t、half、bfloat16_t、int32_t、float
auto dstLayout = AscendC::MakeLayout(MakeShape(mLength, nLength), MakeStride(nLength, Std::Int<1>{}));
auto dstPtr = MakeGMmemPtr(dst);  // dst为给定的gm地址
auto globalC = MakeTensor(dstPtr, dstLayout);

// 其中SrcT为L0C上的数据类型，支持int32_t/float
auto l0CLayout = AscendC::MakeL0CLayout(mLength, nLength);
constexpr int32_t  L0C_MAX = 128*128;
__cc__ SrcT l0CAddr[L0C_MAX];
auto l0CPtr = MakeL0CmemPtr(l0CAddr);
auto l0CTensor = MakeTensor(l0CPtr, l0CLayout);
// 下面接口为3种调用方式的示例，L0C->UB通路类似，需将globalC替换为对应的UB上的Tensor，CopyL0C2GM改为CopyL0C2UB
// 调用方式1
Copy(CopyAtom<CopyTraits<CopyL0C2GM, FixpipeTraitDefault>>{}, globalC, l0CTensor);
// 调用方式2
CopyAtom<CopyTraits<CopyL0C2GM, FixpipeTraitDefault>>{}.Call(globalC, l0CTensor);
// 调用方式3
auto atomCopyL0C2GM = MakeCopy(CopyL0C2GM{}, FixpipeTraitDefault{});
atomCopyL0C2GM.Call(globalC, l0CTensor);

// 带坐标场景传参调用示例, 其中row和col为期望搬运到目标输出GM上的偏移坐标
Copy(CopyAtom<CopyTraits<CopyL0C2GM, FixpipeTraitDefault>>{}, globalC, l0CTensor, MakeCoord(row, col));

// 运行时参数调用示例，其中FixpipeParams中unitFlag值传参为3
Copy(CopyAtom<CopyTraits<CopyL0C2GM, FixpipeTraitDefault>>{}, globalC, l0CTensor, MakeCoord(row, col), AscendC::Te::FixpipeParams{3});

// Scalar量化场景调用示例
float originQuantScalar = 0.5;
uint64_t quantScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&originQuantScalar));
Copy(CopyAtom<CopyTraits<CopyL0C2GM, FixpipeTraitDefault>>{}, globalC, l0CTensor, quantScalar, MakeCoord(row, col), AscendC::Te::FixpipeParams{3});

// Vector量化场景调用示例
// 构造L1上量化Tensor对象
__cbuf__ uint64_t l1QuantAddr[128];
auto l1QuantBuff = l1QuantPtr + 0;
auto l1QuantLayout = AscendC::MakeLayout(MakeShape(AscendC::Std::Int<1>{}, nLength), MakeStride(Std::Int<1>{}, Std::Int<1>{}));
auto quantTensor = MakeTensor(l1QuantBuff, l1QuantLayout);
// 使用CopyGM2L1方法将GM上的量化Tensor搬运到L1上，其中globalQuant为GM上的量化Tensor
Copy(CopyAtom<CopyTraits<CopyGM2L1, DataCopyTraitDefault>>{}, quantTensor, globalQuant);
// 调用CopyL0C2GM时将L1上的量化Tensor作为参数传递
Copy(CopyAtom<CopyTraits<CopyL0C2GM, FixpipeTraitDefault>>{}, globalC, l0CTensor, quantTensor, MakeCoord(row, col), AscendC::Te::FixpipeParams{3});

// 用户自定义FixpipeTrait参数
constexpr FixpipeTrait fixpipeTrait = {RoundMode::DEFAULT, true, false, SINGLE_DST_MODE};

struct FixpipeTraitCustom {
    using TraitType = FixpipeTrait;
    static constexpr const TraitType value = fixpipeTrait;
};

Copy(CopyAtom<CopyTraits<CopyL0C2GM, FixpipeTraitCustom>>{}, globalC, l0CTensor);
...
```
完整样例请参考[TensorAPI样例代码](../../../../../../examples/02_features/05_tensor_api)。
