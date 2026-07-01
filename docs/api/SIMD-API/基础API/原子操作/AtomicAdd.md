# AtomicAdd<a name="ZH-CN_TOPIC_0000002598784427"></a>

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- Atlas 200I/500 A2 推理产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas 推理系列产品AI Core：不支持
<!-- end id5 -->
<!-- npu="310p" id6 -->
- Atlas 推理系列产品Vector Core：不支持
<!-- end id6 -->
<!-- npu="910" id7 -->
- Atlas 训练系列产品：不支持
<!-- end id7 -->

## 功能说明<a name="section163431416121118"></a>

头文件路径为：`"basic_api/kernel_operator_atomic_intf.h"`。

该接口用于在指定GM地址上进行原子加操作，将**address**指向的GM地址上的旧值（**old_value**）与输入标量值（**value**）求和，将和结果（**new_value**）写回GM地址，返回该地址修改前的值（**old_value**）。

计算公式如下：

$$new\_value = Add(old\_value, value)$$

## 函数原型<a name="section1171112415119"></a>

```cpp
template <typename T>
__aicore__ inline T AtomicAdd(__gm__ T *address, T value)
```

## 参数说明<a name="section086962912112"></a>

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。 |

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| address | 输入 | 输入GM的地址。 |
| value | 输入 | 标量值，支持数据类型和address指向的数据类型一致。 |

## 数据类型<a name="section832119018128"></a>

address、value、返回值的数据类型相同，支持的数据类型为int32_t、uint32_t、float、int64_t、uint64_t。

## 返回值说明<a name="section97313820116"></a>

GM地址上做原子操作前的数据。

## 约束说明<a name="section24727811123"></a>

- 在开启编译器自动同步功能（参见[自动同步使用约束说明](../同步控制/核内同步/关键特性说明.md#自动同步使用约束说明)）的前提下，编译器能够自动在PIPE_MTE2/PIPE_MTE3与PIPE_S之间插入同步（参见[编译器自动同步](../同步控制/核内同步/关键特性说明.md#tpipe-tque框架编程范式和编译器自动同步支持情况)）。但是，AtomicAdd为标量计算，在读写GM时如果与搬运单元（MTE2/MTE3）存在数据依赖，编译器却无法自动插入同步，开发者需要根据实际情况手动插入同步。
- Scalar原子操作会绕过DCache，需要调用[DataCacheCleanAndInvalid](../缓存控制/DataCacheCleanAndInvalid.md)接口确保GM与DCache的一致性。

## 调用示例<a name="section191505489122"></a>

```cpp
extern "C" __global__ __aicore__ void atomic_histogram_kernel(__gm__ uint32_t* hist, __gm__ uint8_t* input, uint32_t dataSize)
{
    // 每个核遍历输入数据，统计各灰度值出现的频次。
    for (uint32_t i = blockIdx.x; i < dataSize; i += blockDim.x) {
        // 使用AtomicAdd将对应bin的计数原子地加1。
        AscendC::AtomicAdd(&hist[input[i]], 1);
    }
}
```

假设上述函数在多个核上执行，每个核处理各自分片数据，通过AtomicAdd原子地累加到直方图数组hist中，最终得到全局直方图统计结果。
