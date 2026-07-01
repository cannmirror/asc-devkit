# AtomicCas<a name="ZH-CN_TOPIC_0000002568264812"></a>

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

调用该接口后，可在指定GM地址上进行原子比较操作，读取**address**指向的GM地址上的旧值（**old_value**）与输入标量值**value1**进行比较：

- 如果相等，则将输入标量值**value2**写入GM地址；
- 如果不相等，则GM地址上的值保持不变。
返回该地址修改前的值（**old_value**）。

计算公式如下：

$$new\_value = (old\_value == value1\ ?\ value2\ :\ old\_value)$$

## 函数原型<a name="section1171112415119"></a>

```cpp
template <typename T>
__aicore__ inline T AtomicCas(__gm__ T *address, T value1, T value2)
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
| value1/value2 | 输入 | 标量值，支持数据类型和address指向的数据类型保持一致。 |

## 数据类型<a name="section832119018128"></a>

address、value1/value2、返回值的数据类型相同，支持的数据类型为uint32_t、uint64_t。

## 返回值说明<a name="section97313820116"></a>

GM地址上做原子操作前的数据。

## 约束说明<a name="section24727811123"></a>

- 在开启编译器自动同步功能（参见[自动同步使用约束说明](../同步控制/核内同步/关键特性说明.md#自动同步使用约束说明)）的前提下，编译器能够自动在PIPE_MTE2/PIPE_MTE3与PIPE_S之间插入同步（参见[编译器自动同步](../同步控制/核内同步/关键特性说明.md#tpipe-tque框架编程范式和编译器自动同步支持情况)）。但是，AtomicCas为标量计算，在读写GM时如果与搬运单元（MTE2/MTE3）存在数据依赖，编译器却无法自动插入同步，开发者需要根据实际情况手动插入同步。
- Scalar原子操作会绕过DCache，需要调用[DataCacheCleanAndInvalid](../缓存控制/DataCacheCleanAndInvalid.md)接口确保GM与DCache的一致性。

## 调用示例<a name="section191505489122"></a>

```cpp
__gm__ uint32_t lock = 0;  // 0 表示未锁定，1 表示已锁定。

extern "C" __global__ __aicore__ void atomic_lock_acquire_kernel(__gm__ int32_t* shared_data)
{
    // 使用AtomicCas尝试获取锁：当lock为0时将其原子地置为1。
    while (AscendC::AtomicCas(&lock, 0, 1) != 0) {
        // 自旋等待，直到成功获取锁。
    }
    // 获取锁后，在临界区中安全地操作共享数据。
    shared_data[0] = shared_data[0] + 1;
    // 注意：解锁需要配合AtomicExch将lock置为0。
}
```

在上述示例中，多个核并发执行atomic_lock_acquire_kernel，每次只有一个核能成功将lock从0变为1并进入临界区，其他核自旋等待，从而保证shared_data的互斥访问。在实现自旋锁时，AtomicCas一般与AtomicExch配对使用，以确保锁的正确释放，释放锁的示例请参考[AtomicExch调用示例](AtomicExch.md#调用示例)。
