# asc_set_mask_count

## AI处理器支持情况

|AI处理器类型|是否支持|
| :------------ | :------------: |
| <term>Ascend 910C</term> | √ |
| <term>Ascend 910B</term> | √ |

## 功能说明

设置mask模式为Counter模式。该模式下，不需要开发者去感知迭代次数、处理非对齐的尾块等操作，可直接传入数据计算数据量，实际迭代次数由Vector计算单元自动推断。

## 函数原型

```cpp
__aicore__ inline void asc_set_mask_count()
```

## 参数说明

无

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

设置为Counter模式的场景需要在矢量计算和[asc_set_vector_mask](asc_set_vector_mask.md)配合使用。

## 调用示例

```cpp
asc_set_mask_count();
asc_set_vector_mask(0, static_cast<uint64_t>(64)); // 设置前64个元素参与计算
... // 计算操作
```