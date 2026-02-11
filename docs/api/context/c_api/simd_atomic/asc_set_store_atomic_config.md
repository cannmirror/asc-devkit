# asc_set_store_atomic_config

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √    |

## 功能说明

设置原子操作使能位与原子操作类型的值。

## 函数原型

```c++
__aicore__ inline void asc_set_store_atomic_config(asc_store_atomic_config& config)
```

## 参数说明

| 参数名       | 输入/输出 | 描述               |
| --------- | ----- | ---------------- |
| asc_store_atomic_config       | 输入   | 用于设置原子操作使能位和原子操作类型的值，详细说明请参考[asc_store_atomic_config](../struct/asc_store_atomic_config.md) 。|
## 返回值说明

无。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```c++
asc_store_atomic_config config;
config.atomic_type = 1; // 使能原子操作,进行原子操作的数据类型为float，值为1
config.atomic_op = 0; // 求和操作,值为0
asc_set_store_atomic_config(config); // 配置float类型原子求和操作
```
