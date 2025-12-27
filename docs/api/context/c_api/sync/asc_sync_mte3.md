# asc_sync_mte3

## 产品支持情况

| 产品 | 是否支持  |
| :-----------| :------: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

针对PIPE_TYPE_MTE3流水线执行同步操作。

## 函数原型

```cpp
__aicore__ inline void asc_sync_mte3(int id)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| id | 输入 | 同步ID。 |

## 返回值说明

无

## 流水类型

PIPE_TYPE_S

## 约束说明

无

## 调用示例

```cpp
asc_sync_mte3(0);
```
