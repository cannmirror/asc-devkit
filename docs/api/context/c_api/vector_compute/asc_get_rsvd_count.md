# asc_get_rsvd_count

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

此接口用于获取执行GatherMask操作后剩余的元素数量。

## 函数原型

```cpp
__aicore__ inline int64_t asc_get_rsvd_count()
```

## 参数说明

无

## 返回值说明

执行GatherMask操作后剩余的元素数量。

## 流水类型

PIPE_TYPE_S

## 约束说明

- 需和GatherMask操作配合使用。
- 需通过同步操作确保GatherMask执行完成后再调用本接口获取结果。

## 调用示例

```cpp
...                                       // 进行GatherMask计算
asc_sync_notify(PIPE_TYPE_V, PIPE_TYPE_S, 0);       // 设置等待和同步信号
asc_sync_wait(PIPE_TYPE_V, PIPE_TYPE_S, 0);
int64_t result = asc_get_rsvd_count();    // 获取结果
```