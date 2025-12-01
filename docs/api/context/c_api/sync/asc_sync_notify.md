# asc_sync_notify

## AI处理器支持情况

| AI处理器类型 | 是否支持  |
| :-----------| :------: |
| Ascend 910C |    √     |
| Ascend 910B |    √     |

## 功能说明

设置同步标志，通知目标流水线。

## 函数原型

```cpp
template<typename Pipe, typename TPipe>
__aicore__ inline void asc_sync_notify(Pipe pipe, TPipe tpipe, int id)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| pipe | 输入 | 源流水线类型。 |
| tpipe | 输入 | 目标流水线类型。 |
| id | 输入 | 同步ID。 |

## 返回值说明

无

## 流水类型

PIPE_TYPE_S

## 约束说明

无

## 调用示例

```cpp
Pipe pipe;
TPipe tpipe;
asc_sync_notify(pipe, tpipe, 0);
```
