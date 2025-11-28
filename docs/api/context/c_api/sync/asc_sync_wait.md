# asc_sync_wait

## AI处理器支持情况

| AI处理器类型 | 是否支持  |
| :-----------| :------: |
| Ascend 910C |    √     |
| Ascend 910B |    √     |

## 功能说明

等待特定的同步标志，阻塞当前流水线直到条件满足。

## 函数原型

```cpp
template<typename Pipe, typename TPipe>
__aicore__ inline void asc_sync_wait(Pipe pipe, TPipe tpipe, int id)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| pipe | 输入 | 等待的源流水线类型 |
| tpipe | 输入 | 当前流水线类型 |
| id | 输入 | 同步ID |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
Pipe pipe;
TPipe tpipe;
asc_sync_wait(pipe, tpipe, 0);
```
