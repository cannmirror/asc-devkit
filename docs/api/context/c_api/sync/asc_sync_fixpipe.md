# asc_sync_fixpipe

## AI处理器支持情况

| AI处理器类型 | 是否支持  |
| :-----------| :------: |
| Ascend 910C |    √     |
| Ascend 910B |    √     |

## 功能说明

针对Fixpipe流水线执行同步操作。

## 函数原型

```cpp
__aicore__ inline void asc_sync_fixpipe(int id)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| id | 输入 | 同步ID |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
asc_sync_fixpipe(0);
```
