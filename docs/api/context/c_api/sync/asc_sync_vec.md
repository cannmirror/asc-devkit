# asc_sync_vec

## AI处理器支持情况

| AI处理器类型 | 是否支持  |
| :-----------| :------: |
| Ascend 910C |    √     |
| Ascend 910B |    √     |

## 功能说明

同步所有流水线，等同于全屏障。

## 函数原型

```cpp
__aicore__ inline void asc_sync_vec()
```

## 参数说明

无

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
asc_sync_vec();
```
