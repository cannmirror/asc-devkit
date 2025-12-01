# asc_sync

## AI处理器支持情况

| AI处理器类型 | 是否支持  |
| :-----------| :------: |
| Ascend 910C |    √     |
| Ascend 910B |    √     |

## 功能说明

等待指定流水线或所有流水线操作完成。

## 函数原型

```cpp
template<typename Pipe>
__aicore__ inline void asc_sync(Pipe pipe);
__aicore__ inline void asc_sync()
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| pipe | 输入 | 指定需要同步的流水线（可选，无参数时默认为所有）。 |

## 返回值说明

无

## 流水类型

PIPE_TYPE_S

## 约束说明

无

## 调用示例

```cpp
asc_sync();
```
