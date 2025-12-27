# asc_sync

## 产品支持情况

| 产品 | 是否支持  |
| :-----------| :------: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

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
