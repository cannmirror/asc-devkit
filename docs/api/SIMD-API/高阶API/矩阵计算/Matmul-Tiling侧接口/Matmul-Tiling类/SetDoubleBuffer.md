# SetDoubleBuffer

## 功能说明

设置A/B/C/Bias是否开启double buffer功能，以及是否需要做ND2NZ或者NZ2ND的转换，主要用于Tiling函数内部调优。

**该接口为预留接口，当前版本暂不支持。**

## 函数原型

```
int32_t SetDoubleBuffer(bool a, bool b, bool c, bool bias, bool transND2NZ = true, bool transNZ2ND = true)
```

## 参数说明

**表 1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| a | 输入 | 设置A矩阵是否开启double buffer。 |
| b | 输入 | 设置B矩阵是否开启double buffer。 |
| c | 输入 | 设置C矩阵是否开启double buffer。 |
| bias | 输入 | 设置Bias矩阵是否开启double buffer。 |
| transND2NZ | 输入 | 设置是否需要[ND2NZ](../../../../基础API/Memory数据搬运/DataCopy/随路转换ND2NZ搬运.md#fig128961542184620)。 |
| transNZ2ND | 输入 | 设置是否需要[NZ2ND](../../../../基础API/Memory数据搬运/DataCopy/随路转换NZ2ND搬运.md#fig15851251122815)。 |

## 返回值说明

-1表示设置失败； 0表示设置成功。

## 约束说明

无
