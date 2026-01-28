# asc_set_l0c2gm_quant
## AI处理器支持情况

|AI处理器类型   | 是否支持 |
| ------------|:----:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

数据搬运过程中进行随路量化时，通过调用该接口设置量化流程中的矢量量化参数。

## 函数原型

```c++
__aicore__ inline void asc_set_l0c2gm_quant(uint64_t quant)
```

## 参数说明

|参数名|输入/输出|描述|
|------------|------------|-----------|
| quant | 输入     | 量化操作前张量的矢量起始地址。|
## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

支持以下三种传参形式：
- 同时设置reluPre和quantPre。
- 仅传入reluPrequantPre传入0。
- 仅传入quantPre，reluPre传入0。

## 调用示例

```c++
constexpr uint64_t reluPre = 0;
constexpr uint64_t quantPre = 0x1000;// 假设量化操作有效地址为 0x1000
asc_set_l0c_copy_config(reluPre, quantPre);
```