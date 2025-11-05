## 概述

本样例介绍了对基础Matmul结果进行量化计算的算子。
- 算子功能：基于Layout实现的基础K轴遍历Matmul算子，并将int8_t的结果量化为float16类型。
- 使用的BlockMmad模板：MatmulMultiBlockWithLayout。
- 使用的scheduler模板：IterateKScheduler。
- 使用的BlockEpilogue模板：BlockEpilogueQuant。
- 使用的kernel模板：KernelMatmulMixWorkspace。
- 调用方式：<<<>>>直调。


## 支持产品型号
- Ascend910B


## 算子描述

算子规格：

<table>

<tr><td rowspan="7" align="center">算子输入</td></tr>
<tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">Layout</td></tr>
<tr><td align="center">a</td><td align="center">\</td><td align="center">int8_t</td><td align="center">RowMajor</td></tr>
<tr><td align="center">b</td><td align="center">\</td><td align="center">int8_t</td><td align="center">ColumnMajor</td></tr>
<tr><td align="center">scale</td><td align="center">\</td><td align="center">int8_t</td><td align="center">RowMajor</td></tr>
<tr><td align="center">perTokenScale</td><td align="center">\</td><td align="center">int8_t</td><td align="center">RowMajor</td></tr>
<tr><td align="center">bias</td><td align="center">预留参数</td><td align="center">预留参数</td><td align="center">预留参数</td></tr>
<tr></tr>

<tr><td rowspan="2" align="center">算子输出</td></tr>
<tr><td align="center">out</td><td align="center">\</td><td align="center">float16</td><td align="center">RowMajor</td></tr>

<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">MatmulOp</td></tr>
</table>


## 目录结构

```
├── 03_quant_matmul
│   ├── CMakeLists.txt
│   ├── cmake                   # cmake文件
│   │   └── ...
│   ├── main.cpp                # 算子实现文件
│   ├── run.sh                  # 执行脚本
│   └── testcase
│       └── case.csv            # 用例
```


## 编译运行

在样例目录`examples/03_quant_matmul`下执行run.sh脚本命令，可参考[quickstart](../../docs/01_quickstart.md#编译运行)了解执行脚本参数说明。本样例中的量化操作在Vector Core上进行epilogue操作，run.sh脚本会将[`device_matmul.h`](../../include/matmul/device/device_matmul.h)里的kernel type由KERNEL_TYPE_AIC_ONLY改成KERNEL_TYPE_MIX_AIC_1_2，在样例执行结束后，run.sh脚本将kernel type改回原值。
```
# cd examples/03_quant_matmul
bash run.sh -r npu -v Ascend910B1 -p 0
```
结果打屏中出现`Success`，说明该条用例精度比对成功，打印结果如下所示。精度测试的执行结果将被写入`output/result_{time_stamp}.csv`。
```
----------RESULT--------------
['case_name','wrong_num', 'total_num','result','task_duration']
['case001',0,1,'Success']
```