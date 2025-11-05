## 概述

本样例介绍了使用了带L2切分的错位分核调度器的BatchMatmul算子。
- 算子功能：基于Layout实现的带L2切分的使用错位分核策略的BatchMatmul算子。
- 使用的BlockMmad模板：MatmulMultiBlockWithLayout。
- 使用的scheduler模板：L2TailOptMisplaceCoreScheduler。
- 调用方式：<<<>>>直调。


## 支持产品型号
- Ascend910B


## 算子描述

算子规格：

<table>

<tr><td rowspan="5" align="center">算子输入</td></tr>
<tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">Layout</td></tr>
<tr><td align="center">a</td><td align="center">\</td><td align="center">float16</td><td align="center">RowMajor</td></tr>
<tr><td align="center">b</td><td align="center">\</td><td align="center">float16</td><td align="center">RowMajor</td></tr>
<tr><td align="center">bias</td><td align="center">预留参数</td><td align="center">预留参数</td><td align="center">预留参数</td></tr>
<tr></tr>

<tr><td rowspan="2" align="center">算子输出</td></tr>
<tr><td align="center">c</td><td align="center">\</td><td align="center">float16</td><td align="center">RowMajor</td></tr>

<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">MatmulOp</td></tr>
</table>


## 目录结构

```
├── 05_l2_misplace_core_batchmatmul
│   ├── CMakeLists.txt
│   ├── cmake                   # cmake文件
│   │   └── ...
│   ├── main.cpp                # 算子实现文件
│   ├── run.sh                  # 执行脚本
│   └── testcase
│       └── case.csv            # 用例
```


## 编译运行

在样例目录`examples/05_l2_misplace_core_batchmatmul`下执行run.sh脚本命令，可参考[quickstart](../../docs/01_quickstart.md#编译运行)了解执行脚本参数说明。
```
# cd examples/05_l2_misplace_core_batchmatmul
bash run.sh -r npu -v Ascend910B1 -p 0
```
结果打屏中出现`Success`，说明该条用例精度比对成功，打印结果如下所示。精度测试的执行结果将被写入`output/result_{time_stamp}.csv`。
```
----------RESULT--------------
['case_name','wrong_num', 'total_num','result','task_duration']
['case001',0,2,'Success']
```