# 快速入门

## 概述

以BasicMatmul样例为例，展示如何基于ACT快速开发实现Matmul算子，帮助开发者理解和掌握使用ACT开发算子的搭建、编译、运行等过程。


## 目录结构

进入00_basic_matmul样例目录，命令如下。
```bash
# 切换到样例目录
cd ../examples/00_basic_matmul
```

样例的目录结构如下所示。

```bash
└── 00_basic_matmul
    ├── CMakeLists.txt
    ├── cmake                   # cpu和npu场景的cmake文件
    │   └── ...
    ├── main.cpp                # 算子实现文件
    ├── run.sh                  # 执行脚本
    └── testcase
        └── case.csv            # 用例
```


## 算子实现

算子实现文件`main.cpp`主要包含定义Kernel模板调用函数、调用Kernel、定义main函数三部分。

### 定义Kernel模板调用函数

算子实现文件中需要配置的include头文件如下所示。定义Matmul计算中A、B、C矩阵的[Layout](./03_layout.md)，然后配置Kernel模板，其中的概念及配置方式详见[编程指南](./02_programming_guidelines.md)，最后定义Kernel调用函数MatmulOp。

```cpp
#include <iostream>
#include <cstdint>
#include <sstream>
#include <acl/acl.h>

#include "tiling/platform/platform_ascendc.h"
#include "include/matmul/block/block_scheduler_policy.h"
#include "include/matmul/block/block_mmad_builder.h"
#include "include/matmul/kernel/kernel_matmul.h"
#include "include/matmul/device/device_matmul.h"
#include "include/utils/host_utils.h"
#include "include/utils/layout_utils.h"
#include "include/utils/status_utils.h"
#include "../utils.h"

using namespace AscendC;
using namespace Act;
using namespace Act::Gemm;

bool isBias = false;

// Define the basic block shapes for L1 and L0
using L1TileShape = AscendC::Shape<_128, _256, _256>;
using L0TileShape = AscendC::Shape<_128, _256, _64>;

// Define matrix data type and layout
using AType = half;
using BType = half;
using CType = half;

using LayoutA = layout::RowMajor;
using LayoutB = layout::RowMajor;
using LayoutC = layout::RowMajor;

// Define scheduler type
using BlockScheduler = IterateKScheduler;

// Define BlockMmad type
using BlockMmad = Block::BlockMmadBuilder<AType, LayoutA, BType, LayoutB, CType, LayoutC, CType, LayoutC, L1TileShape,
                                          L0TileShape, BlockScheduler,
                                          MatmulMultiBlockWithLayout<>>; // dispatch policy

// Define epilogue type
using BlockEpilogue = Block::BlockEpilogueEmpty;

// Define the shape dimensions, tuple stores (m, n, k, batch)
using ProblemShape = MatmulShape;

// Define kernel type
using MatmulKernel = Kernel::KernelMatmul<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
using Arguments = typename MatmulKernel::Arguments;

// Define device matmul type
using DeviceMatmul = Device::DeviceMatmul<MatmulKernel>;

void MatmulOp(uint8_t* x1, uint8_t* x2, uint8_t* y, uint8_t* bias, int64_t m, int64_t n, int64_t k,
              void* stream = nullptr)
{
    uint8_t* workspaceDevice;
    MatmulShape shape{m, n, k, 1};
    Arguments args = {
        shape,             // problem shape
        {x1, x2, y, bias}, // mmad args
        {}                 // epilogue args
    };

    // Instantiate matmul with specfied kernel
    DeviceMatmul mm;

    // Query workspace size
    size_t workspaceSize = DeviceMatmul::GetWorkspaceSize(args);

    // Allocate workspace on device
    CHECK_ACL(aclrtMalloc((void**)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ACT_CHECK(mm.CheckArgs(args));

    // Initialize kernel with arguments and workspace pointer
    mm.InitParams(args, workspaceDevice);

    // Launch kernel
    mm();

    CHECK_ACL(aclrtFree(workspaceDevice));
}
```

### 调用Kernel
在TestMatmul函数中，根据各个输入、输出的数据量，申请计算资源；读取各输入数据后，调用Kernel，保存输出数据，最后释放申请的计算资源。

```cpp
void TestAclInit(aclrtContext& context, aclrtStream& stream, int64_t& deviceId)
{
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    CHECK_ACL(aclrtCreateStream(&stream));
}

void TestAclDeInit(aclrtContext& context, aclrtStream& stream, int64_t& deviceId)
{
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
}

void TestMatmul(int64_t m, int64_t n, int64_t k)
{
    size_t x1FileSize = m * k * sizeof(half);
    size_t x2FileSize = k * n * sizeof(half);
    size_t yFileSize = m * n * sizeof(half);
    size_t biasFileSize = 1 * n * sizeof(half);

    aclrtContext context;
    aclrtStream stream = nullptr;
    int64_t deviceId = 0;
    TestAclInit(context, stream, deviceId);

    uint8_t* x1Host;
    uint8_t* x1Device;
    CHECK_ACL(aclrtMallocHost((void**)(&x1Host), x1FileSize));
    CHECK_ACL(aclrtMalloc((void**)&x1Device, x1FileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("../input/x1_gm.bin", x1FileSize, x1Host, x1FileSize);
    CHECK_ACL(aclrtMemcpy(x1Device, x1FileSize, x1Host, x1FileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* x2Host;
    uint8_t* x2Device;
    CHECK_ACL(aclrtMallocHost((void**)(&x2Host), x2FileSize));
    CHECK_ACL(aclrtMalloc((void**)&x2Device, x2FileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("../input/x2_gm.bin", x2FileSize, x2Host, x2FileSize);
    CHECK_ACL(aclrtMemcpy(x2Device, x2FileSize, x2Host, x2FileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* biasHost = nullptr;
    uint8_t* biasDevice = nullptr;
    if (isBias) {
        CHECK_ACL(aclrtMallocHost((void**)(&biasHost), biasFileSize));
        CHECK_ACL(aclrtMalloc((void**)&biasDevice, biasFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
        ReadFile("../input/bias_gm.bin", biasFileSize, biasHost, biasFileSize);
        CHECK_ACL(aclrtMemcpy(biasDevice, biasFileSize, biasHost, biasFileSize, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    uint8_t* yHost = nullptr;
    uint8_t* yDevice = nullptr;
    CHECK_ACL(aclrtMallocHost((void**)(&yHost), yFileSize));
    CHECK_ACL(aclrtMalloc((void**)&yDevice, yFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    MatmulOp(x1Device, x2Device, yDevice, biasDevice, m, n, k, stream);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(yHost, yFileSize, yDevice, yFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("../output/output.bin", yHost, yFileSize);

    if (isBias) {
        CHECK_ACL(aclrtFree(biasDevice));
        CHECK_ACL(aclrtFreeHost(biasHost));
    }
    CHECK_ACL(aclrtFree(x1Device));
    CHECK_ACL(aclrtFreeHost(x1Host));
    CHECK_ACL(aclrtFree(x2Device));
    CHECK_ACL(aclrtFreeHost(x2Host));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFreeHost(yHost));
    TestAclDeInit(context, stream, deviceId);
}
```

### 定义`main`函数
定义入口函数，从用例文件case.csv中读取用例的shape：m, n, k；调用TestMatmul函数。
```cpp
int32_t main(int32_t argc, const char* args[])
{
    int64_t problem[3] = {1, 1, 1};               // Init problem shape [m, n, k]

    for (int32_t i = 1; i < argc && i < 4; ++i) { // Read [m, n, k, batch] from case.csv
        std::stringstream ss(args[i]);
        ss >> problem[i - 1];
    }

    TestMatmul(problem[0], problem[1], problem[2]);

    return 0;
}
```
## 用例文件

用例文件为`testcase/case.csv`，您可以按照格式：`是否执行，用例名，m，n，k`，修改或增加需要执行的用例。

```
1,case001,1,1,1
1,case002,2,2,2
1,case003,128,256,128
0,case004,1,512,128
...
```
## 编译运行

在`CMakeLists.txt`中添加`target`及对应实现文件`main.cpp`，`act_examples_add_executable`函数在examples目录下的[CMakeLists.txt](../examples/CMakeLists.txt)中定义。

```cmake
act_examples_add_executable(
    00_basic_matmul
    main.cpp
)
```
编译与运行命令已集成在`run.sh`执行脚本中，直接运行执行脚本即可进行精度或性能测试。

```
bash run.sh -r [RUN_MODE] -v [SOC_VERSION] -p [IS_PERF]
```

其中脚本参数说明如下：
- 必选参数：
  - RUN_MODE： 编译执行方式，当前仅支持NPU上板，对应取值为[npu]。
  - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行`npu-smi info`命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。
- 可选参数：
  - IS_PERF： 是否获取执行性能数据，可选择关闭和开启该功能，对应参数分别为[0 / 1]。

示例如下，Ascendxxxyy请替换为实际的AI处理器型号。
```
bash run.sh -r npu -v Ascendxxxyy -p 0
```

## 精度测试

在样例目录`examples/00_basic_matmul`下执行run.sh脚本命令，结果打屏中出现`Success`，说明该条用例精度比对成功，打印结果如下所示。精度测试的执行结果将被写入`output/result_{time_stamp}.csv`。

```bash
----------RESULT--------------
['case_name','wrong_num', 'total_num','result','task_duration']
['case001',0,1,'Success']
['case002',0,4,'Success']
['case003',0,32768,'Success']
```

## 性能测试

性能测试依赖算子调优工具msProf，首先执行`msprof op -h`，确认当前环境是否支持使用msProf，如果支持，会生成如下提示：

```bash
msopprof (Mindstudio Profiler For Operator) is part of Mindstudio Operator-dev Tools.
Used for Ascend C operator profiling by running on the board.
```

在样例目录`examples/00_basic_matmul`下执行run.sh脚本命令，同时IS_PERF参数取值为1，即可执行算子性能测试。打屏结果中出现`Success`且`task_duration`一列不为0，说明该条用例成功执行性能测试。执行性能测试时，不进行精度比对，因此`wrong_num`一列均为初始值`-1`。`task_duration`表示该条用例执行的平均性能，单位为`us`。性能测试的执行结果将被写入`output/result_{time_stamp}.csv`。

```bash
----------RESULT--------------
['case_name','wrong_num', 'total_num','result','task_duration']
['case001',-1,1,'Success','9.8']
['case002',-1,4,'Success','15.6']
['case003',-1,32768,'Success,'284.23']
```

## 性能分析

性能测试执行成功后，会在样例目录下生成`profiling`文件：

```bash
└── OPPROF_{timestamp}_XXX
    ├── dump
    ├── ArithmeticUtilization.csv
    ├── L2Cache.csv
    ├── Memory.csv
    ├── MemoryL0.csv
    ├── MemoryUB.csv
    ├── OpBasicInfo.csv
    ├── PipeUtilization.csv
    ├── ResourceConflictRatio.csv
    └── visualize_data.bin
```

各文件说明如下：

<table>
    <tr>
        <th>名称</th>
        <th>说明</th>
    </tr>
    <tr>
        <td>dump文件夹</td>
        <td>原始的性能数据，用户无需关注。</td>
    </tr>
    <tr>
        <td>ArithmeticUtilization.csv</td>
        <td>cube和vector类型的指令耗时和占比，可参考<a href="https://hiascend.com/document/redirect/CannCommunityToolMsProf">算子调优（msProf）</a>中的“性能数据文件 > msprof op > ArithmeticUtilization（cube及vector类型指令耗时和占比）”章节。</td>
    </tr>
    <tr>
        <td>L2Cache.csv</td>
        <td>L2 Cache命中率，可参考<a href="https://hiascend.com/document/redirect/CannCommunityToolMsProf">算子调优（msProf）</a>中的“性能数据文件 > msprof op > L2Cache（L2 Cache命中率）”章节。</td>
    </tr>
    <tr>
        <td>Memory.csv</td>
        <td>UB/L1/L2/主存储器获取内存读写带宽速率，可参考<a href="https://hiascend.com/document/redirect/CannCommunityToolMsProf">算子调优（msProf）</a>中的“性能数据文件 > msprof op > Memory（内存读写带宽速率）”章节。</td>
    </tr>
    <tr>
        <td>MemoryL0.csv</td>
        <td>L0A/L0B/L0C获取内存读写带宽速率，可参考<a href="https://hiascend.com/document/redirect/CannCommunityToolMsProf">算子调优（msProf）</a>中的“性能数据文件 > msprof op > MemoryL0（L0读写带宽速率）”章节。</td>
    </tr>
    <tr>
        <td>MemoryUB.csv</td>
        <td>mte/vector/scalar获取ub读写带宽速率，可参考<a href="https://hiascend.com/document/redirect/CannCommunityToolMsProf">算子调优（msProf）</a>中的“性能数据文件 > msprof op > MemoryUB（UB读写带宽速率）”章节。</td>
    </tr>
    <tr>
        <td>PipeUtilization.csv</td>
        <td>获取计算单元和搬运单元耗时和占比，可参考<a href="https://hiascend.com/document/redirect/CannCommunityToolMsProf">算子调优（msProf）</a>中的“性能数据文件 > msprof op > PipeUtilization（计算单元和搬运单元耗时占比）”章节。</td>
    </tr>
    <tr>
        <td>ResourceConflictRatio.csv</td>
        <td>UB上的bank group、bank conflict和资源冲突在所有指令中的占比，可参考<a href="https://hiascend.com/document/redirect/CannCommunityToolMsProf">算子调优（msProf）</a>中的“性能数据文件 > msprof op > ResourceConflictRatio（资源冲突占比）”章节。</td>
    </tr>
    <tr>
        <td>OpBasicInfo.csv</td>
        <td>算子基础信息，包含算子名称、block dim和耗时等信息，可参考<a href="https://hiascend.com/document/redirect/CannCommunityToolMsProf">算子调优（msProf）</a>中的“性能数据文件 > msprof op > OpBasicInfo（算子基础信息）”章节。</td>
    </tr>
    <tr>
        <td>visualize_data.bin</td>
        <td>算子基础信息和计算单元负载等信息的可视化呈现文件，可参考<a href="https://hiascend.com/document/redirect/CannCommunityToolMsProf">算子调优（msProf）</a>中的“计算内存热力图”章节。</td>
    </tr>
</table>
