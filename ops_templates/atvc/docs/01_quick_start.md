# 快速入门
这篇文档帮助你体验ATVC开发Add算子的整个流程，帮助开发者快速熟悉使用ATVC模板库开发算子的基本步骤。完整样例请参考[add examples](../examples/add/add.cpp)。<br>

## 环境准备
- 硬件型号支持  
快速入门样例仅支持Atlas A2训练系列产品、Atlas 800I A2推理产品、A200I A2 Box 异构组件。
- 配套软件  
参考[ascendc-api-adv仓通用环境准备章节](../../../README.md)，完成源码下载和CANN软件包及相关依赖的安装。
- 下载代码
```bash
git clone https://gitee.com/ascend/ascendc-api-adv.git
```

## 算子实现
本示例将展示如何基于ATVC提供的模板以及接口快速搭建Add算子，示例内展示了ATVC框架下区别于传统Ascend C Add的实现代码。<br>

### 定义算子描述
首先通过ATVC提供的[ATVC::OpTraits](../include/common/atvc_opdef.h)模板结构体来描述Add算子的输入输出信息，定义如下：
```cpp
// Add算子中有两个输入，一个输出。类型均为float
using ADD_OPTRAITS = ATVC::OpTraits<ATVC::OpInputs<float, float>, ATVC::OpOutputs<float>>;
```

### 实现算子计算逻辑
用户需要通过Ascend C API来搭建Add算子的核心计算逻辑，在ATVC框架中，这类算子的核心计算逻辑是通过定义一个结构体的仿函数来实现。它需要`ATVC::OpTraits`作为固定模板参数，并重载`operator()`来被提供的Kernel层算子模板类调用。<br> 
Add算子的计算仿函数定义如下：
```cpp
#include "atvc.h" // 包含所有atvc模板api的总入口头文件

// 传入编译态参数ATVC::OpTraits
template<typename Traits>
struct AddComputeFunc {
    /*
    函数说明： c = a + b
    参数说明：
        a                   : 参与运算的输入
        b                   : 参与运算的输入
        c                   : 参与运算的输出
    */
    template<typename T> 
    // 重载operator，提供给算子模板类调用
    __aicore__ inline void operator()(AscendC::LocalTensor<T> a, AscendC::LocalTensor<T> b, AscendC::LocalTensor<T> c) {
        AscendC::Add(c, a, b, c.GetSize()); // 开发调用AscendC API自行实现计算逻辑, 通过c.GetSize()获取单次计算的元素数量
    }
};
```


## 实现核函数
ATVC提供的[ATVC::Kernel::EleWiseOpTemplate](../include/elewise/elewise_op_template.h)算子模板类实现了核内的数据搬运、资源申请和计算调度功能。它将计算仿函数作为模板参数传入来完成构造实例化，用户可通过调用`ATVC::Kernel::EleWiseOpTemplate`算子模板类的`Run(Args&&... args)`接口完成算子的功能计算，完成完整核函数的实现。
在`examples/add`用例中，算子核函数的形式参数除了输入输出之外，还需额外传入`GM_ADDR param`的形参。该参数包含算子模板类进行数据搬运数据的必要参数，由`ATVC::Host::CalcEleWiseTiling` API计算得出。
<br>
完整的`AddCustom`核函数定义如下：

```cpp
template<class Traits>
/*
 * 该函数为Add算子核函数入口
 * a        Device上的gm地址，指向Add算子第一个输入
 * b        Device上的gm地址，指向Add算子第二个输入
 * c        Device上的gm地址，指向Add算子第一个输出
 * param    指向运行态ATVC::EleWiseParam数据
*/
__global__ __aicore__ void AddCustom(GM_ADDR a, GM_ADDR b, GM_ADDR c, ATVC::EleWiseParam param)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto op = ATVC::Kernel::EleWiseOpTemplate<AddComputeFunc<Traits>>();  // 将AddComputeFunc仿函数作为模板参数传入，实例化EleWiseOpTemplate模板类
    op.Run(a, b, c, &param); // 按照输入、输出、param的顺序传入Run函数，实现GM->GM的数据计算
}
```

## 算子调用
完成`AddCustom`核函数实现后，用户可在host侧通过[<<<>>>](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha002/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0014.html)的方式调用核函数。
`AddCustom`的关键调用代码如下：
```cpp
// 声明运行态参数param
ATVC::ElewiseParam param;
// totalCnt描述EleWise单输入的元素个数
int32_t totalCnt = 1024; 
// ADD_OPTRAITS为ADD算子描述原型，根据算子输入输出个数和实际元素数量计算出Tiling数据后填入param中
if (!ATVC::Host::CalcEleWiseTiling<ADD_OPTRAITS>(totalCnt, param)) {
    printf("Elewise tiling error.\n");
    return -1;
};

uint32_t blockNum = param.tilingData.blockNum;
// 调用核函数
// aDevice  Device上的gm地址，指向Add算子第一个输入
// bDevice  Device上的gm地址，指向Add算子第二个输入
// cDevice  Device上的gm地址，指向Add算子第一个输出
// param    Device上的gm地址，指向运行态ATVC::EleWiseParam数据
AddCustom<ADD_OPTRAITS><<<blockNum, nullptr, stream>>>(aDevice, bDevice, cDevice, param);
```
相比于通用的算子调用，ATVC内部封装实现了kernel数据搬入搬出等底层通用操作及通用tiling计算，实现了高效的算子开发模式。

## 算子编译&执行
完成算子代码编写后，调用以下命令编译代码并执行：
```bash
cd ./ops_templates/atvc/examples
bash run_examples.sh add
```

至此，您已完成Add算子开发的学习，可以参考[ATVC开发指南](02_developer_guide.md)了解更多ATVC模板的使用指导，可以参考[examples目录](../examples/)了解更多算子样例。
