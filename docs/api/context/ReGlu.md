# ReGlu<a name="ZH-CN_TOPIC_0000001791730910"></a>

## 产品支持情况<a name="section1586581915393"></a>

<a name="table169596713360"></a>
<table><thead align="left"><tr id="row129590715369"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p17959971362"><a name="p17959971362"></a><a name="p17959971362"></a><span id="ph895914718367"><a name="ph895914718367"></a><a name="ph895914718367"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p89594763612"><a name="p89594763612"></a><a name="p89594763612"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row18959157103612"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p13959117193618"><a name="p13959117193618"></a><a name="p13959117193618"></a><span id="ph9959117173614"><a name="ph9959117173614"></a><a name="ph9959117173614"></a><term id="zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term12835255145414"><a name="zh-cn_topic_0000001312391781_term12835255145414"></a><a name="zh-cn_topic_0000001312391781_term12835255145414"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1095914793613"><a name="p1095914793613"></a><a name="p1095914793613"></a>√</p>
</td>
</tr>
<tr id="row89591478362"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p7959157163619"><a name="p7959157163619"></a><a name="p7959157163619"></a><span id="ph1995997193619"><a name="ph1995997193619"></a><a name="ph1995997193619"></a><term id="zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term1551319498507"><a name="zh-cn_topic_0000001312391781_term1551319498507"></a><a name="zh-cn_topic_0000001312391781_term1551319498507"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p149598793615"><a name="p149598793615"></a><a name="p149598793615"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

ReGlu是一种GLU变体，使用Relu作为激活函数，计算公式如下：

![](figures/zh-cn_formulaimage_0000002316175130.png)

其中Relu激活函数的计算公式如下：

![](figures/zh-cn_formulaimage_0000001792534088.png)

## 函数原型<a name="section620mcpsimp"></a>

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <typename T, bool isReuseSource = false>
    __aicore__ inline void ReGlu(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0, const LocalTensor<T>& srcTensor1, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
    ```

-   接口框架申请临时空间

    ```
    template <typename T, bool isReuseSource = false>
    __aicore__ inline void ReGlu(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0, const LocalTensor<T>& srcTensor1, const uint32_t calCount)
    ```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者**通过sharedTmpBuffer入参传入**和**接口框架申请**两种方式。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过[GetReGluMaxMinTmpSize](GetReGluMaxMinTmpSize.md)中提供的接口获取需要预留空间范围的大小。

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table575571914269"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001538537601_row18755131942614"><th class="cellrowborder" valign="top" width="19.39%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001538537601_p675519193268"><a name="zh-cn_topic_0000001538537601_p675519193268"></a><a name="zh-cn_topic_0000001538537601_p675519193268"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.61%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001538537601_p375511918267"><a name="zh-cn_topic_0000001538537601_p375511918267"></a><a name="zh-cn_topic_0000001538537601_p375511918267"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001538537601_row14755141911264"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001538537601_p47551198266"><a name="zh-cn_topic_0000001538537601_p47551198266"></a><a name="zh-cn_topic_0000001538537601_p47551198266"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001538537601_p125969172719"><a name="zh-cn_topic_0000001538537601_p125969172719"></a><a name="zh-cn_topic_0000001538537601_p125969172719"></a>操作数的数据类型。</p>
<p id="zh-cn_topic_0000001538537601_p44574117347"><a name="zh-cn_topic_0000001538537601_p44574117347"></a><a name="zh-cn_topic_0000001538537601_p44574117347"></a><span id="ph3804154652813"><a name="ph3804154652813"></a><a name="ph3804154652813"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_1"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term12835255145414_1"><a name="zh-cn_topic_0000001312391781_term12835255145414_1"></a><a name="zh-cn_topic_0000001312391781_term12835255145414_1"></a>Atlas A3 推理系列产品</term></span>，支持的数据类型为：half、bfloat16_t、float。</p>
<p id="zh-cn_topic_0000001538537601_p10451541143416"><a name="zh-cn_topic_0000001538537601_p10451541143416"></a><a name="zh-cn_topic_0000001538537601_p10451541143416"></a><span id="ph1158915592814"><a name="ph1158915592814"></a><a name="ph1158915592814"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_1"><a name="zh-cn_topic_0000001312391781_term11962195213215_1"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_1"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term1551319498507_1"><a name="zh-cn_topic_0000001312391781_term1551319498507_1"></a><a name="zh-cn_topic_0000001312391781_term1551319498507_1"></a>Atlas A2 推理系列产品</term></span>，支持的数据类型为：half、bfloat16_t、float。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001538537601_row9756719122620"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001538537601_p1682112447268"><a name="zh-cn_topic_0000001538537601_p1682112447268"></a><a name="zh-cn_topic_0000001538537601_p1682112447268"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001538537601_p98212044172612"><a name="zh-cn_topic_0000001538537601_p98212044172612"></a><a name="zh-cn_topic_0000001538537601_p98212044172612"></a>是否允许修改源操作数。该参数预留，传入默认值false即可。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table148471830151913"></a>
<table><thead align="left"><tr id="row1984733010194"><th class="cellrowborder" valign="top" width="17.380000000000003%" id="mcps1.2.4.1.1"><p id="p2847730181917"><a name="p2847730181917"></a><a name="p2847730181917"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.74%" id="mcps1.2.4.1.2"><p id="p58476303197"><a name="p58476303197"></a><a name="p58476303197"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.88%" id="mcps1.2.4.1.3"><p id="p10847203021913"><a name="p10847203021913"></a><a name="p10847203021913"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row98477303196"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p15847183018194"><a name="p15847183018194"></a><a name="p15847183018194"></a>dstTensor</p>
</td>
<td class="cellrowborder" valign="top" width="9.74%" headers="mcps1.2.4.1.2 "><p id="p148471930161917"><a name="p148471930161917"></a><a name="p148471930161917"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.88%" headers="mcps1.2.4.1.3 "><p id="p17444349398"><a name="p17444349398"></a><a name="p17444349398"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000001530181537_ph173308471594"><a name="zh-cn_topic_0000001530181537_ph173308471594"></a><a name="zh-cn_topic_0000001530181537_ph173308471594"></a><span id="zh-cn_topic_0000001530181537_ph9902231466"><a name="zh-cn_topic_0000001530181537_ph9902231466"></a><a name="zh-cn_topic_0000001530181537_ph9902231466"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816"><a name="zh-cn_topic_0000001530181537_ph1782115034816"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row11848103091920"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p58481330191917"><a name="p58481330191917"></a><a name="p58481330191917"></a>srcTensor0</p>
</td>
<td class="cellrowborder" valign="top" width="9.74%" headers="mcps1.2.4.1.2 "><p id="p158485305196"><a name="p158485305196"></a><a name="p158485305196"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.88%" headers="mcps1.2.4.1.3 "><p id="p3707144233915"><a name="p3707144233915"></a><a name="p3707144233915"></a>源操作数。</p>
<p id="p261714443395"><a name="p261714443395"></a><a name="p261714443395"></a><span id="zh-cn_topic_0000001530181537_ph173308471594_1"><a name="zh-cn_topic_0000001530181537_ph173308471594_1"></a><a name="zh-cn_topic_0000001530181537_ph173308471594_1"></a><span id="zh-cn_topic_0000001530181537_ph9902231466_1"><a name="zh-cn_topic_0000001530181537_ph9902231466_1"></a><a name="zh-cn_topic_0000001530181537_ph9902231466_1"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816_1"><a name="zh-cn_topic_0000001530181537_ph1782115034816_1"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p515144315188"><a name="p515144315188"></a><a name="p515144315188"></a>源操作数的数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row16916354172710"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p1763165913275"><a name="p1763165913275"></a><a name="p1763165913275"></a>srcTensor1</p>
</td>
<td class="cellrowborder" valign="top" width="9.74%" headers="mcps1.2.4.1.2 "><p id="p18631175915277"><a name="p18631175915277"></a><a name="p18631175915277"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.88%" headers="mcps1.2.4.1.3 "><p id="p16631359182714"><a name="p16631359182714"></a><a name="p16631359182714"></a>源操作数。</p>
<p id="p136319590276"><a name="p136319590276"></a><a name="p136319590276"></a><span id="zh-cn_topic_0000001530181537_ph173308471594_2"><a name="zh-cn_topic_0000001530181537_ph173308471594_2"></a><a name="zh-cn_topic_0000001530181537_ph173308471594_2"></a><span id="zh-cn_topic_0000001530181537_ph9902231466_2"><a name="zh-cn_topic_0000001530181537_ph9902231466_2"></a><a name="zh-cn_topic_0000001530181537_ph9902231466_2"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816_2"><a name="zh-cn_topic_0000001530181537_ph1782115034816_2"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p4631195916270"><a name="p4631195916270"></a><a name="p4631195916270"></a>源操作数的数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row4848123011192"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p1313415271911"><a name="p1313415271911"></a><a name="p1313415271911"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="9.74%" headers="mcps1.2.4.1.2 "><p id="p5133352201914"><a name="p5133352201914"></a><a name="p5133352201914"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.88%" headers="mcps1.2.4.1.3 "><p id="p1148552183919"><a name="p1148552183919"></a><a name="p1148552183919"></a>临时缓存。</p>
<p id="p5983205412394"><a name="p5983205412394"></a><a name="p5983205412394"></a><span id="zh-cn_topic_0000001530181537_ph173308471594_3"><a name="zh-cn_topic_0000001530181537_ph173308471594_3"></a><a name="zh-cn_topic_0000001530181537_ph173308471594_3"></a><span id="zh-cn_topic_0000001530181537_ph9902231466_3"><a name="zh-cn_topic_0000001530181537_ph9902231466_3"></a><a name="zh-cn_topic_0000001530181537_ph9902231466_3"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816_3"><a name="zh-cn_topic_0000001530181537_ph1782115034816_3"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816_3"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p104071111204211"><a name="p104071111204211"></a><a name="p104071111204211"></a>用于ReGlu内部复杂计算时存储中间变量，由开发者提供。</p>
<p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>临时空间大小BufferSize的获取方式请参考<a href="GetReGluMaxMinTmpSize.md">GetReGluMaxMinTmpSize</a>。</p>
</td>
</tr>
<tr id="row128437174616"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p1328514137476"><a name="p1328514137476"></a><a name="p1328514137476"></a>calCount</p>
</td>
<td class="cellrowborder" valign="top" width="9.74%" headers="mcps1.2.4.1.2 "><p id="p17337238144716"><a name="p17337238144716"></a><a name="p17337238144716"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.88%" headers="mcps1.2.4.1.3 "><p id="p169435510340"><a name="p169435510340"></a><a name="p169435510340"></a>实际计算数据元素个数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section38228281712"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   **不支持源操作数与目的操作数地址重叠。**
-   不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。
-   当前仅支持ND格式的输入，不支持其他格式。

## 调用示例<a name="section642mcpsimp"></a>

kernel侧调用示例

```
#include "kernel_operator.h"
template <typename srcType>
class KernelReGlu
{
public:
    __aicore__ inline KernelReGlu() {}
    __aicore__ inline void Init(GM_ADDR src0Gm, GM_ADDR src1Gm, GM_ADDR dstGm, uint32_t srcSize)
    {
        dataSize = srcSize;
        src0Global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(src0Gm), dataSize);
        src1Global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(src1Gm), dataSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(dstGm), dataSize);
        pipe.InitBuffer(inQueueX, 1, dataSize * sizeof(srcType));
        pipe.InitBuffer(inQueueY, 1, dataSize * sizeof(srcType));
        pipe.InitBuffer(outQueue, 1, dataSize * sizeof(srcType));
        if (sizeof(srcType) != sizeof(float))
        {
            pipe.InitBuffer(calcBufs, dataSize * (sizeof(float) / sizeof(uint8_t)) * 3);
        }
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<srcType> src0Local = inQueueX.AllocTensor<srcType>();
        AscendC::LocalTensor<srcType> src1Local = inQueueY.AllocTensor<srcType>();
        AscendC::DataCopy(src0Local, src0Global, dataSize);
        AscendC::DataCopy(src1Local, src1Global, dataSize);
        inQueueX.EnQue(src0Local);
        inQueueY.EnQue(src1Local);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();
        AscendC::LocalTensor<srcType> src0Local = inQueueX.DeQue<srcType>();
        AscendC::LocalTensor<srcType> src1Local = inQueueY.DeQue<srcType>();
        AscendC::LocalTensor<uint8_t> tmpLocal;
        if (sizeof(srcType) != sizeof(float))
        {
            tmpLocal = calcBufs.Get<uint8_t>();
            AscendC::ReGlu<srcType, false>(dstLocal, src0Local, src1Local, tmpLocal, dataSize);
        }
        else
        {
            AscendC::ReGlu<srcType, false>(dstLocal, src0Local, src1Local, dataSize);
        }
        outQueue.EnQue<srcType>(dstLocal);
        inQueueX.FreeTensor(src0Local);
        inQueueY.FreeTensor(src1Local);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<srcType> dstLocal = outQueue.DeQue<srcType>();
        AscendC::DataCopy(dstGlobal, dstLocal, dataSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    AscendC::GlobalTensor<srcType> src0Global;
    AscendC::GlobalTensor<srcType> src1Global;
    AscendC::GlobalTensor<srcType> dstGlobal;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBufs;
    uint32_t dataSize = 0;
};
template <typename dataType>
__aicore__ void kernel_reglu_operator(GM_ADDR src0Gm, GM_ADDR src1Gm, GM_ADDR dstGm, uint32_t srcSize)
{
    KernelReGlu<dataType> op;
    op.Init(src0Gm, src1Gm, dstGm, srcSize);
    op.Process();
}
```

结果示例如下：

```
输入数据(srcLocal0): 
[ 22.28125    78.375     -10.3515625 -80.75      -22.8125     84.375
  -8.96875    70.5       -51.75       66.875      69.8125      5.2734375
 -51.         50.5       -30.765625  -52.125       8.03125    75.8125
  50.4375    -97.1875    -80.6875     17.125     -30.640625  -13.671875
  92.375      68.8125     53.75        5.1054688  39.6875    -46.71875
  90.25       67.75     ]
输入数据(srcLocal1): 
[ 61.46875   -36.5625    -93.3125    -87.6875    -17.96875   -88.125
 -46.65625   -18.78125    13.4921875 -87.875      65.75      -25.96875
 -44.5625     53.        -69.375      96.5       -24.703125   77.5625
  78.875      -6.0898438 -40.5625    -69.625      57.         18.640625
 -73.875      94.375      91.5        -9.7109375  84.125      79.0625
  88.5        96.3125   ]
输出数据(dstLocal): 
[ 0.0000e+00  0.0000e+00  0.0000e+00 -6.5450e+02  0.0000e+00  1.2544e+02
  3.7880e+03  1.0519e+02 -0.0000e+00 -0.0000e+00 -0.0000e+00  0.0000e+00
 -2.0110e+03  0.0000e+00 -2.8020e+03 -0.0000e+00  0.0000e+00 -2.6120e+03
  6.8840e+03 -0.0000e+00  8.6550e+02 -0.0000e+00  0.0000e+00 -7.4120e+03
 -1.9700e+03  2.3140e+03 -0.0000e+00  0.0000e+00 -0.0000e+00  7.6760e+03
 -4.8828e-01 -0.0000e+00]
```

