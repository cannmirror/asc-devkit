# AscendQuant<a name="ZH-CN_TOPIC_0000001666545372"></a>

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

按元素做量化计算，比如将half/float数据类型量化为int8\_t数据类型。计算公式如下，round表示四舍六入五成双取整：

-   PER\_TENSOR量化：整个srcTensor对应一个量化参数，量化参数的shape为\[1\]。

    ![](figures/zh-cn_formulaimage_0000001666705128.png)

-   PER\_CHANNEL量化：srcTensor的shape为\[m, n\], 每个channel维度对应一个量化参数，量化参数的shape为\[n\]。

    ![](figures/zh-cn_formulaimage_0000001738671878.png)

## 实现原理<a name="section13229175017585"></a>

**图 1**  AscendQuant算法框图scale和offset都是scalar<a name="fig966236152318"></a>  
![](figures/AscendQuant算法框图scale和offset都是scalar.png "AscendQuant算法框图scale和offset都是scalar")

**图 2**  AscendQuant算法框图scale和offset都是Tensor<a name="fig2405134711019"></a>  
![](figures/AscendQuant算法框图scale和offset都是Tensor.png "AscendQuant算法框图scale和offset都是Tensor")

**图 3**  AscendQuant算法框图scale是Tensor&offset是Scalar<a name="fig6542182812108"></a>  
![](figures/AscendQuant算法框图scale是Tensor-offset是Scalar.png "AscendQuant算法框图scale是Tensor-offset是Scalar")

如上图所示是AscendQuant内部算法框图，计算过程大致描述为如下几步，均在Vector上进行：

1.  精度转换：当输入的src，scale或者offset是float类型时，将其转换为half类型；
2.  broadcast：当输入的scale或者offset是向量时，将其broadcast成和src相同维度；
3.  计算scale：当src和scale为向量时做Mul计算，当scale是scalar时做Muls计算，得到Tmp1；
4.  计算offset：当Tmp1和offset为向量时做Add计算，当offset是scalar时做Adds计算，得到Tmp2；
5.  精度转换：将Tmp2从half转换成int8\_t类型，得到output。

## 函数原型<a name="section19670529163214"></a>

-   dstTensor为int8\_t数据类型
    -   PER\_TENSOR量化：
        -   通过sharedTmpBuffer入参传入临时空间
            -   源操作数Tensor全部/部分参与计算

                ```
                template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
                __aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const float scale, const float offset, const uint32_t calCount)
                ```

            -   源操作数Tensor全部参与计算

                ```
                template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
                __aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const float scale, const float offset)
                ```

        -   接口框架申请临时空间
            -   源操作数Tensor全部/部分参与计算

                ```
                template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
                __aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const float scale, const float offset, const uint32_t calCount)
                ```

            -   源操作数Tensor全部参与计算

                ```
                template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
                __aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const float scale, const float offset)
                ```

    -   PER\_CHANNEL量化：
        -   通过sharedTmpBuffer入参传入临时空间
            -   源操作数Tensor全部/部分参与计算

                ```
                template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
                __aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor, const T offset, const uint32_t scaleCount, const uint32_t calCount)
                ```

                ```
                template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
                __aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor, const LocalTensor<T>& offsetTensor, const uint32_t scaleCount, const uint32_t offsetCount, const uint32_t calCount)
                ```

            -   源操作数Tensor全部参与计算

                ```
                template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
                __aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor, const T offset)
                ```

                ```
                template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
                __aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor, const LocalTensor<T>& offsetTensor)
                ```

        -   接口框架申请临时空间
            -   源操作数Tensor全部/部分参与计算

                ```
                template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
                __aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& scaleTensor, const T offset, const uint32_t scaleCount, const uint32_t calCount)
                ```

                ```
                template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
                __aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& scaleTensor, const LocalTensor<T>& offsetTensor, const uint32_t scaleCount, const uint32_t offsetCount, const uint32_t calCount)
                ```

            -   源操作数Tensor全部参与计算

                ```
                template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
                __aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& scaleTensor, const T offset)
                ```

                ```
                template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
                __aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& scaleTensor, const LocalTensor<T>& offsetTensor)
                ```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为sharedTmpBuffer申请空间。临时空间大小BufferSize的获取方式如下：通过[GetAscendQuantMaxMinTmpSize](GetAscendQuantMaxMinTmpSize.md)中提供的GetAscendQuantMaxMinTmpSize接口获取需要预留空间的范围大小。

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
<p id="p14715112410363"><a name="p14715112410363"></a><a name="p14715112410363"></a><span id="ph571592417363"><a name="ph571592417363"></a><a name="ph571592417363"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_1"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term12835255145414_1"><a name="zh-cn_topic_0000001312391781_term12835255145414_1"></a><a name="zh-cn_topic_0000001312391781_term12835255145414_1"></a>Atlas A3 推理系列产品</term></span>，支持的数据类型为：half、float。</p>
<p id="p14496195212317"><a name="p14496195212317"></a><a name="p14496195212317"></a><span id="ph104968521634"><a name="ph104968521634"></a><a name="ph104968521634"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_1"><a name="zh-cn_topic_0000001312391781_term11962195213215_1"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_1"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term1551319498507_1"><a name="zh-cn_topic_0000001312391781_term1551319498507_1"></a><a name="zh-cn_topic_0000001312391781_term1551319498507_1"></a>Atlas A2 推理系列产品</term></span>，支持的数据类型为：half、float。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001538537601_row9756719122620"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001538537601_p1682112447268"><a name="zh-cn_topic_0000001538537601_p1682112447268"></a><a name="zh-cn_topic_0000001538537601_p1682112447268"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001538537601_p98212044172612"><a name="zh-cn_topic_0000001538537601_p98212044172612"></a><a name="zh-cn_topic_0000001538537601_p98212044172612"></a>是否允许修改源操作数。该参数预留，传入默认值false即可。</p>
</td>
</tr>
<tr id="row1529110458389"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p2292745133816"><a name="p2292745133816"></a><a name="p2292745133816"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p1744211392255"><a name="p1744211392255"></a><a name="p1744211392255"></a>结构体模板参数，此参数可选配，AscendQuantConfig类型，具体定义如下。</p>
<a name="screen16476116195910"></a><a name="screen16476116195910"></a><pre class="code_wrap" codetype="Cpp" id="screen16476116195910">struct AscendQuantConfig{
uint32_t calcCount = 0;
uint32_t offsetCount = 0;
uint32_t scaleCount = 0;
uint32_t workLocalSize = 0;
};</pre>
<a name="ul1882194222413"></a><a name="ul1882194222413"></a><ul id="ul1882194222413"><li>calcCount：实际计算数据元素个数。calcCount∈[0, srcTensor.GetSize()]，在调用带有scaleCount入参的接口时，calcCount若取非零值则必须是scaleCount的整数倍。</li><li>offsetCount：实际量化参数元素个数。offsetCount∈[0, offsetTensor.GetSize()]，offsetCount与scaleCount的取值必须相等，要求是32的整数倍。若调用的接口不含offsetCount入参，取值为0即可。</li><li>scaleCount：实际量化参数元素个数。scaleCount∈[0, scaleTensor.GetSize()]，要求是32的整数倍。若调用的接口不含scaleCount入参，取值为0即可。</li><li>workLocalSize：临时缓存sharedTmpBuffer的大小，sharedTmpBuffer的大小/workLocalSize的获取方式请参考<a href="GetAscendQuantMaxMinTmpSize.md">GetAscendQuantMaxMinTmpSize</a>。该参数取值不能大于sharedTmpBuffer的大小。若调用的接口不含sharedTmpBuffer入参，取值为0即可。</li></ul>
<p id="p1189143944315"><a name="p1189143944315"></a><a name="p1189143944315"></a>当上述参数的取值满足如下任一种场景，将使能参数常量化，即编译过程中使用常量化的相关参数，从而减少Scalar计算。</p>
<a name="ul12991368432"></a><a name="ul12991368432"></a><ul id="ul12991368432"><li>若调用的接口不含scaleCount入参，calcCount和workLocalSize取值为非0时，使能参数常量化。</li><li>若调用的接口带有scaleCount入参，scaleCount、calcCount和workLocalSize取值为非0时，使能参数常量化。</li></ul>
<p id="p76421594583"><a name="p76421594583"></a><a name="p76421594583"></a>默认参数的配置示例如下。</p>
<a name="screen19241326175913"></a><a name="screen19241326175913"></a><pre class="code_wrap" codetype="Cpp" id="screen19241326175913">constexpr AscendQuantConfig ASCEND_QUANT_DEFAULT_CFG = {0, 0, 0, 0};</pre>
</td>
</tr>
</tbody>
</table>

**表 2**  PER\_TENSOR接口参数说明

<a name="table44731299481"></a>
<table><thead align="left"><tr id="row247482914489"><th class="cellrowborder" valign="top" width="16.45%" id="mcps1.2.4.1.1"><p id="p147413295483"><a name="p147413295483"></a><a name="p147413295483"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.31%" id="mcps1.2.4.1.2"><p id="p1147432994819"><a name="p1147432994819"></a><a name="p1147432994819"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.24000000000001%" id="mcps1.2.4.1.3"><p id="p74749297483"><a name="p74749297483"></a><a name="p74749297483"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row12474329104814"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p1047411294482"><a name="p1047411294482"></a><a name="p1047411294482"></a>dstTensor</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p047412984813"><a name="p047412984813"></a><a name="p047412984813"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p3989161814016"><a name="p3989161814016"></a><a name="p3989161814016"></a>目的操作数。</p>
<p id="p1747492917489"><a name="p1747492917489"></a><a name="p1747492917489"></a><span id="zh-cn_topic_0000001530181537_ph173308471594"><a name="zh-cn_topic_0000001530181537_ph173308471594"></a><a name="zh-cn_topic_0000001530181537_ph173308471594"></a><span id="zh-cn_topic_0000001530181537_ph9902231466"><a name="zh-cn_topic_0000001530181537_ph9902231466"></a><a name="zh-cn_topic_0000001530181537_ph9902231466"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816"><a name="zh-cn_topic_0000001530181537_ph1782115034816"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row18474729124817"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p54741029164810"><a name="p54741029164810"></a><a name="p54741029164810"></a>srcTensor</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p144741829194814"><a name="p144741829194814"></a><a name="p144741829194814"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p6914123244017"><a name="p6914123244017"></a><a name="p6914123244017"></a>源操作数。</p>
<p id="p1493334184019"><a name="p1493334184019"></a><a name="p1493334184019"></a><span id="zh-cn_topic_0000001530181537_ph173308471594_1"><a name="zh-cn_topic_0000001530181537_ph173308471594_1"></a><a name="zh-cn_topic_0000001530181537_ph173308471594_1"></a><span id="zh-cn_topic_0000001530181537_ph9902231466_1"><a name="zh-cn_topic_0000001530181537_ph9902231466_1"></a><a name="zh-cn_topic_0000001530181537_ph9902231466_1"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816_1"><a name="zh-cn_topic_0000001530181537_ph1782115034816_1"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row1747412296483"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p74741029204817"><a name="p74741029204817"></a><a name="p74741029204817"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p1747452954810"><a name="p1747452954810"></a><a name="p1747452954810"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p191160465422"><a name="p191160465422"></a><a name="p191160465422"></a>临时缓存。</p>
<p id="p979635010404"><a name="p979635010404"></a><a name="p979635010404"></a><span id="zh-cn_topic_0000001530181537_ph173308471594_2"><a name="zh-cn_topic_0000001530181537_ph173308471594_2"></a><a name="zh-cn_topic_0000001530181537_ph173308471594_2"></a><span id="zh-cn_topic_0000001530181537_ph9902231466_2"><a name="zh-cn_topic_0000001530181537_ph9902231466_2"></a><a name="zh-cn_topic_0000001530181537_ph9902231466_2"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816_2"><a name="zh-cn_topic_0000001530181537_ph1782115034816_2"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>临时空间大小BufferSize的获取方式请参考<a href="GetAscendQuantMaxMinTmpSize.md">GetAscendQuantMaxMinTmpSize</a>。</p>
</td>
</tr>
<tr id="row524952410266"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p19249424182610"><a name="p19249424182610"></a><a name="p19249424182610"></a>scale</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p172491124122610"><a name="p172491124122610"></a><a name="p172491124122610"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p26761711274"><a name="p26761711274"></a><a name="p26761711274"></a>量化参数。</p>
<p id="p3249924172610"><a name="p3249924172610"></a><a name="p3249924172610"></a>类型为Scalar，支持的数据类型为float。</p>
</td>
</tr>
<tr id="row8946172732612"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p4871237282"><a name="p4871237282"></a><a name="p4871237282"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p118710312812"><a name="p118710312812"></a><a name="p118710312812"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p1587193112813"><a name="p1587193112813"></a><a name="p1587193112813"></a>量化参数。</p>
<p id="p13871143192817"><a name="p13871143192817"></a><a name="p13871143192817"></a>类型为Scalar，支持的数据类型为float。</p>
</td>
</tr>
<tr id="row16421712252"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p1949611581317"><a name="p1949611581317"></a><a name="p1949611581317"></a>calCount</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p174961758436"><a name="p174961758436"></a><a name="p174961758436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p11378261546"><a name="p11378261546"></a><a name="p11378261546"></a>参与计算的元素个数。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  PER\_CHANNEL接口参数说明

<a name="table8690143212334"></a>
<table><thead align="left"><tr id="row969063243317"><th class="cellrowborder" valign="top" width="16.45%" id="mcps1.2.4.1.1"><p id="p11690123212330"><a name="p11690123212330"></a><a name="p11690123212330"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.31%" id="mcps1.2.4.1.2"><p id="p769083263318"><a name="p769083263318"></a><a name="p769083263318"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.24000000000001%" id="mcps1.2.4.1.3"><p id="p19690163216331"><a name="p19690163216331"></a><a name="p19690163216331"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1369014325334"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p1669014322334"><a name="p1669014322334"></a><a name="p1669014322334"></a>dstTensor</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p1690183223314"><a name="p1690183223314"></a><a name="p1690183223314"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p0690173203319"><a name="p0690173203319"></a><a name="p0690173203319"></a>目的操作数。</p>
<p id="p3690232203314"><a name="p3690232203314"></a><a name="p3690232203314"></a><span id="zh-cn_topic_0000001530181537_ph173308471594_3"><a name="zh-cn_topic_0000001530181537_ph173308471594_3"></a><a name="zh-cn_topic_0000001530181537_ph173308471594_3"></a><span id="zh-cn_topic_0000001530181537_ph9902231466_3"><a name="zh-cn_topic_0000001530181537_ph9902231466_3"></a><a name="zh-cn_topic_0000001530181537_ph9902231466_3"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816_3"><a name="zh-cn_topic_0000001530181537_ph1782115034816_3"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816_3"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row126918321336"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p2691632133315"><a name="p2691632133315"></a><a name="p2691632133315"></a>srcTensor</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p8691193273312"><a name="p8691193273312"></a><a name="p8691193273312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p176912324339"><a name="p176912324339"></a><a name="p176912324339"></a>源操作数。</p>
<p id="p186912032113317"><a name="p186912032113317"></a><a name="p186912032113317"></a><span id="zh-cn_topic_0000001530181537_ph173308471594_4"><a name="zh-cn_topic_0000001530181537_ph173308471594_4"></a><a name="zh-cn_topic_0000001530181537_ph173308471594_4"></a><span id="zh-cn_topic_0000001530181537_ph9902231466_4"><a name="zh-cn_topic_0000001530181537_ph9902231466_4"></a><a name="zh-cn_topic_0000001530181537_ph9902231466_4"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816_4"><a name="zh-cn_topic_0000001530181537_ph1782115034816_4"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816_4"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row569133217332"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p869113263312"><a name="p869113263312"></a><a name="p869113263312"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p12691183223312"><a name="p12691183223312"></a><a name="p12691183223312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p166910324330"><a name="p166910324330"></a><a name="p166910324330"></a>临时缓存。</p>
<p id="p146911332173314"><a name="p146911332173314"></a><a name="p146911332173314"></a><span id="zh-cn_topic_0000001530181537_ph173308471594_5"><a name="zh-cn_topic_0000001530181537_ph173308471594_5"></a><a name="zh-cn_topic_0000001530181537_ph173308471594_5"></a><span id="zh-cn_topic_0000001530181537_ph9902231466_5"><a name="zh-cn_topic_0000001530181537_ph9902231466_5"></a><a name="zh-cn_topic_0000001530181537_ph9902231466_5"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816_5"><a name="zh-cn_topic_0000001530181537_ph1782115034816_5"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816_5"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1691163215337"><a name="p1691163215337"></a><a name="p1691163215337"></a>临时空间大小BufferSize的获取方式请参考<a href="GetAscendQuantMaxMinTmpSize.md">GetAscendQuantMaxMinTmpSize</a>。</p>
</td>
</tr>
<tr id="row3691143243310"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p9691123216338"><a name="p9691123216338"></a><a name="p9691123216338"></a>scaleTensor</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p2691332113316"><a name="p2691332113316"></a><a name="p2691332113316"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p66910324338"><a name="p66910324338"></a><a name="p66910324338"></a>量化参数。</p>
<p id="p13772144914345"><a name="p13772144914345"></a><a name="p13772144914345"></a><span id="zh-cn_topic_0000001530181537_ph173308471594_6"><a name="zh-cn_topic_0000001530181537_ph173308471594_6"></a><a name="zh-cn_topic_0000001530181537_ph173308471594_6"></a><span id="zh-cn_topic_0000001530181537_ph9902231466_6"><a name="zh-cn_topic_0000001530181537_ph9902231466_6"></a><a name="zh-cn_topic_0000001530181537_ph9902231466_6"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816_6"><a name="zh-cn_topic_0000001530181537_ph1782115034816_6"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816_6"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row11691232163318"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p1669273263311"><a name="p1669273263311"></a><a name="p1669273263311"></a>offsetTensor</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p16692123203315"><a name="p16692123203315"></a><a name="p16692123203315"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p156921323335"><a name="p156921323335"></a><a name="p156921323335"></a>量化参数。</p>
<p id="p8389256113411"><a name="p8389256113411"></a><a name="p8389256113411"></a><span id="zh-cn_topic_0000001530181537_ph173308471594_7"><a name="zh-cn_topic_0000001530181537_ph173308471594_7"></a><a name="zh-cn_topic_0000001530181537_ph173308471594_7"></a><span id="zh-cn_topic_0000001530181537_ph9902231466_7"><a name="zh-cn_topic_0000001530181537_ph9902231466_7"></a><a name="zh-cn_topic_0000001530181537_ph9902231466_7"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816_7"><a name="zh-cn_topic_0000001530181537_ph1782115034816_7"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816_7"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row192947255353"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p97107538356"><a name="p97107538356"></a><a name="p97107538356"></a>scaleCount</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p20295162511352"><a name="p20295162511352"></a><a name="p20295162511352"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p573415227366"><a name="p573415227366"></a><a name="p573415227366"></a>实际量化参数元素个数，且scaleCount∈[0, min(scaleTensor.GetSize(),dstTensor.GetSize())]，要求是32的整数倍。</p>
</td>
</tr>
<tr id="row666619297352"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p1070112544354"><a name="p1070112544354"></a><a name="p1070112544354"></a>offsetCount</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p1766742953517"><a name="p1766742953517"></a><a name="p1766742953517"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p4535132373614"><a name="p4535132373614"></a><a name="p4535132373614"></a>实际量化参数元素个数，且offsetCount∈[0, min(offsetTensor.GetSize(),dstTensor.GetSize())]，并且和scaleCount必须相等，要求是32的整数倍。</p>
</td>
</tr>
<tr id="row06925328336"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p569293273314"><a name="p569293273314"></a><a name="p569293273314"></a>calCount</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p8692432163315"><a name="p8692432163315"></a><a name="p8692432163315"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p6692103213316"><a name="p6692103213316"></a><a name="p6692103213316"></a>参与计算的元素个数。calCount必须是scaleCount的整数倍。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section38228281712"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   源操作数与目的操作数可以复用。
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   输入输出操作数参与计算的数据长度要求32B对齐。
-   当Scale为float类型时，其取值范围仍为half类型的取值范围。

## 调用示例<a name="section642mcpsimp"></a>

```
// 输入shape为1024
uint32_t dataSize = 1024; 
// 输入类型为float/half, scale=2.0, offset=0.9，预留临时空间
AscendC::AscendQuant<srcType>(dstLocal, srcLocal, 2.0f, 0.9f, dataSize);
// 使用模板参数使能参数常量化的示例
// static constexpr AscendC::AscendQuantConfig static_config = {1024, 0, 0, 0};
// 使用AscendQuantConfig类型的参数static_config，传入模板参数将参数常量化
// AscendC::AscendQuant<srcType, false, static_config>(dstLocal, srcLocal, 2.0f, 0.9f, dataSize);
```

结果示例如下：

```
输入数据（srcLocal）: 
[-3.22      2.09     -2.025    -2.895    -1.349    -3.336     1.376
  2.453     3.861     1.085    -2.273     0.3923    0.3645   -2.127
 -3.09     -0.002726 -2.783     0.2615   -0.904     1.507    -1.017
  3.568     2.219     0.8643    0.922     1.144    -1.853     2.002
 -1.705     1.675    -3.482     1.519     0.4172    0.4307   -1.228
 -2.62      0.3354   -3.586     2.604     1.688    -3.646    -3.389
 -3.918     3.955     0.7954   -2.562    -1.085     2.91     -0.398
  3.771    -2.914     1.726     3.367     3.482     3.49      1.382
  3.512     0.1938   -0.4087   -3.75      2.873    -2.54      1.826
  3.738     3.188     2.676     0.724    -1.108    -2.682    -0.4783
  2.082    -0.462    -2.955    -2.543     3.98     -1.85      3.018
 -2.688     3.596    -0.799     1.222     1.686    -0.7925    3.295
 -3.568    -0.03836  -2.002    -1.212     1.927    -1.11      1.046
  3.793    -0.6226   -3.494    -3.371    -2.354    -1.7      -0.948
  2.682    -3.344     2.566     2.533    -1.335     1.405     3.867
  3.674     1.359     3.145    -1.22      1.054    -2.492    -1.214
  3.879     2.014     2.664    -2.863    -3.88      2.857     1.695
  2.852     2.893     2.367    -0.1832   -3.254    -1.49      1.13
  0.672    -1.863    -3.547     3.281    -1.573    -1.349    -3.547
 -3.766    -2.99     -3.203    -2.703    -2.793    -1.501     0.4785
 -1.216    -1.205     0.9097   -3.438     0.781    -1.505    -1.982
  0.2037    0.4595    0.759     0.844    -3.396     0.4778   -0.899
 -2.342    -0.961    -2.531    -0.10913  -3.516    -3.66      1.337
 -3.44      0.7495    1.958     2.775     0.0968   -3.       -2.13
 -1.818     2.664     2.066    -1.923     2.97     -2.047    -3.598
  0.1661   -0.179     3.186    -1.247     2.777    -3.344    -3.148
  2.275     2.916    -1.081    -3.213     2.87     -3.12     -3.066
 -0.6      -3.78     -3.012    -3.86     -0.707    -0.2203   -3.338
 -2.273     2.062    -2.422    -0.443    -1.333    -2.2      -1.478
 -2.816     1.134     0.2115   -2.459     3.842    -2.768     2.822
  1.3125   -2.143     1.971    -3.543    -0.07794  -0.1265    0.763
 -3.26      3.514     3.629     0.1902    1.277    -0.1652   -0.006435
 -1.25      2.258    -2.887     3.66      2.729    -3.27     -0.5615
 -3.176    -1.2295    1.556    -0.6626   -2.777     1.946    -0.338
 -2.977    -0.8135   -2.37      0.7764    3.525    -0.6196    2.436
  2.38     -1.708     0.814     0.4688   -1.255     1.04     -1.077
  3.176     1.859     0.9194    2.703     1.436     1.762     2.2
  1.794    -1.234    -2.148    -2.393     2.846     1.854     0.3428
 -2.379     0.2429   -1.561     2.582     0.6836    1.811    -2.53
 -3.951    -2.096    -2.639     2.02      2.799    -0.8936   -1.295
 -3.914    -1.82      2.541    -2.773     1.733     3.955    -3.092
  0.04095   0.82     -1.071     3.93     -3.158    -2.5      -0.5415
 -1.98     -0.1626    3.092    -1.3125    3.387    -2.496     2.355
 -3.033    -3.814    -3.191     2.686     1.377     1.381    -3.047
  2.127    -0.4927   -1.718     2.371    -0.1648    1.885    -0.6826
 -3.121    -2.379    -3.959    -2.164     2.262    -2.973     3.092
  2.111    -0.03732   2.836    -2.725     3.436     1.017     2.877
 -2.926     2.547     0.8574    2.643     2.646    -0.889     3.363
 -0.3147   -0.09546   0.0551   -3.947    -1.434    -0.6104   -3.41
 -2.176    -1.866     3.975    -3.031    -1.25      3.918     3.697
  3.21     -2.436    -3.281    -3.225     0.7856    2.043     1.415
 -2.252    -1.648     0.03824  -3.432     0.3271    1.458    -0.02289
 -0.643     1.441    -0.1847    1.062     3.545     0.367     1.796
 -1.687     2.06      0.2373    3.748    -2.752     2.73     -2.693
 -3.54     -2.275    -3.033    -1.622    -3.936     1.295     2.586
 -2.926    -2.314     2.527    -1.619    -0.04037  -3.225     1.771
  3.064    -1.173    -2.324     3.332    -0.8257    1.075    -3.287
  1.075    -2.262     1.419    -0.344    -0.4988    1.113     3.068
 -1.104     2.531     2.645     0.6333    0.3677   -3.186    -0.3726
  2.549    -0.3347    2.227    -3.963    -2.564     3.656     1.069
 -3.684    -1.388    -0.2568   -0.726     0.4883    1.946    -1.579
 -0.8438   -2.014     2.332     0.306    -3.305    -3.588    -1.038
  3.299     0.832     0.8594   -1.163     1.2705    2.018    -3.352
  2.537     2.111    -3.61      0.645    -2.459    -2.469     1.002
 -3.914     1.079    -0.9214   -2.111    -3.88     -0.5254   -1.908
 -1.19      3.559    -3.285    -2.266     3.672     0.001524 -1.964
 -1.742     1.895     3.887     1.737     0.909     0.5044    2.55
  0.8936    2.139    -3.658     1.828    -3.688    -3.26      1.436
 -1.321    -3.19      2.764    -3.305    -2.52     -2.441    -0.32
 -2.402     2.252    -1.527     0.719     0.2328    0.1766   -2.088
  3.729     0.844    -1.174    -0.7427    0.8296   -0.1885   -0.0379
  2.92      2.502     3.846     1.657    -3.58     -3.352    -3.904
 -2.43      1.159    -1.707     2.21      2.367    -0.5864   -1.647
  1.952   ]
输出数据（dstLocal）: 
[-6  5 -3 -5 -2 -6  4  6  9  3 -4  2  2 -3 -5  1 -5  1 -1  4 -1  8  5  3
  3  3 -3  5 -3  4 -6  4  2  2 -2 -4  2 -6  6  4 -6 -6 -7  9  2 -4 -1  7
  0  8 -5  4  8  8  8  4  8  1  0 -7  7 -4  5  8  7  6  2 -1 -4  0  5  0
 -5 -4  9 -3  7 -4  8 -1  3  4 -1  7 -6  1 -3 -2  5 -1  3  8  0 -6 -6 -4
 -2 -1  6 -6  6  6 -2  4  9  8  4  7 -2  3 -4 -2  9  5  6 -5 -7  7  4  7
  7  6  1 -6 -2  3  2 -3 -6  7 -2 -2 -6 -7 -5 -6 -5 -5 -2  2 -2 -2  3 -6
  2 -2 -3  1  2  2  3 -6  2 -1 -4 -1 -4  1 -6 -6  4 -6  2  5  6  1 -5 -3
 -3  6  5 -3  7 -3 -6  1  1  7 -2  6 -6 -5  5  7 -1 -6  7 -5 -5  0 -7 -5
 -7 -1  0 -6 -4  5 -4  0 -2 -3 -2 -5  3  1 -4  9 -5  7  4 -3  5 -6  1  1
  2 -6  8  8  1  3  1  1 -2  5 -5  8  6 -6  0 -5 -2  4  0 -5  5  0 -5 -1
 -4  2  8  0  6  6 -3  3  2 -2  3 -1  7  5  3  6  4  4  5  4 -2 -3 -4  7
  5  2 -4  1 -2  6  2  5 -4 -7 -3 -4  5  6 -1 -2 -7 -3  6 -5  4  9 -5  1
  3 -1  9 -5 -4  0 -3  1  7 -2  8 -4  6 -5 -7 -5  6  4  4 -5  5  0 -3  6
  1  5  0 -5 -4 -7 -3  5 -5  7  5  1  7 -5  8  3  7 -5  6  3  6  6 -1  8
  0  1  1 -7 -2  0 -6 -3 -3  9 -5 -2  9  8  7 -4 -6 -6  2  5  4 -4 -2  1
 -6  2  4  1  0  4  1  3  8  2  4 -2  5  1  8 -5  6 -4 -6 -4 -5 -2 -7  3
  6 -5 -4  6 -2  1 -6  4  7 -1 -4  8 -1  3 -6  3 -4  4  0  0  3  7 -1  6
  6  2  2 -5  0  6  0  5 -7 -4  8  3 -6 -2  0 -1  2  5 -2 -1 -3  6  2 -6
 -6 -1  7  3  3 -1  3  5 -6  6  5 -6  2 -4 -4  3 -7  3 -1 -3 -7  0 -3 -1
  8 -6 -4  8  1 -3 -3  5  9  4  3  2  6  3  5 -6  5 -6 -6  4 -2 -5  6 -6
 -4 -4  0 -4  5 -2  2  1  1 -3  8  3 -1 -1  3  1  1  7  6  9  4 -6 -6 -7
 -4  3 -3  5  6  0 -2  5]
```

