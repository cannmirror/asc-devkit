# AscendDequant<a name="ZH-CN_TOPIC_0000001863640817"></a>

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

按元素做反量化计算，比如将int32\_t数据类型反量化为half/float等数据类型。**本接口最多支持输入为二维数据，不支持更高维度的输入。**

-   假设输入srcTensor的shape为**（m, n）**，每行数据（即n个输入数据）所占字节数要求**32字节对齐**，每行中进行反量化的元素个数为**calCount**；
-   反量化系数deqScale可以为标量或者向量，为向量的情况下，calCount <= deqScale的元素个数，只有前CalCount个反量化系数生效；
-   输出dstTensor的shape为**（m, n\_dst）**， n \* sizeof\(dstT\)不满足32字节对齐时，需要**向上补齐为32字节**，n\_dst为向上补齐后的列数。

下面通过两个具体的示例来解释参数的配置和计算逻辑（下文中DequantParams类型为存储shape信息的结构体\{m, n, calCount\}）：

-   如下图示例中，srcTensor的数据类型为int32\_t，m = 4，n = 8，calCount = 4，表明srcTensor中每行进行反量化的元素个数为4，deqScale中的前4个数生效，后12个数不参与反量化计算；dstTensor的数据类型为bfloat16\_t，m = 4，n\_dst = 16 \(16 \* sizeof\(bfloat16\_t\) % 32 = 0\)。计算逻辑是srcTensor的每n个数为一行，对于每行中的前calCount个元素，该行srcTensor的第i个元素与deqScale的第i个元素进行相乘写入dstTensor对应行的第i个元素，dstTensor对应行的第calCount + 1个元素\~第n\_dst个元素均为不确定的值。

    ![](figures/zh-cn_image_0000002155016964.png)

-   如下示例中，srcTensor的数据类型为int32\_t，m = 4，n = 8， calCount = 4，表明srcTensor中每行进行反量化的元素个数为4；dstTensor的数据类型为float，m = 4，n\_dst = 8 \(8 \* sizeof\(float\) % 32 = 0\)。对于srcTensor每行中的前4个元素都和标量deqScale相乘并写入dstTensor中每行的对应位置。

    ![](figures/zh-cn_image_0000001819864122.png)

当用户将模板参数中的mode配置为**DEQUANT\_WITH\_SINGLE\_ROW**时：

针对DequantParams \{m, n, calCount\}， 若同时满足以下3个条件：

1.  m = 1
2.  calCount为 32 / sizeof\(dstT\)的倍数
3.  n % calCount = 0

此时 \{1, n, calCount\}会被视作为** \{n / calCount, calCount, calCount\}**  进行反量化的计算。

具体效果可看下图所示，传入的DequantParams为 \{1, 16, 8\}。因为dstT为float，所以calCount满足为8的倍数，在**DEQUANT\_WITH\_SINGLE\_ROW**模式下会将\{1, 2 \* 8, 8\}转换为 \{2, 8, 8\}进行计算。

![](figures/zh-cn_image_0000001820178290.png)

![](figures/zh-cn_image_0000001866976705.png)

## 实现原理<a name="section13229175017585"></a>

以数据类型int32\_t，shape为\[m, n\]的输入srcTensor，数据类型scaleT，shape为\[n\]的输入deqScale和数据类型dstT，shape为\[m, n\]的输出dstTensor为例，描述AscendDequant高阶API内部算法框图，如下图所示。

**图 1**  AscendDequant内部算法框图<a name="fig15964941122"></a>  
![](figures/AscendDequant内部算法框图.png "AscendDequant内部算法框图")

计算过程分为如下几步，均在Vector上进行：

1.  精度转换：将srcTensor和deqScale都转换成FP32精度的tensor，分别得到srcFP32和deqScaleFP32；
2.  Mul计算：srcFP32一共有m行，每行长度为n；通过m次循环，将srcFP32的每行与deqScaleFP32相乘，通过mask控制仅对前dequantParams.calcount个数进行mul计算，图中index的取值范围为 \[0, m\)，对应srcFP32的每一行；计算所得结果为mulRes，shape为\[m, n\]；
3.  结果数据精度转换：mulRes从FP32转换成dstT类型的tensor，所得结果为dstTensor，shape为\[m, n\]。

## 函数原型<a name="section620mcpsimp"></a>

-   反量化参数deqScale为矢量
    -   通过sharedTmpBuffer入参传入临时空间

        ```
        template <typename dstT, typename scaleT, DeQuantMode mode = DeQuantMode::DEQUANT_WITH_SINGLE_ROW>
        __aicore__ inline void AscendDequant(const LocalTensor<dstT>& dstTensor, const LocalTensor<int32_t>& srcTensor, const LocalTensor<scaleT>& deqScale, const LocalTensor<uint8_t>& sharedTmpBuffer, DequantParams params)
        ```

    -   接口框架申请临时空间

        ```
        template <typename dstT, typename scaleT, DeQuantMode mode = DeQuantMode::DEQUANT_WITH_SINGLE_ROW>
        __aicore__ inline void AscendDequant(const LocalTensor<dstT>& dstTensor, const LocalTensor<int32_t>& srcTensor, const LocalTensor<scaleT>& deqScale, DequantParams params)
        ```

-   反量化参数deqScale为标量
    -   通过sharedTmpBuffer入参传入临时空间

        ```
        template <typename dstT, typename scaleT, DeQuantMode mode = DeQuantMode::DEQUANT_WITH_SINGLE_ROW>
        __aicore__ inline void AscendDequant(const LocalTensor<dstT>& dstTensor, const LocalTensor<int32_t>& srcTensor, const scaleT deqScale, const LocalTensor<uint8_t>& sharedTmpBuffer, DequantParams params)
        ```

    -   接口框架申请临时空间

        ```
        template <typename dstT, typename scaleT, DeQuantMode mode = DeQuantMode::DEQUANT_WITH_SINGLE_ROW>
        __aicore__ inline void AscendDequant(const LocalTensor<dstT>& dstTensor, const LocalTensor<int32_t>& srcTensor, const scaleT deqScale, DequantParams params)
        ```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为sharedTmpBuffer申请空间。临时空间大小BufferSize的获取方式如下：通过[GetAscendDequantMaxMinTmpSize](GetAscendDequantMaxMinTmpSize.md)中提供的GetAscendDequantMaxMinTmpSize接口获取需要预留空间的范围大小。

以下接口不推荐使用，新开发内容不要使用如下接口：

```
template <typename dstT, typename scaleT, DeQuantMode mode = DeQuantMode::DEQUANT_WITH_SINGLE_ROW>
__aicore__ inline void AscendDequant(const LocalTensor<dstT>& dstTensor, const LocalTensor<int32_t>& srcTensor, const LocalTensor<scaleT>& deqScale, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
```

```
template <typename dstT, typename scaleT, DeQuantMode mode = DeQuantMode::DEQUANT_WITH_SINGLE_ROW>
__aicore__ inline void AscendDequant(const LocalTensor<dstT>& dstTensor, const LocalTensor<int32_t>& srcTensor, const LocalTensor<scaleT>& deqScale, const LocalTensor<uint8_t>& sharedTmpBuffer)
```

```
template <typename dstT, typename scaleT, DeQuantMode mode = DeQuantMode::DEQUANT_WITH_SINGLE_ROW>
__aicore__ inline void AscendDequant(const LocalTensor<dstT>& dstTensor, const LocalTensor<int32_t>& srcTensor, const LocalTensor<scaleT>& deqScale, const uint32_t calCount)
```

```
template <typename dstT, typename scaleT, DeQuantMode mode = DeQuantMode::DEQUANT_WITH_SINGLE_ROW>
__aicore__ inline void AscendDequant(const LocalTensor<dstT>& dstTensor, const LocalTensor<int32_t>& srcTensor, const LocalTensor<scaleT>& deqScale)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table575571914269"></a>
<table><thead align="left"><tr id="row18755131942614"><th class="cellrowborder" valign="top" width="19.39%" id="mcps1.2.3.1.1"><p id="p675519193268"><a name="p675519193268"></a><a name="p675519193268"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.61%" id="mcps1.2.3.1.2"><p id="p375511918267"><a name="p375511918267"></a><a name="p375511918267"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row14755141911264"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p47551198266"><a name="p47551198266"></a><a name="p47551198266"></a>dstT</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p125969172719"><a name="p125969172719"></a><a name="p125969172719"></a>目的操作数的数据类型。</p>
</td>
</tr>
<tr id="row6356241194912"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p143561041144915"><a name="p143561041144915"></a><a name="p143561041144915"></a>scaleT</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p4299155115268"><a name="p4299155115268"></a><a name="p4299155115268"></a>deqScale的数据类型。</p>
</td>
</tr>
<tr id="row9756719122620"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p1682112447268"><a name="p1682112447268"></a><a name="p1682112447268"></a>mode</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><div class="p" id="p7851923173111"><a name="p7851923173111"></a><a name="p7851923173111"></a>决定当DequantParams为{1, n, calCount}时的计算逻辑，传入enum DeQuantMode，支持以下 2 种配置：<a name="ul168513231318"></a><a name="ul168513231318"></a><ul id="ul168513231318"><li><strong id="b14851162310311"><a name="b14851162310311"></a><a name="b14851162310311"></a>DEQUANT_WITH_SINGLE_ROW</strong>：当DequantParams {m, n, calCount} 同时满足以下条件：1、m = 1；2、calCount为 32 / sizeof(dstT)的倍数；3、n % calCount = 0时，即 {1, n, calCount} 会当作 {n / calCount, calCount, calCount} 进行计算。</li><li><strong id="b1385182313116"><a name="b1385182313116"></a><a name="b1385182313116"></a>DEQUANT_WITH_MULTI_ROW</strong>：即使满足上述所有条件，{1, n, calCount} 依然只会当作 {1, n, calCount} 进行计算， 即总共n个数，前calCount个数进行反量化的计算。</li></ul>
</div>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table44731299481"></a>
<table><thead align="left"><tr id="row247482914489"><th class="cellrowborder" valign="top" width="15.55%" id="mcps1.2.4.1.1"><p id="p147413295483"><a name="p147413295483"></a><a name="p147413295483"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.24%" id="mcps1.2.4.1.2"><p id="p1147432994819"><a name="p1147432994819"></a><a name="p1147432994819"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.21%" id="mcps1.2.4.1.3"><p id="p74749297483"><a name="p74749297483"></a><a name="p74749297483"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row12474329104814"><td class="cellrowborder" valign="top" width="15.55%" headers="mcps1.2.4.1.1 "><p id="p1047411294482"><a name="p1047411294482"></a><a name="p1047411294482"></a>dstTensor</p>
</td>
<td class="cellrowborder" valign="top" width="9.24%" headers="mcps1.2.4.1.2 "><p id="p047412984813"><a name="p047412984813"></a><a name="p047412984813"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.21%" headers="mcps1.2.4.1.3 "><p id="p3989161814016"><a name="p3989161814016"></a><a name="p3989161814016"></a>目的操作数。类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</p>
<p id="p3343929193720"><a name="p3343929193720"></a><a name="p3343929193720"></a><span id="ph1634312291372"><a name="ph1634312291372"></a><a name="ph1634312291372"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_1"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term12835255145414_1"><a name="zh-cn_topic_0000001312391781_term12835255145414_1"></a><a name="zh-cn_topic_0000001312391781_term12835255145414_1"></a>Atlas A3 推理系列产品</term></span>，支持的数据类型为：half、bfloat16_t、float。</p>
<p id="p1647418294485"><a name="p1647418294485"></a><a name="p1647418294485"></a><span id="ph74741329164816"><a name="ph74741329164816"></a><a name="ph74741329164816"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_1"><a name="zh-cn_topic_0000001312391781_term11962195213215_1"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_1"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term1551319498507_1"><a name="zh-cn_topic_0000001312391781_term1551319498507_1"></a><a name="zh-cn_topic_0000001312391781_term1551319498507_1"></a>Atlas A2 推理系列产品</term></span>，支持的数据类型为：half、bfloat16_t、float。</p>
<a name="ul66981895213"></a><a name="ul66981895213"></a><ul id="ul66981895213"><li>dstTensor的行数和srcTensor的行数保持一致。</li><li>n * sizeof(dstT)不满足32字节对齐时，需要<strong id="b1782911052214"><a name="b1782911052214"></a><a name="b1782911052214"></a>向上补齐为32字节</strong>，n_dst为向上补齐后的列数。如srcTensor数据类型为int32_t，shape为 (4, 8)，dstTensor为bfloat16_t，则n_dst应从8补齐为16，dstTensor shape为(4, 16)。补齐的计算过程为：n_dst = (8 * sizeof(bfloat16_t) + 32 - 1) / 32 * 32 / sizeof(bfloat16_t)。</li></ul>
</td>
</tr>
<tr id="row18474729124817"><td class="cellrowborder" valign="top" width="15.55%" headers="mcps1.2.4.1.1 "><p id="p54741029164810"><a name="p54741029164810"></a><a name="p54741029164810"></a>srcTensor</p>
</td>
<td class="cellrowborder" valign="top" width="9.24%" headers="mcps1.2.4.1.2 "><p id="p144741829194814"><a name="p144741829194814"></a><a name="p144741829194814"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.21%" headers="mcps1.2.4.1.3 "><p id="p143185337247"><a name="p143185337247"></a><a name="p143185337247"></a>源操作数。类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</p>
<p id="p149295884015"><a name="p149295884015"></a><a name="p149295884015"></a><span id="ph8929208184020"><a name="ph8929208184020"></a><a name="ph8929208184020"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_2"><a name="zh-cn_topic_0000001312391781_term1253731311225_2"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_2"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term12835255145414_2"><a name="zh-cn_topic_0000001312391781_term12835255145414_2"></a><a name="zh-cn_topic_0000001312391781_term12835255145414_2"></a>Atlas A3 推理系列产品</term></span>，支持的数据类型为：int32_t。</p>
<p id="p14496195212317"><a name="p14496195212317"></a><a name="p14496195212317"></a><span id="ph104968521634"><a name="ph104968521634"></a><a name="ph104968521634"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_2"><a name="zh-cn_topic_0000001312391781_term11962195213215_2"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_2"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term1551319498507_2"><a name="zh-cn_topic_0000001312391781_term1551319498507_2"></a><a name="zh-cn_topic_0000001312391781_term1551319498507_2"></a>Atlas A2 推理系列产品</term></span>，支持的数据类型为：int32_t。</p>
<p id="p2836135311718"><a name="p2836135311718"></a><a name="p2836135311718"></a>shape为 [m, n]，n个输入数据所占字节数要求<strong id="b733515211256"><a name="b733515211256"></a><a name="b733515211256"></a>32字节对齐</strong>。</p>
</td>
</tr>
<tr id="row617218172310"><td class="cellrowborder" valign="top" width="15.55%" headers="mcps1.2.4.1.1 "><p id="p16660132211315"><a name="p16660132211315"></a><a name="p16660132211315"></a>deqScale</p>
</td>
<td class="cellrowborder" valign="top" width="9.24%" headers="mcps1.2.4.1.2 "><p id="p156601822153115"><a name="p156601822153115"></a><a name="p156601822153115"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.21%" headers="mcps1.2.4.1.3 "><p id="p12593175910168"><a name="p12593175910168"></a><a name="p12593175910168"></a>源操作数。类型为标量或者<a href="LocalTensor.md">LocalTensor</a>。类型为LocalTensor时，支持的TPosition为VECIN/VECCALC/VECOUT。</p>
<p id="p887385104019"><a name="p887385104019"></a><a name="p887385104019"></a><span id="ph4873125184020"><a name="ph4873125184020"></a><a name="ph4873125184020"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_3"><a name="zh-cn_topic_0000001312391781_term1253731311225_3"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_3"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term12835255145414_3"><a name="zh-cn_topic_0000001312391781_term12835255145414_3"></a><a name="zh-cn_topic_0000001312391781_term12835255145414_3"></a>Atlas A3 推理系列产品</term></span>，当deqScale为矢量时，支持的数据类型为：uint64_t、float、bfloat16_t；当deqScale为标量时，支持的数据类型为bfloat16_t、float。</p>
<p id="p814261713282"><a name="p814261713282"></a><a name="p814261713282"></a><span id="ph76609228311"><a name="ph76609228311"></a><a name="ph76609228311"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_3"><a name="zh-cn_topic_0000001312391781_term11962195213215_3"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_3"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term1551319498507_3"><a name="zh-cn_topic_0000001312391781_term1551319498507_3"></a><a name="zh-cn_topic_0000001312391781_term1551319498507_3"></a>Atlas A2 推理系列产品</term></span>，当deqScale为矢量时，支持的数据类型为：uint64_t、float、bfloat16_t；当deqScale为标量时，支持的数据类型为bfloat16_t、float。</p>
<p id="p11825155515364"><a name="p11825155515364"></a><a name="p11825155515364"></a>dstTensor、srcTensor、deqScale支持的数据类型组合请参考<a href="#table1963437121712">表3</a>和<a href="#table16300356102013">表4</a>。</p>
</td>
</tr>
<tr id="row1747412296483"><td class="cellrowborder" valign="top" width="15.55%" headers="mcps1.2.4.1.1 "><p id="p74741029204817"><a name="p74741029204817"></a><a name="p74741029204817"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="9.24%" headers="mcps1.2.4.1.2 "><p id="p1747452954810"><a name="p1747452954810"></a><a name="p1747452954810"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.21%" headers="mcps1.2.4.1.3 "><p id="p191160465422"><a name="p191160465422"></a><a name="p191160465422"></a>临时缓存。类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</p>
<p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>临时空间大小BufferSize的获取方式请参考<a href="GetAscendDequantMaxMinTmpSize.md">GetAscendDequantMaxMinTmpSize</a>。</p>
<p id="p3990133464118"><a name="p3990133464118"></a><a name="p3990133464118"></a><span id="ph4990133415414"><a name="ph4990133415414"></a><a name="ph4990133415414"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_4"><a name="zh-cn_topic_0000001312391781_term1253731311225_4"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_4"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term12835255145414_4"><a name="zh-cn_topic_0000001312391781_term12835255145414_4"></a><a name="zh-cn_topic_0000001312391781_term12835255145414_4"></a>Atlas A3 推理系列产品</term></span>，支持的数据类型为：uint8_t。</p>
<p id="p372173815911"><a name="p372173815911"></a><a name="p372173815911"></a><span id="ph173638145913"><a name="ph173638145913"></a><a name="ph173638145913"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_4"><a name="zh-cn_topic_0000001312391781_term11962195213215_4"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_4"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term1551319498507_4"><a name="zh-cn_topic_0000001312391781_term1551319498507_4"></a><a name="zh-cn_topic_0000001312391781_term1551319498507_4"></a>Atlas A2 推理系列产品</term></span>，支持的数据类型为：uint8_t。</p>
</td>
</tr>
<tr id="row850382835820"><td class="cellrowborder" valign="top" width="15.55%" headers="mcps1.2.4.1.1 "><p id="p714563310589"><a name="p714563310589"></a><a name="p714563310589"></a>params</p>
</td>
<td class="cellrowborder" valign="top" width="9.24%" headers="mcps1.2.4.1.2 "><p id="p210914395588"><a name="p210914395588"></a><a name="p210914395588"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.21%" headers="mcps1.2.4.1.3 "><p id="p49691232086"><a name="p49691232086"></a><a name="p49691232086"></a>srcTensor的shape信息。DequantParams类型，具体定义如下：</p>
<a name="screen641172125916"></a><a name="screen641172125916"></a><pre class="screen" codetype="Cpp" id="screen641172125916">struct DequantParams
{
    uint32_t m;             // srcTensor的行数
    uint32_t n;             // srcTensor的列数
    uint32_t calCount;      // 针对srcTensor每一行，前calCount个数为有效数据，与deqScale的前calCount个数或者deqScale标量进行乘法计算
};</pre>
<a name="ul19603810111110"></a><a name="ul19603810111110"></a><ul id="ul19603810111110"><li>DequantParams.n * sizeof(T)必须是32字节的整数倍，T为srcTensor中元素的数据类型。</li><li>因为是每n个数中的前calCount个数进行乘法运算，因此DequantParams.n和calCount需要满足以下关系<p id="p10782229968"><a name="p10782229968"></a><a name="p10782229968"></a>1 &lt;= DequantParams.calCount &lt;= DequantParams.n。</p>
</li><li>deqScale为矢量时，DequantParams.calCount &lt;= deqScale的元素个数。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 3**  支持的数据类型组合（deqScale为LocalTensor）

<a name="table1963437121712"></a>
<table><thead align="left"><tr id="row16963183711175"><th class="cellrowborder" valign="top" width="30.05300530053005%" id="mcps1.2.4.1.1"><p id="p1296310378174"><a name="p1296310378174"></a><a name="p1296310378174"></a>dstTensor</p>
</th>
<th class="cellrowborder" valign="top" width="31.773177317731772%" id="mcps1.2.4.1.2"><p id="p3963237121719"><a name="p3963237121719"></a><a name="p3963237121719"></a>srcTensor</p>
</th>
<th class="cellrowborder" valign="top" width="38.173817381738175%" id="mcps1.2.4.1.3"><p id="p596323771711"><a name="p596323771711"></a><a name="p596323771711"></a>deqScale</p>
</th>
</tr>
</thead>
<tbody><tr id="row19963203771713"><td class="cellrowborder" valign="top" width="30.05300530053005%" headers="mcps1.2.4.1.1 "><p id="p8964203741720"><a name="p8964203741720"></a><a name="p8964203741720"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="31.773177317731772%" headers="mcps1.2.4.1.2 "><p id="p896463713171"><a name="p896463713171"></a><a name="p896463713171"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" width="38.173817381738175%" headers="mcps1.2.4.1.3 "><p id="p496411375175"><a name="p496411375175"></a><a name="p496411375175"></a>uint64_t</p>
<p id="p0792388399"><a name="p0792388399"></a><a name="p0792388399"></a>注意：当deqScale的数据类型是uint64_t时，数值低32位是参与计算的数据，数据类型是float，数值高32位是一些控制参数，本接口不使用。</p>
</td>
</tr>
<tr id="row14964537111719"><td class="cellrowborder" valign="top" width="30.05300530053005%" headers="mcps1.2.4.1.1 "><p id="p16964143711712"><a name="p16964143711712"></a><a name="p16964143711712"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="31.773177317731772%" headers="mcps1.2.4.1.2 "><p id="p125026399197"><a name="p125026399197"></a><a name="p125026399197"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" width="38.173817381738175%" headers="mcps1.2.4.1.3 "><p id="p8964153701717"><a name="p8964153701717"></a><a name="p8964153701717"></a>float</p>
</td>
</tr>
<tr id="row996423751717"><td class="cellrowborder" valign="top" width="30.05300530053005%" headers="mcps1.2.4.1.1 "><p id="p1396463781710"><a name="p1396463781710"></a><a name="p1396463781710"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="31.773177317731772%" headers="mcps1.2.4.1.2 "><p id="p16204019195"><a name="p16204019195"></a><a name="p16204019195"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" width="38.173817381738175%" headers="mcps1.2.4.1.3 "><p id="p10964163715171"><a name="p10964163715171"></a><a name="p10964163715171"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row1196413375174"><td class="cellrowborder" valign="top" width="30.05300530053005%" headers="mcps1.2.4.1.1 "><p id="p7964133731719"><a name="p7964133731719"></a><a name="p7964133731719"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="31.773177317731772%" headers="mcps1.2.4.1.2 "><p id="p253912402195"><a name="p253912402195"></a><a name="p253912402195"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" width="38.173817381738175%" headers="mcps1.2.4.1.3 "><p id="p596414373170"><a name="p596414373170"></a><a name="p596414373170"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row361702025017"><td class="cellrowborder" valign="top" width="30.05300530053005%" headers="mcps1.2.4.1.1 "><p id="p16178207509"><a name="p16178207509"></a><a name="p16178207509"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="31.773177317731772%" headers="mcps1.2.4.1.2 "><p id="p4617202014509"><a name="p4617202014509"></a><a name="p4617202014509"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" width="38.173817381738175%" headers="mcps1.2.4.1.3 "><p id="p1161712085012"><a name="p1161712085012"></a><a name="p1161712085012"></a>float</p>
</td>
</tr>
</tbody>
</table>

**表 4**  支持的数据类型组合（deqScale为标量）

<a name="table16300356102013"></a>
<table><thead align="left"><tr id="row930015616207"><th class="cellrowborder" valign="top" width="29.982998299829983%" id="mcps1.2.4.1.1"><p id="p12300135632011"><a name="p12300135632011"></a><a name="p12300135632011"></a>dstTensor</p>
</th>
<th class="cellrowborder" valign="top" width="31.623162316231625%" id="mcps1.2.4.1.2"><p id="p430065632012"><a name="p430065632012"></a><a name="p430065632012"></a>srcTensor</p>
</th>
<th class="cellrowborder" valign="top" width="38.39383938393839%" id="mcps1.2.4.1.3"><p id="p53001569204"><a name="p53001569204"></a><a name="p53001569204"></a>deqScale</p>
</th>
</tr>
</thead>
<tbody><tr id="row9300145642013"><td class="cellrowborder" valign="top" width="29.982998299829983%" headers="mcps1.2.4.1.1 "><p id="p15300135615203"><a name="p15300135615203"></a><a name="p15300135615203"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="31.623162316231625%" headers="mcps1.2.4.1.2 "><p id="p10300356132010"><a name="p10300356132010"></a><a name="p10300356132010"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" width="38.39383938393839%" headers="mcps1.2.4.1.3 "><p id="p3300155662012"><a name="p3300155662012"></a><a name="p3300155662012"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row114049495503"><td class="cellrowborder" valign="top" width="29.982998299829983%" headers="mcps1.2.4.1.1 "><p id="p204049499508"><a name="p204049499508"></a><a name="p204049499508"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="31.623162316231625%" headers="mcps1.2.4.1.2 "><p id="p74041049145010"><a name="p74041049145010"></a><a name="p74041049145010"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" width="38.39383938393839%" headers="mcps1.2.4.1.3 "><p id="p12404149165018"><a name="p12404149165018"></a><a name="p12404149165018"></a>float</p>
</td>
</tr>
<tr id="row1216518171715"><td class="cellrowborder" valign="top" width="29.982998299829983%" headers="mcps1.2.4.1.1 "><p id="p236646111720"><a name="p236646111720"></a><a name="p236646111720"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="31.623162316231625%" headers="mcps1.2.4.1.2 "><p id="p193661968173"><a name="p193661968173"></a><a name="p193661968173"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" width="38.39383938393839%" headers="mcps1.2.4.1.3 "><p id="p43669631716"><a name="p43669631716"></a><a name="p43669631716"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row335551062319"><td class="cellrowborder" valign="top" width="29.982998299829983%" headers="mcps1.2.4.1.1 "><p id="p03551010192316"><a name="p03551010192316"></a><a name="p03551010192316"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="31.623162316231625%" headers="mcps1.2.4.1.2 "><p id="p635510107236"><a name="p635510107236"></a><a name="p635510107236"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" width="38.39383938393839%" headers="mcps1.2.4.1.3 "><p id="p418282714233"><a name="p418282714233"></a><a name="p418282714233"></a>float</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section38228281712"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   **不支持源操作数与目的操作数地址重叠。**
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

## 调用示例<a name="section642mcpsimp"></a>

```
rowLen = m;                 // m = 4
colLen = n;                 // n = 8
//输入srcLocal的shape为4*8，类型为int32_t，deqScaleLocal的shape为8，类型为float，预留临时空间
AscendC::AscendDequant(dstLocal, srcLocal, deqScaleLocal, {rowLen, colLen, deqScaleLocal.GetSize()});
```

结果示例如下：

```
输入数据(srcLocal) int32_t数据类型:
[ -8  5 -5 -7 -3 -8  3  6
   9  2 -5  0  0 -5 -7  0 
  -6  0 -2  3 -2 8   5  2 
   2  2 -4  5 -4  4 -8  3 ]

反量化参数deqScale float数据类型:  
[ 10.433567  10.765296   -30.694275   -65.47741    8.386527    -89.646194   65.11153    42.213394]

输出数据(dstLocal) float数据类型:  
[-83.46854      53.82648    153.47137    458.34186    -25.15958   717.16956    195.33458   253.28036 
 93.9021        21.530592   153.47137    -0.          0.          448.23096    -455.7807   0.    
 -62.601402     0.          61.38855     -196.43222   -16.773054  -717.16956   325.55762   84.42679 
 20.867134      21.530592   122.7771     -327.38705   -33.54611   -358.58478   -520.8922   126.64018 ]
```

