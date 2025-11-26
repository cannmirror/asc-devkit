# Load2D<a name="ZH-CN_TOPIC_0000002510751549"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row15984141454611"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 ">&nbsp;</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p8984151413464"><a name="p8984151413464"></a><a name="p8984151413464"></a><span id="ph344042535512"><a name="ph344042535512"></a><a name="ph344042535512"></a>√</span></p>
</td>
</tr>
<tr id="row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p48327011813"><a name="p48327011813"></a><a name="p48327011813"></a><span id="ph583230201815"><a name="ph583230201815"></a><a name="ph583230201815"></a><term id="zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term12835255145414"><a name="zh-cn_topic_0000001312391781_term12835255145414"></a><a name="zh-cn_topic_0000001312391781_term12835255145414"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p7948163910184"><a name="p7948163910184"></a><a name="p7948163910184"></a>√</p>
</td>
</tr>
<tr id="row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p14832120181815"><a name="p14832120181815"></a><a name="p14832120181815"></a><span id="ph1483216010188"><a name="ph1483216010188"></a><a name="ph1483216010188"></a><term id="zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term1551319498507"><a name="zh-cn_topic_0000001312391781_term1551319498507"></a><a name="zh-cn_topic_0000001312391781_term1551319498507"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p19948143911820"><a name="p19948143911820"></a><a name="p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

Load2D支持如下数据通路的搬运：

GM-\>A1; GM-\>B1; GM-\>A2; GM-\>B2;

A1-\>A2; B1-\>B2。

源操作数/目的操作数的数据类型为uint8\_t/int8\_t时，分形矩阵大小在A1/A2上为16\*32， 在B1/B2上为32\*16。

源操作数/目的操作数的数据类型为uint16\_t/int16\_t/half/bfloat16\_t时，分形矩阵在A1/B1/A2/B2上的大小为16\*16。

源操作数/目的操作数的数据类型为uint32\_t/int32\_t/float时，分形矩阵大小在A1/A2上为16\*8， 在B1/B2上为8\*16。

## 函数原型<a name="section620mcpsimp"></a>

-   Load2D接口

    ```
    template <typename T>
    void LoadData(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LoadData2DParams& loadDataParams)
    template <typename T> 
    void LoadData(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const LoadData2DParams& loadDataParams)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table07381635103112"></a>
<table><thead align="left"><tr id="row117393350314"><th class="cellrowborder" valign="top" width="16.55%" id="mcps1.2.3.1.1"><p id="p14739335193119"><a name="p14739335193119"></a><a name="p14739335193119"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="83.45%" id="mcps1.2.3.1.2"><p id="p8739203514314"><a name="p8739203514314"></a><a name="p8739203514314"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row18739935193119"><td class="cellrowborder" valign="top" width="16.55%" headers="mcps1.2.3.1.1 "><p id="p173953516310"><a name="p173953516310"></a><a name="p173953516310"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="83.45%" headers="mcps1.2.3.1.2 "><p id="p12739193516313"><a name="p12739193516313"></a><a name="p12739193516313"></a>源操作数和目的操作数的数据类型。</p>
<a name="ul773801764017"></a><a name="ul773801764017"></a><ul id="ul773801764017"><li><strong id="b183431082710"><a name="b183431082710"></a><a name="b183431082710"></a>Load2D接口</strong><p id="p11966920164019"><a name="p11966920164019"></a><a name="p11966920164019"></a><span id="ph126684217571"><a name="ph126684217571"></a><a name="ph126684217571"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_1"><a name="zh-cn_topic_0000001312391781_term11962195213215_1"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_1"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term1551319498507_1"><a name="zh-cn_topic_0000001312391781_term1551319498507_1"></a><a name="zh-cn_topic_0000001312391781_term1551319498507_1"></a>Atlas A2 推理系列产品</term></span>，支持数据类型为：uint8_t/int8_t/uint16_t/int16_t/half/bfloat16_t/uint32_t/int32_t/float</p>
<p id="p3966720184016"><a name="p3966720184016"></a><a name="p3966720184016"></a><span id="ph16239174011416"><a name="ph16239174011416"></a><a name="ph16239174011416"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_1"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term12835255145414_1"><a name="zh-cn_topic_0000001312391781_term12835255145414_1"></a><a name="zh-cn_topic_0000001312391781_term12835255145414_1"></a>Atlas A3 推理系列产品</term></span>，支持数据类型为：uint8_t/int8_t/uint16_t/int16_t/half/bfloat16_t/uint32_t/int32_t/float</p>
</li></ul>
</td>
</tr>
<tr id="row14580104484717"><td class="cellrowborder" valign="top" width="16.55%" headers="mcps1.2.3.1.1 "><p id="p127976994513"><a name="p127976994513"></a><a name="p127976994513"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="83.45%" headers="mcps1.2.3.1.2 "><a name="ul062813586431"></a><a name="ul062813586431"></a>
</td>
</tr>
<tr id="row09851744154415"><td class="cellrowborder" valign="top" width="16.55%" headers="mcps1.2.3.1.1 "><p id="p11985204464419"><a name="p11985204464419"></a><a name="p11985204464419"></a><span id="ph6814456194414"><a name="ph6814456194414"></a><a name="ph6814456194414"></a>Src</span></p>
</td>
<td class="cellrowborder" valign="top" width="83.45%" headers="mcps1.2.3.1.2 ">&nbsp;</td>
</tr>
<tr id="row1130353815467"><td class="cellrowborder" valign="top" width="16.55%" headers="mcps1.2.3.1.1 "><p id="p1230363812467"><a name="p1230363812467"></a><a name="p1230363812467"></a><span id="ph8514741194616"><a name="ph8514741194616"></a><a name="ph8514741194616"></a>Dst</span></p>
</td>
<td class="cellrowborder" valign="top" width="83.45%" headers="mcps1.2.3.1.2 ">&nbsp;</td>
</tr>
</tbody>
</table>

**表 2**  通用参数说明

<a name="table18368155193919"></a>
<table><thead align="left"><tr id="row1036805543911"><th class="cellrowborder" valign="top" width="16.89168916891689%" id="mcps1.2.4.1.1"><p id="p1836835511393"><a name="p1836835511393"></a><a name="p1836835511393"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="11.111111111111112%" id="mcps1.2.4.1.2"><p id="p10368255163915"><a name="p10368255163915"></a><a name="p10368255163915"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.99719971997199%" id="mcps1.2.4.1.3"><p id="p436875573911"><a name="p436875573911"></a><a name="p436875573911"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1436825518395"><td class="cellrowborder" valign="top" width="16.89168916891689%" headers="mcps1.2.4.1.1 "><p id="p9649151061720"><a name="p9649151061720"></a><a name="p9649151061720"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="11.111111111111112%" headers="mcps1.2.4.1.2 "><p id="p1649121041718"><a name="p1649121041718"></a><a name="p1649121041718"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.99719971997199%" headers="mcps1.2.4.1.3 "><p id="p66101129164712"><a name="p66101129164712"></a><a name="p66101129164712"></a>目的操作数，类型为LocalTensor。</p>
<p id="p3610102994715"><a name="p3610102994715"></a><a name="p3610102994715"></a>数据连续排列顺序由目的操作数所在TPosition决定，具体约束如下：</p>
<a name="ul76107290479"></a><a name="ul76107290479"></a><ul id="ul76107290479"><li>A2：ZZ格式；</li><li>B2：ZN格式；</li><li>A1/B1：无格式要求，一般情况下为NZ格式。</li></ul>
</td>
</tr>
<tr id="row1836875519393"><td class="cellrowborder" valign="top" width="16.89168916891689%" headers="mcps1.2.4.1.1 "><p id="p7650141019171"><a name="p7650141019171"></a><a name="p7650141019171"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="11.111111111111112%" headers="mcps1.2.4.1.2 "><p id="p4650610141715"><a name="p4650610141715"></a><a name="p4650610141715"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.99719971997199%" headers="mcps1.2.4.1.3 "><p id="p165920401156"><a name="p165920401156"></a><a name="p165920401156"></a>源操作数，类型为LocalTensor或GlobalTensor。</p>
<p id="p192019400435"><a name="p192019400435"></a><a name="p192019400435"></a>数据类型需要与dst保持一致。</p>
</td>
</tr>
<tr id="row1767431631917"><td class="cellrowborder" valign="top" width="16.89168916891689%" headers="mcps1.2.4.1.1 "><p id="p667418162198"><a name="p667418162198"></a><a name="p667418162198"></a>loadDataParams</p>
</td>
<td class="cellrowborder" valign="top" width="11.111111111111112%" headers="mcps1.2.4.1.2 "><p id="p11675191610195"><a name="p11675191610195"></a><a name="p11675191610195"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.99719971997199%" headers="mcps1.2.4.1.3 "><p id="p1667541617193"><a name="p1667541617193"></a><a name="p1667541617193"></a>LoadData参数结构体，类型为：</p>
<a name="ul207951119112217"></a><a name="ul207951119112217"></a><ul id="ul207951119112217"><li>LoadData2DParams，具体参考<a href="#table8955841508">表3</a>。</li></ul>
<p id="p21811725744"><a name="p21811725744"></a><a name="p21811725744"></a>上述结构体参数定义请参考<span id="ph10562197165916"><a name="ph10562197165916"></a><a name="ph10562197165916"></a>${INSTALL_DIR}</span>/include/ascendc/basic_api/interface/kernel_struct_mm.h，<span id="ph14322531015"><a name="ph14322531015"></a><a name="ph14322531015"></a>${INSTALL_DIR}</span>请替换为CANN软件安装后文件存储路径。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  LoadData2DParams结构体内参数说明

<a name="table8955841508"></a>
<table><thead align="left"><tr id="row15956194105014"><th class="cellrowborder" valign="top" width="18.56%" id="mcps1.2.3.1.1"><p id="p7956144195014"><a name="p7956144195014"></a><a name="p7956144195014"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="81.44%" id="mcps1.2.3.1.2"><p id="p16956144145011"><a name="p16956144145011"></a><a name="p16956144145011"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row5956546509"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p1855384918180"><a name="p1855384918180"></a><a name="p1855384918180"></a>startIndex</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p35535499185"><a name="p35535499185"></a><a name="p35535499185"></a>分形矩阵ID，说明搬运起始位置为源操作数中第几个分形（0为源操作数中第1个分形矩阵）。取值范围：startIndex∈[0, 65535] 。单位：512B。默认为0。</p>
</td>
</tr>
<tr id="row4956154125018"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p955315493182"><a name="p955315493182"></a><a name="p955315493182"></a>repeatTimes</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p15553949101816"><a name="p15553949101816"></a><a name="p15553949101816"></a>迭代次数，每个迭代可以处理512B数据。取值范围：repeatTimes∈[1, 255]。</p>
</td>
</tr>
<tr id="row11771625161812"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p055374920185"><a name="p055374920185"></a><a name="p055374920185"></a>srcStride</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p18553154931812"><a name="p18553154931812"></a><a name="p18553154931812"></a>相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔，单位：512B。取值范围：src_stride∈[0, 65535]。默认为0。</p>
</td>
</tr>
<tr id="row93311227171815"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p1555320499185"><a name="p1555320499185"></a><a name="p1555320499185"></a>sid</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p1755334901812"><a name="p1755334901812"></a><a name="p1755334901812"></a>预留参数，配置为0即可。</p>
</td>
</tr>
<tr id="row1321772919185"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p125531449181816"><a name="p125531449181816"></a><a name="p125531449181816"></a>dstGap</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p1755412492183"><a name="p1755412492183"></a><a name="p1755412492183"></a>相邻迭代间，目的操作数前一个分形结束地址与后一个分形起始地址的间隔，单位：512B。取值范围：dstGap∈[0, 65535]。默认为0。</p>
</td>
</tr>
<tr id="row16697631171819"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p1555415492180"><a name="p1555415492180"></a><a name="p1555415492180"></a>ifTranspose</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p1955414941814"><a name="p1955414941814"></a><a name="p1955414941814"></a>是否启用转置功能，对每个分形矩阵进行转置，默认为false:</p>
<a name="ul7554154914184"></a><a name="ul7554154914184"></a><ul id="ul7554154914184"><li>true：启用</li><li>false：不启用</li></ul>
<p id="p15554134918181"><a name="p15554134918181"></a><a name="p15554134918181"></a>注意：只有A1-&gt;A2和B1-&gt;B2通路才能使能转置，使能转置功能时，源操作数、目的操作数仅支持uint16_t/int16_t/half数据类型。</p>
</td>
</tr>
<tr id="row497391184"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p16554134921816"><a name="p16554134921816"></a><a name="p16554134921816"></a>addrMode</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p155541749141820"><a name="p155541749141820"></a><a name="p155541749141820"></a>预留参数，配置为0即可。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

## 返回值说明<a name="section640mcpsimp"></a>

无

