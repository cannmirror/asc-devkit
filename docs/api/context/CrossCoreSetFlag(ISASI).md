# CrossCoreSetFlag\(ISASI\)<a name="ZH-CN_TOPIC_0000001834069637"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p48327011813"><a name="p48327011813"></a><a name="p48327011813"></a><span id="ph583230201815"><a name="ph583230201815"></a><a name="ph583230201815"></a><term id="zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term12835255145414"><a name="zh-cn_topic_0000001312391781_term12835255145414"></a><a name="zh-cn_topic_0000001312391781_term12835255145414"></a>Atlas A3 推理系列产品</term></span></p>
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

面向分离模式的核间同步控制接口。

该接口和[CrossCoreWaitFlag](CrossCoreWaitFlag(ISASI).md)接口配合使用。使用时需传入核间同步的标记ID\(flagId\)，每个ID对应一个初始值为0的计数器。执行CrossCoreSetFlag后ID对应的计数器增加1；执行CrossCoreWaitFlag时如果对应的计数器数值为0则阻塞不执行；如果对应的计数器大于0，则计数器减一，同时后续指令开始执行。

同步控制分为以下几种模式，如[图1](#fig37581010773)所示：

-   模式0：AI Core核间的同步控制。对于AIC场景，同步所有的AIC核，直到所有的AIC核都执行到CrossCoreSetFlag时，CrossCoreWaitFlag后续的指令才会执行；对于AIV场景，同步所有的AIV核，直到所有的AIV核都执行到CrossCoreSetFlag时，CrossCoreWaitFlag后续的指令才会执行。
-   模式1：AI Core内部，AIV核之间的同步控制。如果两个AIV核都运行了CrossCoreSetFlag，CrossCoreWaitFlag后续的指令才会执行。
-   模式2：AI Core内部，AIC与AIV之间的同步控制。在AIC核执行CrossCoreSetFlag之后， 两个AIV上CrossCoreWaitFlag后续的指令才会继续执行；两个AIV都执行CrossCoreSetFlag后，AIC上CrossCoreWaitFlag后续的指令才能执行。

**图 1**  同步控制模式示意图<a name="fig37581010773"></a>  
![](figures/同步控制模式示意图.png "同步控制模式示意图")

## 函数原型<a name="section620mcpsimp"></a>

```
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="18.8%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.2%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="18.8%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>modeId</p>
</td>
<td class="cellrowborder" valign="top" width="81.2%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>核间同步的模式，取值如下：</p>
<a name="ul335269516"></a><a name="ul335269516"></a><ul id="ul335269516"><li>模式0：AI Core核间的同步控制。</li><li>模式1：AI Core内部，Vector核（AIV）之间的同步控制。</li><li>模式2：AI Core内部，Cube核（AIC）与Vector核（AIV）之间的同步控制。</li></ul>
</td>
</tr>
<tr id="row168561422132317"><td class="cellrowborder" valign="top" width="18.8%" headers="mcps1.2.3.1.1 "><p id="p1162623072316"><a name="p1162623072316"></a><a name="p1162623072316"></a>pipe</p>
</td>
<td class="cellrowborder" valign="top" width="81.2%" headers="mcps1.2.3.1.2 "><p id="p13626330102311"><a name="p13626330102311"></a><a name="p13626330102311"></a>设置这条指令所在的流水类型，流水类型可参考<a href="同步控制简介.md#section1272612276459">硬件流水类型</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row2137145181815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p179035252218"><a name="p179035252218"></a><a name="p179035252218"></a>flagId</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p7789185214226"><a name="p7789185214226"></a><a name="p7789185214226"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p10993195419912"><a name="p10993195419912"></a><a name="p10993195419912"></a>核间同步的标记。</p>
<p id="p3301156999"><a name="p3301156999"></a><a name="p3301156999"></a><span id="ph1930145610914"><a name="ph1930145610914"></a><a name="ph1930145610914"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_1"><a name="zh-cn_topic_0000001312391781_term11962195213215_1"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_1"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term1551319498507_1"><a name="zh-cn_topic_0000001312391781_term1551319498507_1"></a><a name="zh-cn_topic_0000001312391781_term1551319498507_1"></a>Atlas A2 推理系列产品</term></span>，取值范围是0-10。</p>
<p id="p19301456896"><a name="p19301456896"></a><a name="p19301456896"></a><span id="ph14307561397"><a name="ph14307561397"></a><a name="ph14307561397"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_1"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term12835255145414_1"><a name="zh-cn_topic_0000001312391781_term12835255145414_1"></a><a name="zh-cn_topic_0000001312391781_term12835255145414_1"></a>Atlas A3 推理系列产品</term></span>，取值范围是0-10。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   使用该同步接口时，需要按照如下规则[设置Kernel类型](设置Kernel类型.md)：
    -   在纯Vector/Cube场景下，需设置Kernel类型为KERNEL\_TYPE\_MIX\_AIV\_1\_0或KERNEL\_TYPE\_MIX\_AIC\_1\_0。
    -   对于Vector和Cube混合场景，需根据实际情况灵活配置Kernel类型。

-   因为[Matmul高阶API](矩阵计算-28.md)内部实现中使用了本接口进行核间同步控制，所以不建议开发者同时使用该接口和Matmul高阶API，否则会有flagID冲突的风险。
-   同一flagId的计数器最多设置15次。

## 调用示例<a name="section837496171220"></a>

```
// 使用模式0的方式同步所有的AIV核
if (g_coreType == AscendC::AIV) {
    AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(0x8);
    AscendC::CrossCoreWaitFlag(0x8);
}

// 使用模式1的方式同步当前AICore内的所有AIV子核
if (g_coreType == AscendC::AIV) {
    AscendC::CrossCoreSetFlag<0x1, PIPE_MTE3>(0x8);
    AscendC::CrossCoreWaitFlag(0x8);
}

// 注意：如果调用高阶API,无需开发者处理AIC和AIV的同步
// AIC侧做完Matmul计算后通知AIV进行后处理
if (g_coreType == AscendC::AIC) {
    // Matmul处理
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);
}

// AIV侧等待AIC Set消息, 进行Vector后处理
if (g_coreType == AscendC::AIV) {
    AscendC::CrossCoreWaitFlag(0x8);
    // Vector后处理
}
```

