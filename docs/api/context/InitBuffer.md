# InitBuffer<a name="ZH-CN_TOPIC_0000001375937722"></a>

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

用于为TQue等队列和TBuf分配内存。

## 函数原型<a name="section620mcpsimp"></a>

-   为TQue等队列分配内存

    ```
    template <class T>
    __aicore__ inline bool InitBuffer(T& que, uint8_t num, uint32_t len)
    ```

-   为TBuf分配内存

    ```
    template <TPosition bufPos>
    __aicore__ inline bool InitBuffer(TBuf<bufPos>& buf, uint32_t len)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  bool InitBuffer\(T& que, uint8\_t num, uint32\_t len\) 原型定义模板参数说明

<a name="table634418773417"></a>
<table><thead align="left"><tr id="row1934537133415"><th class="cellrowborder" valign="top" width="12.13%" id="mcps1.2.3.1.1"><p id="p1934597103419"><a name="p1934597103419"></a><a name="p1934597103419"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="87.87%" id="mcps1.2.3.1.2"><p id="p1034519793416"><a name="p1034519793416"></a><a name="p1034519793416"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row14345472346"><td class="cellrowborder" valign="top" width="12.13%" headers="mcps1.2.3.1.1 "><p id="p149121233183416"><a name="p149121233183416"></a><a name="p149121233183416"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="87.87%" headers="mcps1.2.3.1.2 "><p id="p1034577203412"><a name="p1034577203412"></a><a name="p1034577203412"></a>队列的类型，支持取值<a href="TQue.md">TQue</a>、<a href="TQueBind.md">TQueBind</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  bool InitBuffer\(T& que, uint8\_t num, uint32\_t len\) 原型定义参数说明

<a name="table193329316393"></a>
<table><thead align="left"><tr id="row123331131153919"><th class="cellrowborder" valign="top" width="11.940000000000001%" id="mcps1.2.4.1.1"><p id="p8333133153913"><a name="p8333133153913"></a><a name="p8333133153913"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.8%" id="mcps1.2.4.1.2"><p id="p518118718459"><a name="p518118718459"></a><a name="p518118718459"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.26%" id="mcps1.2.4.1.3"><p id="p833353113393"><a name="p833353113393"></a><a name="p833353113393"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row11660173845017"><td class="cellrowborder" valign="top" width="11.940000000000001%" headers="mcps1.2.4.1.1 "><p id="p466053810507"><a name="p466053810507"></a><a name="p466053810507"></a>que</p>
</td>
<td class="cellrowborder" valign="top" width="12.8%" headers="mcps1.2.4.1.2 "><p id="p885774605014"><a name="p885774605014"></a><a name="p885774605014"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.26%" headers="mcps1.2.4.1.3 "><p id="p0660153818501"><a name="p0660153818501"></a><a name="p0660153818501"></a>需要分配内存的TQue等对象。</p>
</td>
</tr>
<tr id="row03336319398"><td class="cellrowborder" valign="top" width="11.940000000000001%" headers="mcps1.2.4.1.1 "><p id="p11399116193313"><a name="p11399116193313"></a><a name="p11399116193313"></a>num</p>
</td>
<td class="cellrowborder" valign="top" width="12.8%" headers="mcps1.2.4.1.2 "><p id="p111819774511"><a name="p111819774511"></a><a name="p111819774511"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.26%" headers="mcps1.2.4.1.3 "><p id="p6383173514333"><a name="p6383173514333"></a><a name="p6383173514333"></a>分配内存块的个数。double buffer功能通过该参数开启：num设置为1，表示不开启double buffer；num设置为2，表示开启double buffer。</p>
</td>
</tr>
<tr id="row1430772593316"><td class="cellrowborder" valign="top" width="11.940000000000001%" headers="mcps1.2.4.1.1 "><p id="p530752514330"><a name="p530752514330"></a><a name="p530752514330"></a>len</p>
</td>
<td class="cellrowborder" valign="top" width="12.8%" headers="mcps1.2.4.1.2 "><p id="p530711252335"><a name="p530711252335"></a><a name="p530711252335"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.26%" headers="mcps1.2.4.1.3 "><p id="p183081251336"><a name="p183081251336"></a><a name="p183081251336"></a>每个内存块的大小，单位为字节。当传入的len不满足32字节对齐时，API内部会自动向上补齐至32字节对齐，后续的数据搬运过程会涉及非对齐处理。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  InitBuffer\(TBuf<bufPos\>& buf, uint32\_t len\)原型定义模板参数说明

<a name="table873615294112"></a>
<table><thead align="left"><tr id="row473742114115"><th class="cellrowborder" valign="top" width="12.34%" id="mcps1.2.3.1.1"><p id="p1073720204115"><a name="p1073720204115"></a><a name="p1073720204115"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="87.66000000000001%" id="mcps1.2.3.1.2"><p id="p37371826412"><a name="p37371826412"></a><a name="p37371826412"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row167376215416"><td class="cellrowborder" valign="top" width="12.34%" headers="mcps1.2.3.1.1 "><p id="p14568152324113"><a name="p14568152324113"></a><a name="p14568152324113"></a>bufPos</p>
</td>
<td class="cellrowborder" valign="top" width="87.66000000000001%" headers="mcps1.2.3.1.2 "><p id="p188271137124113"><a name="p188271137124113"></a><a name="p188271137124113"></a>TBuf所在的逻辑位置，<a href="TPosition.md">TPosition</a>类型。</p>
</td>
</tr>
</tbody>
</table>

**表 4**  InitBuffer\(TBuf<bufPos\>& buf, uint32\_t len\)原型定义参数说明

<a name="table5376122715308"></a>
<table><thead align="left"><tr id="row1337716275309"><th class="cellrowborder" valign="top" width="12.36%" id="mcps1.2.4.1.1"><p id="p1537762711305"><a name="p1537762711305"></a><a name="p1537762711305"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.370000000000001%" id="mcps1.2.4.1.2"><p id="p153771127123013"><a name="p153771127123013"></a><a name="p153771127123013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.27000000000001%" id="mcps1.2.4.1.3"><p id="p17377162715303"><a name="p17377162715303"></a><a name="p17377162715303"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row19377627133012"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p737710279307"><a name="p737710279307"></a><a name="p737710279307"></a>buf</p>
</td>
<td class="cellrowborder" valign="top" width="12.370000000000001%" headers="mcps1.2.4.1.2 "><p id="p13377122733010"><a name="p13377122733010"></a><a name="p13377122733010"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.27000000000001%" headers="mcps1.2.4.1.3 "><p id="p19377102793016"><a name="p19377102793016"></a><a name="p19377102793016"></a>需要分配内存的TBuf对象。</p>
</td>
</tr>
<tr id="row13377162793019"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p5377527113018"><a name="p5377527113018"></a><a name="p5377527113018"></a>len</p>
</td>
<td class="cellrowborder" valign="top" width="12.370000000000001%" headers="mcps1.2.4.1.2 "><p id="p12377122712304"><a name="p12377122712304"></a><a name="p12377122712304"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.27000000000001%" headers="mcps1.2.4.1.3 "><p id="p6514716314"><a name="p6514716314"></a><a name="p6514716314"></a>为TBuf分配的内存大小，单位为字节。当传入的len不满足32字节对齐时，API内部会自动向上补齐至32字节对齐，后续的数据搬运过程会涉及非对齐处理。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   InitBuffer申请的内存会在TPipe对象销毁时通过析构函数自动释放，无需手动释放。
-   如果需要重新分配InitBuffer申请的内存，可以调用[Reset](Reset.md)，再调用InitBuffer接口。
-   一个kernel中所有使用的Buffer数量之和不能超过64。

## 返回值说明<a name="section640mcpsimp"></a>

返回Buffer初始化的结果。

## 调用示例<a name="section642mcpsimp"></a>

```
// 为TQue分配内存，分配内存块数为2，每块大小为128字节
AscendC::TPipe pipe; // Pipe内存管理对象
AscendC::TQue<AscendC::TPosition::VECOUT, 2> que; // 输出数据队列管理对象，TPosition为VECOUT
uint8_t num = 2;
uint32_t len = 128;
pipe.InitBuffer(que, num, len);

// 为TBuf分配内存，分配长度为128字节
AscendC::TPipe pipe;
AscendC::TBuf<AscendC::TPosition::A1> buf; // 输出数据管理对象，TPosition为A1
uint32_t len = 128;
pipe.InitBuffer(buf, len);
```

