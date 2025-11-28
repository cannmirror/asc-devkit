# SetReduceType<a name="ZH-CN_TOPIC_0000002120820098"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置Reduce操作类型，仅对有归约操作的通信任务生效。

## 函数原型<a name="section620mcpsimp"></a>

```
uint32_t SetReduceType(uint32_t reduceType, uint8_t dstDataType = 0, uint8_t srcDataType = 0)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="14.99%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.02%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.99%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p167361341213"><a name="p167361341213"></a><a name="p167361341213"></a>reduceType</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p137362417119"><a name="p137362417119"></a><a name="p137362417119"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p69241958121317"><a name="p69241958121317"></a><a name="p69241958121317"></a>归约操作类型，仅对有归约操作的通信任务生效。uint32_t类型，取值详见<a href="HCCL使用说明.md#table2469980529">表2</a>参数说明。</p>
</td>
</tr>
<tr id="row20336914520"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p13336171656"><a name="p13336171656"></a><a name="p13336171656"></a>dstDataType</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p17336191251"><a name="p17336191251"></a><a name="p17336191251"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p171784316267"><a name="p171784316267"></a><a name="p171784316267"></a>通信任务中输出数据的数据类型。uint8_t类型，该参数的取值范围请参考<a href="HCCL使用说明.md#table116710585514">表1</a>。</p>
<p id="p32687448262"><a name="p32687448262"></a><a name="p32687448262"></a><span id="ph126815448267"><a name="ph126815448267"></a><a name="ph126815448267"></a><term id="zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312391781_term1253731311225"></a>Ascend 910C</term></span>，该参数暂不支持，配置后不生效。</p>
<p id="p926854417269"><a name="p926854417269"></a><a name="p926854417269"></a><span id="ph11268114402616"><a name="ph11268114402616"></a><a name="ph11268114402616"></a><term id="zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312391781_term11962195213215"></a>Ascend 910B</term></span>，该参数暂不支持，配置后不生效。</p>
</td>
</tr>
<tr id="row3354246512"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p113559415512"><a name="p113559415512"></a><a name="p113559415512"></a>srcDataType</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p535574457"><a name="p535574457"></a><a name="p535574457"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1339119178286"><a name="p1339119178286"></a><a name="p1339119178286"></a>通信任务中输入数据的数据类型。uint8_t类型，该参数的取值范围请参考<a href="HCCL使用说明.md#table116710585514">表1</a>。</p>
<p id="p0256172514298"><a name="p0256172514298"></a><a name="p0256172514298"></a><span id="ph16256142552915"><a name="ph16256142552915"></a><a name="ph16256142552915"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_1"></a>Ascend 910C</term></span>，该参数暂不支持，配置后不生效。</p>
<p id="p72569258296"><a name="p72569258296"></a><a name="p72569258296"></a>针对<span id="ph17256132515290"><a name="ph17256132515290"></a><a name="ph17256132515290"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_1"><a name="zh-cn_topic_0000001312391781_term11962195213215_1"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_1"></a>Ascend 910B</term></span>，该参数暂不支持，配置后不生效。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

-   0表示设置成功。
-   非0表示设置失败。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section1665082013318"></a>

本接口的调用示例请见[调用示例](SetOpType.md#section1665082013318)。

