# GetUserWorkspace<a name="ZH-CN_TOPIC_0000001666932721"></a>

## AI处理器支持情况<a name="section1550532418810"></a>

<a name="zh-cn_topic_0000001666431622_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001666431622_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001666431622_p1883113061818"><a name="zh-cn_topic_0000001666431622_p1883113061818"></a><a name="zh-cn_topic_0000001666431622_p1883113061818"></a><span id="zh-cn_topic_0000001666431622_ph20833205312295"><a name="zh-cn_topic_0000001666431622_ph20833205312295"></a><a name="zh-cn_topic_0000001666431622_ph20833205312295"></a>AI处理器类型</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001666431622_p783113012187"><a name="zh-cn_topic_0000001666431622_p783113012187"></a><a name="zh-cn_topic_0000001666431622_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001666431622_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001666431622_p48327011813"><a name="zh-cn_topic_0000001666431622_p48327011813"></a><a name="zh-cn_topic_0000001666431622_p48327011813"></a><span id="zh-cn_topic_0000001666431622_ph583230201815"><a name="zh-cn_topic_0000001666431622_ph583230201815"></a><a name="zh-cn_topic_0000001666431622_ph583230201815"></a><term id="zh-cn_topic_0000001666431622_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001666431622_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001666431622_zh-cn_topic_0000001312391781_term1253731311225"></a>Ascend 910C</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001666431622_p7948163910184"><a name="zh-cn_topic_0000001666431622_p7948163910184"></a><a name="zh-cn_topic_0000001666431622_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001666431622_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001666431622_p14832120181815"><a name="zh-cn_topic_0000001666431622_p14832120181815"></a><a name="zh-cn_topic_0000001666431622_p14832120181815"></a><span id="zh-cn_topic_0000001666431622_ph1483216010188"><a name="zh-cn_topic_0000001666431622_ph1483216010188"></a><a name="zh-cn_topic_0000001666431622_ph1483216010188"></a><term id="zh-cn_topic_0000001666431622_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001666431622_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001666431622_zh-cn_topic_0000001312391781_term11962195213215"></a>Ascend 910B</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001666431622_p19948143911820"><a name="zh-cn_topic_0000001666431622_p19948143911820"></a><a name="zh-cn_topic_0000001666431622_p19948143911820"></a>√</p>
</td>
</tr>
<tr id="row177153301877"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p871618304717"><a name="p871618304717"></a><a name="p871618304717"></a><span id="ph2010715480019"><a name="ph2010715480019"></a><a name="ph2010715480019"></a>Kirin X90</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p107166301277"><a name="p107166301277"></a><a name="p107166301277"></a>√</p>
</td>
</tr>
<tr id="row63371527104112"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p14731298411"><a name="p14731298411"></a><a name="p14731298411"></a><span id="ph114731729174114"><a name="ph114731729174114"></a><a name="ph114731729174114"></a>Kirin 9030</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p14473329104113"><a name="p14473329104113"></a><a name="p14473329104113"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

获取用户使用的workspace指针。Kernel直调开发方式下，如果未开启HAVE\_WORKSPACE编译选项，框架不会自动设置系统workspace。如果使用了[Matmul Kernel侧接口](Matmul-Kernel侧接口.md)等需要系统workspace的高阶API，kernel侧需要通过[SetSysWorkSpace](SetSysWorkSpace.md)设置系统workspace，此时用户workspace需要通过该接口获取。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline GM_ADDR GetUserWorkspace(GM_ADDR workspace)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  接口参数说明

<a name="table1055216132132"></a>
<table><thead align="left"><tr id="row105531513121315"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.2.4.1.1"><p id="p5553171319138"><a name="p5553171319138"></a><a name="p5553171319138"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.2.4.1.2"><p id="p5553151313131"><a name="p5553151313131"></a><a name="p5553151313131"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.2.4.1.3"><p id="p655316136139"><a name="p655316136139"></a><a name="p655316136139"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row5553201314135"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p8553813111314"><a name="p8553813111314"></a><a name="p8553813111314"></a>workspace</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p755318134134"><a name="p755318134134"></a><a name="p755318134134"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p1185064715302"><a name="p1185064715302"></a><a name="p1185064715302"></a>传入workspace的指针，包括系统workspace和用户使用的workspace。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

用户使用workspace指针。

