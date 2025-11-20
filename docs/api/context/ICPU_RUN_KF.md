# ICPU\_RUN\_KF<a name="ZH-CN_TOPIC_0000002080882157"></a>

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

## 功能说明<a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_section259105813316"></a>

进行核函数的CPU侧运行验证时，CPU调测总入口，完成CPU侧的算子程序调用。

## 函数原型<a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_section2067518173415"></a>

```
#define ICPU_RUN_KF(func, blkdim, ...)
```

## 参数说明<a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_section158061867342"></a>

<a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_row42461942101815"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p284425844311"><a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p284425844311"></a><a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p284425844311"></a>func</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p158449584436"><a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p158449584436"></a><a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p158449584436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p297233812230"><a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p297233812230"></a><a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p297233812230"></a>算子的kernel函数指针。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_row8219113874910"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p16219938104918"><a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p16219938104918"></a><a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p16219938104918"></a>blkdim</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p17219113894919"><a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p17219113894919"></a><a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p17219113894919"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p1721973834915"><a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p1721973834915"></a><a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p1721973834915"></a>算子的核心数，corenum。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_row14935161035014"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p9935110175011"><a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p9935110175011"></a><a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p9935110175011"></a>...</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p199351310145011"><a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p199351310145011"></a><a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p199351310145011"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p79351710125012"><a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p79351710125012"></a><a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_p79351710125012"></a>所有的入参和出参，依次填入，当前参数个数限制为32个，超出32时会出现编译错误。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_section640mcpsimp"></a>

无

## 约束说明<a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_section794123819592"></a>

除了func、blkdim以外，其他的变量都必须是通过GmAlloc分配的共享内存的指针；传入的参数的数量和顺序都必须和kernel保持一致。

## 调用示例<a name="zh-cn_topic_0000001963799134_zh-cn_topic_0000001541924164_section82241477610"></a>

```
ICPU_RUN_KF(sort_kernel0, coreNum, (uint8_t*)x, (uint8_t*)y);
```

