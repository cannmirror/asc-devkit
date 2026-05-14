# fp8数据类型简介<a name="ZH-CN_TOPIC_0000002544131976"></a>

SIMT编程支持3种fp8数据类型，分别是float8\_e4m3\_t、float8\_e5m2\_t、hifloat8\_t。其数据范围如下：

<a name="table1347110513239"></a>
<table><thead align="left"><tr id="row1747155132311"><th class="cellrowborder" valign="top" width="20.327967203279673%" id="mcps1.1.6.1.1"><p id="p747113512239"><a name="p747113512239"></a><a name="p747113512239"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="10.958904109589042%" id="mcps1.1.6.1.2"><p id="p1947105122315"><a name="p1947105122315"></a><a name="p1947105122315"></a>符号位宽</p>
</th>
<th class="cellrowborder" valign="top" width="14.51854814518548%" id="mcps1.1.6.1.3"><p id="p1947116552317"><a name="p1947116552317"></a><a name="p1947116552317"></a>指数位宽</p>
</th>
<th class="cellrowborder" valign="top" width="12.778722127787221%" id="mcps1.1.6.1.4"><p id="p1747111518232"><a name="p1747111518232"></a><a name="p1747111518232"></a>尾数位宽</p>
</th>
<th class="cellrowborder" valign="top" width="41.415858414158585%" id="mcps1.1.6.1.5"><p id="p164711354238"><a name="p164711354238"></a><a name="p164711354238"></a>取值范围</p>
</th>
</tr>
</thead>
<tbody><tr id="row1904173318378"><td class="cellrowborder" valign="top" width="20.327967203279673%" headers="mcps1.1.6.1.1 "><p id="p54711512310"><a name="p54711512310"></a><a name="p54711512310"></a>float8_e4m3_t</p>
</td>
<td class="cellrowborder" valign="top" width="10.958904109589042%" headers="mcps1.1.6.1.2 "><p id="p347118552318"><a name="p347118552318"></a><a name="p347118552318"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="14.51854814518548%" headers="mcps1.1.6.1.3 "><p id="p144728513234"><a name="p144728513234"></a><a name="p144728513234"></a>4</p>
</td>
<td class="cellrowborder" valign="top" width="12.778722127787221%" headers="mcps1.1.6.1.4 "><p id="p14729517236"><a name="p14729517236"></a><a name="p14729517236"></a>3</p>
</td>
<td class="cellrowborder" valign="top" width="41.415858414158585%" headers="mcps1.1.6.1.5 "><p id="p7472125132312"><a name="p7472125132312"></a><a name="p7472125132312"></a>[2<sup id="sup134721452234"><a name="sup134721452234"></a><a name="sup134721452234"></a>6</sup> - 2<sup id="sup947215518239"><a name="sup947215518239"></a><a name="sup947215518239"></a>9</sup>, 2<sup id="sup34721351236"><a name="sup34721351236"></a><a name="sup34721351236"></a>9</sup> - 2<sup id="sup1347285142319"><a name="sup1347285142319"></a><a name="sup1347285142319"></a>6</sup>]</p>
</td>
</tr>
<tr id="row247114513237"><td class="cellrowborder" valign="top" width="20.327967203279673%" headers="mcps1.1.6.1.1 "><p id="p2471155152318"><a name="p2471155152318"></a><a name="p2471155152318"></a>float8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" width="10.958904109589042%" headers="mcps1.1.6.1.2 "><p id="p1447175142319"><a name="p1447175142319"></a><a name="p1447175142319"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="14.51854814518548%" headers="mcps1.1.6.1.3 "><p id="p447115192312"><a name="p447115192312"></a><a name="p447115192312"></a>5</p>
</td>
<td class="cellrowborder" valign="top" width="12.778722127787221%" headers="mcps1.1.6.1.4 "><p id="p1147117512238"><a name="p1147117512238"></a><a name="p1147117512238"></a>2</p>
</td>
<td class="cellrowborder" valign="top" width="41.415858414158585%" headers="mcps1.1.6.1.5 "><p id="p17471357235"><a name="p17471357235"></a><a name="p17471357235"></a>[2<sup id="sup647115572316"><a name="sup647115572316"></a><a name="sup647115572316"></a>13</sup> - 2<sup id="sup04713516232"><a name="sup04713516232"></a><a name="sup04713516232"></a>16</sup>, 2<sup id="sup124711755231"><a name="sup124711755231"></a><a name="sup124711755231"></a>16</sup> - 2<sup id="sup1547111522311"><a name="sup1547111522311"></a><a name="sup1547111522311"></a>13</sup>]</p>
</td>
</tr>
<tr id="row047110515237"><td class="cellrowborder" valign="top" width="20.327967203279673%" headers="mcps1.1.6.1.1 "><p id="p147081836173720"><a name="p147081836173720"></a><a name="p147081836173720"></a>hifloat8_t</p>
</td>
<td class="cellrowborder" valign="top" width="10.958904109589042%" headers="mcps1.1.6.1.2 "><p id="p070833610370"><a name="p070833610370"></a><a name="p070833610370"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="14.51854814518548%" headers="mcps1.1.6.1.3 "><p id="p13707153603718"><a name="p13707153603718"></a><a name="p13707153603718"></a><span id="ph84511382419"><a name="ph84511382419"></a><a name="ph84511382419"></a>由点域编码决定</span></p>
</td>
<td class="cellrowborder" valign="top" width="12.778722127787221%" headers="mcps1.1.6.1.4 "><p id="p770718361371"><a name="p770718361371"></a><a name="p770718361371"></a><span id="ph582411818383"><a name="ph582411818383"></a><a name="ph582411818383"></a>由点域编码决定</span></p>
</td>
<td class="cellrowborder" valign="top" width="41.415858414158585%" headers="mcps1.1.6.1.5 "><p id="p3703163617377"><a name="p3703163617377"></a><a name="p3703163617377"></a>点域编码决定数据精度与取值范围</p>
</td>
</tr>
</tbody>
</table>

浮点数由符号位（S）、指数（E）、尾数（M）三个部分组成，不同类型的浮点数，三个部分所占的比特数可能不同。

-   float8\_e4m3\_t

    下图是一个fp8\_e4m3fn\_t类型的示例，其符号位占用1位，指数占用4位，尾数占用3位，表示的结果为 \(-1\)^1 × 2^-3 × 2^-6。

    **图 1**  float8\_e4m3\_t示例图<a name="fig693644411325"></a>  
    ![](../../../figures/float8_e4m3_t示例图.png "float8_e4m3_t示例图")

-   float8\_e5m2\_t

    下图是一个fp8\_e5m2\_t类型的示例，其符号位占用1位，指数占用5位，尾数占用2位，表示的结果为 \(-1\)^0 × \(2 - 0.25\) × 2^\(30 -15\)=1.75 × 2^15。

    **图 2**  float8\_e5m2\_t示例图<a name="fig4792151573316"></a>  
    ![](../../../figures/float8_e5m2_t示例图.png "float8_e5m2_t示例图")

-   hifloat8\_t

    hifloat8\_t类型相对其他类型增加了指数位宽控制字段D，用于指示指数位和尾数位的编码方式。

    hifloat8\_t类型根据点域的不同，有不同的编码方式，下面一一列出。

    **图 3**  S、E、M在不同点域D值下的bit位分布<a name="fig261021710100"></a>  
    ![](../../../figures/S-E-M在不同点域D值下的bit位分布-65.png "S-E-M在不同点域D值下的bit位分布-65")

    下图示例中，其符号位占用1位，指数占用2位，尾数占用3位，D字段为2比特b01，S<sub>v</sub>=1，E<sub>v</sub>=3，M<sub>v</sub>  = 2<sup>-1</sup>  + 2<sup>-2</sup>，表示的结果为14。下标v表示各部分的具体数值。

    ![](../../../figures/流水任务运行示意图-66.png)

