# Axpy<a name="ZH-CN_TOPIC_0000001424952592"></a>

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

源操作数src中每个元素与标量求积后和目的操作数dst中的对应元素相加，计算公式如下：

![](figures/zh-cn_formulaimage_0000002357932516.png)

## 函数原型<a name="section620mcpsimp"></a>

-   tensor前n个数据计算

    ```
    template <typename T, typename U>
    __aicore__ inline void Axpy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const U& scalarValue, const int32_t& count)
    ```

-   tensor高维切分计算
    -   mask逐bit模式

        ```
        template <typename T, typename U, bool isSetMask = true>
        __aicore__ inline void Axpy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const U& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
        ```

    -   mask连续模式

        ```
        template <typename T, typename U, bool isSetMask = true>
        __aicore__ inline void Axpy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const U& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
        ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="14.549999999999999%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="85.45%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="14.549999999999999%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="85.45%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>目的操作数数据类型。目的操作数和源操作数的数据类型约束请参考<a href="#table17640143723412">表3</a>。</p>
<p id="p209372313303"><a name="p209372313303"></a><a name="p209372313303"></a><span id="ph6937237308"><a name="ph6937237308"></a><a name="ph6937237308"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_1"><a name="zh-cn_topic_0000001312391781_term11962195213215_1"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_1"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term1551319498507_1"><a name="zh-cn_topic_0000001312391781_term1551319498507_1"></a><a name="zh-cn_topic_0000001312391781_term1551319498507_1"></a>Atlas A2 推理系列产品</term></span>，支持的数据类型为：half/float</p>
<p id="p19825153101820"><a name="p19825153101820"></a><a name="p19825153101820"></a><span id="ph58252535188"><a name="ph58252535188"></a><a name="ph58252535188"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_1"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term12835255145414_1"><a name="zh-cn_topic_0000001312391781_term12835255145414_1"></a><a name="zh-cn_topic_0000001312391781_term12835255145414_1"></a>Atlas A3 推理系列产品</term></span>，支持的数据类型为：half/float</p>
</td>
</tr>
<tr id="row1648615377"><td class="cellrowborder" valign="top" width="14.549999999999999%" headers="mcps1.2.3.1.1 "><p id="p1212015191874"><a name="p1212015191874"></a><a name="p1212015191874"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="85.45%" headers="mcps1.2.3.1.2 "><p id="p1912061914715"><a name="p1912061914715"></a><a name="p1912061914715"></a>源操作数数据类型。</p>
<p id="p3814184714550"><a name="p3814184714550"></a><a name="p3814184714550"></a><span id="ph128141447125516"><a name="ph128141447125516"></a><a name="ph128141447125516"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_2"><a name="zh-cn_topic_0000001312391781_term11962195213215_2"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_2"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term1551319498507_2"><a name="zh-cn_topic_0000001312391781_term1551319498507_2"></a><a name="zh-cn_topic_0000001312391781_term1551319498507_2"></a>Atlas A2 推理系列产品</term></span>，支持的数据类型为：half/float</p>
<p id="p1465254919192"><a name="p1465254919192"></a><a name="p1465254919192"></a><span id="ph565254919196"><a name="ph565254919196"></a><a name="ph565254919196"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_2"><a name="zh-cn_topic_0000001312391781_term1253731311225_2"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_2"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term12835255145414_2"><a name="zh-cn_topic_0000001312391781_term12835255145414_2"></a><a name="zh-cn_topic_0000001312391781_term12835255145414_2"></a>Atlas A3 推理系列产品</term></span>，支持的数据类型为：half/float</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001429830437_row18835145716587"><td class="cellrowborder" valign="top" width="14.549999999999999%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p1383515717581"><a name="zh-cn_topic_0000001429830437_p1383515717581"></a><a name="zh-cn_topic_0000001429830437_p1383515717581"></a>isSetMask</p>
</td>
<td class="cellrowborder" valign="top" width="85.45%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p77520541653"><a name="zh-cn_topic_0000001429830437_p77520541653"></a><a name="zh-cn_topic_0000001429830437_p77520541653"></a>是否在接口内部设置mask。</p>
<a name="zh-cn_topic_0000001429830437_ul1163765616511"></a><a name="zh-cn_topic_0000001429830437_ul1163765616511"></a><ul id="zh-cn_topic_0000001429830437_ul1163765616511"><li>true，表示在接口内部设置mask。</li><li>false，表示在接口外部设置mask，开发者需要使用<a href="SetVectorMask.md">SetVectorMask</a>接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table1549711469155"></a>
<table><thead align="left"><tr id="row12534194619150"><th class="cellrowborder" valign="top" width="14.510000000000002%" id="mcps1.2.4.1.1"><p id="p115341446121510"><a name="p115341446121510"></a><a name="p115341446121510"></a><strong id="b125344463152"><a name="b125344463152"></a><a name="b125344463152"></a>参数名称</strong></p>
</th>
<th class="cellrowborder" valign="top" width="9.49%" id="mcps1.2.4.1.2"><p id="p8534164621511"><a name="p8534164621511"></a><a name="p8534164621511"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="76%" id="mcps1.2.4.1.3"><p id="p6534046101518"><a name="p6534046101518"></a><a name="p6534046101518"></a><strong id="b105341546101519"><a name="b105341546101519"></a><a name="b105341546101519"></a>说明</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="row1253413467153"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.4.1.1 "><p id="p1534204617157"><a name="p1534204617157"></a><a name="p1534204617157"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="9.49%" headers="mcps1.2.4.1.2 "><p id="p3534104620153"><a name="p3534104620153"></a><a name="p3534104620153"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="76%" headers="mcps1.2.4.1.3 "><p id="p24155022212"><a name="p24155022212"></a><a name="p24155022212"></a>目的操作数。</p>
<p id="p5945720195112"><a name="p5945720195112"></a><a name="p5945720195112"></a><span id="zh-cn_topic_0000001530181537_ph173308471594"><a name="zh-cn_topic_0000001530181537_ph173308471594"></a><a name="zh-cn_topic_0000001530181537_ph173308471594"></a><span id="zh-cn_topic_0000001530181537_ph9902231466"><a name="zh-cn_topic_0000001530181537_ph9902231466"></a><a name="zh-cn_topic_0000001530181537_ph9902231466"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816"><a name="zh-cn_topic_0000001530181537_ph1782115034816"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p3247968151"><a name="p3247968151"></a><a name="p3247968151"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row3534104617155"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.4.1.1 "><p id="p1534946101517"><a name="p1534946101517"></a><a name="p1534946101517"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="9.49%" headers="mcps1.2.4.1.2 "><p id="p14534164616158"><a name="p14534164616158"></a><a name="p14534164616158"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76%" headers="mcps1.2.4.1.3 "><p id="p6534104601515"><a name="p6534104601515"></a><a name="p6534104601515"></a>源操作数。</p>
<p id="p738903012411"><a name="p738903012411"></a><a name="p738903012411"></a><span id="zh-cn_topic_0000001530181537_ph173308471594_1"><a name="zh-cn_topic_0000001530181537_ph173308471594_1"></a><a name="zh-cn_topic_0000001530181537_ph173308471594_1"></a><span id="zh-cn_topic_0000001530181537_ph9902231466_1"><a name="zh-cn_topic_0000001530181537_ph9902231466_1"></a><a name="zh-cn_topic_0000001530181537_ph9902231466_1"></a><span id="zh-cn_topic_0000001530181537_ph1782115034816_1"><a name="zh-cn_topic_0000001530181537_ph1782115034816_1"></a><a name="zh-cn_topic_0000001530181537_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p10320121219285"><a name="p10320121219285"></a><a name="p10320121219285"></a><span id="ph18912101252818"><a name="ph18912101252818"></a><a name="ph18912101252818"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row1053417466157"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.4.1.1 "><p id="p1253584619151"><a name="p1253584619151"></a><a name="p1253584619151"></a>scalarValue</p>
</td>
<td class="cellrowborder" valign="top" width="9.49%" headers="mcps1.2.4.1.2 "><p id="p053534691510"><a name="p053534691510"></a><a name="p053534691510"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76%" headers="mcps1.2.4.1.3 "><p id="p053524613153"><a name="p053524613153"></a><a name="p053524613153"></a>源操作数，scalar标量。scalarValue的数据类型需要和src保持一致。</p>
</td>
</tr>
<tr id="row12541153217106"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.4.1.1 "><p id="p10406164251014"><a name="p10406164251014"></a><a name="p10406164251014"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="9.49%" headers="mcps1.2.4.1.2 "><p id="p144065422107"><a name="p144065422107"></a><a name="p144065422107"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76%" headers="mcps1.2.4.1.3 "><p id="p1329419116580"><a name="p1329419116580"></a><a name="p1329419116580"></a>参与计算的元素个数。</p>
</td>
</tr>
<tr id="row75351846161514"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.4.1.1 "><p id="p2554141321313"><a name="p2554141321313"></a><a name="p2554141321313"></a>mask/mask[]</p>
</td>
<td class="cellrowborder" valign="top" width="9.49%" headers="mcps1.2.4.1.2 "><p id="p10535746191515"><a name="p10535746191515"></a><a name="p10535746191515"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001530181537_p0554313181312"><a name="zh-cn_topic_0000001530181537_p0554313181312"></a><a name="zh-cn_topic_0000001530181537_p0554313181312"></a><span id="zh-cn_topic_0000001530181537_ph793119540147"><a name="zh-cn_topic_0000001530181537_ph793119540147"></a><a name="zh-cn_topic_0000001530181537_ph793119540147"></a>mask用于控制每次迭代内参与计算的元素。</span></p>
<a name="zh-cn_topic_0000001530181537_ul1255411133132"></a><a name="zh-cn_topic_0000001530181537_ul1255411133132"></a><ul id="zh-cn_topic_0000001530181537_ul1255411133132"><li>逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。<p id="zh-cn_topic_0000001530181537_p121114581013"><a name="zh-cn_topic_0000001530181537_p121114581013"></a><a name="zh-cn_topic_0000001530181537_p121114581013"></a>mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 2<sup id="zh-cn_topic_0000001530181537_sup1411059101"><a name="zh-cn_topic_0000001530181537_sup1411059101"></a><a name="zh-cn_topic_0000001530181537_sup1411059101"></a>64</sup>-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000001530181537_sup1711155161017"><a name="zh-cn_topic_0000001530181537_sup1711155161017"></a><a name="zh-cn_topic_0000001530181537_sup1711155161017"></a>64</sup>-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000001530181537_sup181195111019"><a name="zh-cn_topic_0000001530181537_sup181195111019"></a><a name="zh-cn_topic_0000001530181537_sup181195111019"></a>32</sup>-1]。</p>
<p id="zh-cn_topic_0000001530181537_p711354105"><a name="zh-cn_topic_0000001530181537_p711354105"></a><a name="zh-cn_topic_0000001530181537_p711354105"></a>例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。</p>
</li></ul>
<a name="zh-cn_topic_0000001530181537_ul18554121313135"></a><a name="zh-cn_topic_0000001530181537_ul18554121313135"></a><ul id="zh-cn_topic_0000001530181537_ul18554121313135"><li>连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。</li></ul>
</td>
</tr>
<tr id="row1953574611513"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.4.1.1 "><p id="p3535194615157"><a name="p3535194615157"></a><a name="p3535194615157"></a>repeatTime</p>
</td>
<td class="cellrowborder" valign="top" width="9.49%" headers="mcps1.2.4.1.2 "><p id="p0535104621511"><a name="p0535104621511"></a><a name="p0535104621511"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76%" headers="mcps1.2.4.1.3 "><p id="p353564621520"><a name="p353564621520"></a><a name="p353564621520"></a>重复迭代次数。</p>
<p id="p15733456113418"><a name="p15733456113418"></a><a name="p15733456113418"></a>矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。</p>
</td>
</tr>
<tr id="row10720141532815"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.4.1.1 "><p id="p393816287103"><a name="p393816287103"></a><a name="p393816287103"></a>repeatParams</p>
</td>
<td class="cellrowborder" valign="top" width="9.49%" headers="mcps1.2.4.1.2 "><p id="p1393942810108"><a name="p1393942810108"></a><a name="p1393942810108"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001530181537_p455461351319"><a name="zh-cn_topic_0000001530181537_p455461351319"></a><a name="zh-cn_topic_0000001530181537_p455461351319"></a>控制操作数地址步长的参数。<a href="UnaryRepeatParams.md">UnaryRepeatParams</a>类型，包含操作数相邻迭代间相同<span id="zh-cn_topic_0000001530181537_ph1256166185416"><a name="zh-cn_topic_0000001530181537_ph1256166185416"></a><a name="zh-cn_topic_0000001530181537_ph1256166185416"></a>DataBlock</span>的地址步长，操作数同一迭代内不同<span id="zh-cn_topic_0000001530181537_ph131833567170"><a name="zh-cn_topic_0000001530181537_ph131833567170"></a><a name="zh-cn_topic_0000001530181537_ph131833567170"></a>DataBlock</span>的地址步长等参数。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  数据类型约束

<a name="table17640143723412"></a>
<table><thead align="left"><tr id="row264013712343"><th class="cellrowborder" valign="top" width="14.45%" id="mcps1.2.6.1.1"><p id="p12640123716343"><a name="p12640123716343"></a><a name="p12640123716343"></a>src数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="14.26%" id="mcps1.2.6.1.2"><p id="p7640113723419"><a name="p7640113723419"></a><a name="p7640113723419"></a>scalar数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="15.920000000000002%" id="mcps1.2.6.1.3"><p id="p1764017372344"><a name="p1764017372344"></a><a name="p1764017372344"></a>dst数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="8.290000000000001%" id="mcps1.2.6.1.4"><p id="p129529555614"><a name="p129529555614"></a><a name="p129529555614"></a>PAR</p>
</th>
<th class="cellrowborder" valign="top" width="47.08%" id="mcps1.2.6.1.5"><p id="p182911919589"><a name="p182911919589"></a><a name="p182911919589"></a>支持的型号</p>
</th>
</tr>
</thead>
<tbody><tr id="row1464018376343"><td class="cellrowborder" valign="top" width="14.45%" headers="mcps1.2.6.1.1 "><p id="p46401237143413"><a name="p46401237143413"></a><a name="p46401237143413"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="14.26%" headers="mcps1.2.6.1.2 "><p id="p364093711349"><a name="p364093711349"></a><a name="p364093711349"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="15.920000000000002%" headers="mcps1.2.6.1.3 "><p id="p16411737133418"><a name="p16411737133418"></a><a name="p16411737133418"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="8.290000000000001%" headers="mcps1.2.6.1.4 "><p id="p15952559568"><a name="p15952559568"></a><a name="p15952559568"></a>128</p>
</td>
<td class="cellrowborder" valign="top" width="47.08%" headers="mcps1.2.6.1.5 "><p id="p0177122102219"><a name="p0177122102219"></a><a name="p0177122102219"></a><span id="ph1117712218226"><a name="ph1117712218226"></a><a name="ph1117712218226"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_3"><a name="zh-cn_topic_0000001312391781_term11962195213215_3"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_3"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term1551319498507_3"><a name="zh-cn_topic_0000001312391781_term1551319498507_3"></a><a name="zh-cn_topic_0000001312391781_term1551319498507_3"></a>Atlas A2 推理系列产品</term></span></p>
<p id="p101771215223"><a name="p101771215223"></a><a name="p101771215223"></a><span id="ph1017752142220"><a name="ph1017752142220"></a><a name="ph1017752142220"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_3"><a name="zh-cn_topic_0000001312391781_term1253731311225_3"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_3"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term12835255145414_3"><a name="zh-cn_topic_0000001312391781_term12835255145414_3"></a><a name="zh-cn_topic_0000001312391781_term12835255145414_3"></a>Atlas A3 推理系列产品</term></span></p>
</td>
</tr>
<tr id="row764103713411"><td class="cellrowborder" valign="top" width="14.45%" headers="mcps1.2.6.1.1 "><p id="p15641937103415"><a name="p15641937103415"></a><a name="p15641937103415"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="14.26%" headers="mcps1.2.6.1.2 "><p id="p94755865617"><a name="p94755865617"></a><a name="p94755865617"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="15.920000000000002%" headers="mcps1.2.6.1.3 "><p id="p1979975814562"><a name="p1979975814562"></a><a name="p1979975814562"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="8.290000000000001%" headers="mcps1.2.6.1.4 "><p id="p199524519569"><a name="p199524519569"></a><a name="p199524519569"></a>64</p>
</td>
<td class="cellrowborder" valign="top" width="47.08%" headers="mcps1.2.6.1.5 "><p id="p18918134806"><a name="p18918134806"></a><a name="p18918134806"></a><span id="ph9918534503"><a name="ph9918534503"></a><a name="ph9918534503"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_4"><a name="zh-cn_topic_0000001312391781_term11962195213215_4"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_4"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term1551319498507_4"><a name="zh-cn_topic_0000001312391781_term1551319498507_4"></a><a name="zh-cn_topic_0000001312391781_term1551319498507_4"></a>Atlas A2 推理系列产品</term></span></p>
<p id="p29188342020"><a name="p29188342020"></a><a name="p29188342020"></a><span id="ph14918434403"><a name="ph14918434403"></a><a name="ph14918434403"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_4"><a name="zh-cn_topic_0000001312391781_term1253731311225_4"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_4"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term12835255145414_4"><a name="zh-cn_topic_0000001312391781_term12835255145414_4"></a><a name="zh-cn_topic_0000001312391781_term12835255145414_4"></a>Atlas A3 推理系列产品</term></span></p>
</td>
</tr>
<tr id="row11641183743420"><td class="cellrowborder" valign="top" width="14.45%" headers="mcps1.2.6.1.1 "><p id="p464117371344"><a name="p464117371344"></a><a name="p464117371344"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="14.26%" headers="mcps1.2.6.1.2 "><p id="p36411437133419"><a name="p36411437133419"></a><a name="p36411437133419"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="15.920000000000002%" headers="mcps1.2.6.1.3 "><p id="p1064183723417"><a name="p1064183723417"></a><a name="p1064183723417"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="8.290000000000001%" headers="mcps1.2.6.1.4 "><p id="p139527513561"><a name="p139527513561"></a><a name="p139527513561"></a>64</p>
</td>
<td class="cellrowborder" valign="top" width="47.08%" headers="mcps1.2.6.1.5 "><p id="p15711371707"><a name="p15711371707"></a><a name="p15711371707"></a><span id="ph18571137306"><a name="ph18571137306"></a><a name="ph18571137306"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_5"><a name="zh-cn_topic_0000001312391781_term11962195213215_5"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_5"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term1551319498507_5"><a name="zh-cn_topic_0000001312391781_term1551319498507_5"></a><a name="zh-cn_topic_0000001312391781_term1551319498507_5"></a>Atlas A2 推理系列产品</term></span></p>
<p id="p1157111376018"><a name="p1157111376018"></a><a name="p1157111376018"></a><span id="ph557211371505"><a name="ph557211371505"></a><a name="ph557211371505"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_5"><a name="zh-cn_topic_0000001312391781_term1253731311225_5"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_5"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term12835255145414_5"><a name="zh-cn_topic_0000001312391781_term12835255145414_5"></a><a name="zh-cn_topic_0000001312391781_term12835255145414_5"></a>Atlas A3 推理系列产品</term></span></p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section17124037164714"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   操作数地址重叠约束请参考[通用地址重叠约束](通用说明和约束.md#section668772811100)。

-   使用tensor高维切分计算接口时，src和scalar的数据类型为half、dst的数据类型为float的情况下，一个迭代处理的源操作数元素个数需要和目的操作数保持一致，所以每次迭代选取前4个datablock参与计算。设置Repeat Stride参数和mask参数以及地址重叠时，需要考虑该限制。

## 调用示例<a name="section642mcpsimp"></a>

本样例中只展示Compute流程中的部分代码。如果您需要运行样例代码，请将该代码段拷贝并替换[更多样例](#section10630194131318)完整样例模板中Compute函数的部分代码即可。

-   tensor高维切分计算样例-mask连续模式

    ```
    // repeatTime = 4, mask = 128, 128 elements one repeat, 512 elements total
    // srcLocal数据类型为half，scalar数据类型为half，dstLocal数据类型为half
    // dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat
    // dstRepStride, srcRepStride = 8, no gap between repeats 
    AscendC::Axpy(dstLocal, srcLocal, (half)2.0, 128, 4,{ 1, 1, 8, 8 });
    
    // srcLocal数据类型为half，scalar数据类型为half，dstLocal数据类型为float
    // repeatTime = 8, mask = 64, 64 elements one repeat, 512 elements total
    // dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat
    // dstRepStride = 8, srcRepStride = 4, no gap between repeats 
    AscendC::Axpy(dstLocal, srcLocal, (half)2.0, 64, 8,{ 1, 1, 8, 4 }); // 每次迭代选取源操作数前4个datablock参与计算
    ```

-   tensor高维切分计算样例-mask逐bit模式

    ```
    uint64_t mask[2] = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
    // repeatTime = 4, 128 elements one repeat, 512 elements total, half精度组合
    // dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat
    // dstRepStride, srcRepStride = 8, no gap between repeats
    AscendC::Axpy(dstLocal, srcLocal, (half)2.0, mask, 4,{ 1, 1, 8, 8 });
    ```

-   tensor前n个数据计算样例

    ```
    AscendC::Axpy(dstLocal, src0Local, (half)2.0, 512);// half精度组合
    ```

## 更多样例<a name="section10630194131318"></a>

-   完整样例一：srcLocal、scalar、dstLocal的数据类型均为half。

    ```
    #include "kernel_operator.h"
    class KernelAxpy {
    public:
        __aicore__ inline KernelAxpy() {}
        __aicore__ inline void Init(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm)
        {
            srcGlobal.SetGlobalBuffer((__gm__ half*)srcGm);
            dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
            pipe.InitBuffer(inQueueSrc, 1, 512 * sizeof(half));
            pipe.InitBuffer(outQueueDst, 1, 512 * sizeof(half));
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
            AscendC::LocalTensor<half> srcLocal = inQueueSrc.AllocTensor<half>();
            AscendC::DataCopy(srcLocal, srcGlobal, 512);
            inQueueSrc.EnQue(srcLocal);
        }
        __aicore__ inline void Compute()
        {
            AscendC::LocalTensor<half> srcLocal = inQueueSrc.DeQue<half>();
            AscendC::LocalTensor<half> dstLocal = outQueueDst.AllocTensor<half>();
     
            AscendC::Duplicate(dstLocal, (half)0.0, 512);
            AscendC::Axpy(dstLocal, srcLocal, (half)2.0, 512);
     
            outQueueDst.EnQue<half>(dstLocal);
            inQueueSrc.FreeTensor(srcLocal);
        }
        __aicore__ inline void CopyOut()
        {
            AscendC::LocalTensor<half> dstLocal = outQueueDst.DeQue<half>();
            AscendC::DataCopy(dstGlobal, dstLocal, 512);
            outQueueDst.FreeTensor(dstLocal);
        }
    private:
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
        AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
        AscendC::GlobalTensor<half> srcGlobal, dstGlobal;
    };
    extern "C" __global__ __aicore__ void kernel_vec_ternary_scalar_Axpy_half_2_half(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm)
    {
        KernelAxpy op;
        op.Init(srcGm, dstGm);
        op.Process();
    }
    ```

    结果示例如下：

    ```
    输入数据(srcGm):
    [1. 1. 1. 1. 1. 1. ... 1.]
    输出数据(dstGm):
    [2. 2. 2. 2. 2. 2. ... 2.]
    ```

-   完整样例二：srcLocal、scalar的数据类型为half，dstLocal的数据类型为float。

    ```
    #include "kernel_operator.h"
    class KernelAxpy {
    public:
        __aicore__ inline KernelAxpy() {}
        __aicore__ inline void Init(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm)
        {
            srcGlobal.SetGlobalBuffer((__gm__ half*)srcGm);
            dstGlobal.SetGlobalBuffer((__gm__ float*)dstGm);
            pipe.InitBuffer(outQueueDst, 1, 512 * sizeof(float));
            pipe.InitBuffer(inQueueSrc, 1, 512 * sizeof(half));
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
            AscendC::LocalTensor<half> srcLocal = inQueueSrc.AllocTensor<half>();
            AscendC::DataCopy(srcLocal, srcGlobal, 512);
            inQueueSrc.EnQue(srcLocal);
        }
        __aicore__ inline void Compute()
        {
            AscendC::LocalTensor<half> srcLocal = inQueueSrc.DeQue<half>();
            AscendC::LocalTensor<float> dstLocal = outQueueDst.AllocTensor<float>();
     
            AscendC::Duplicate(dstLocal, 0.0f, 512);
            AscendC::Axpy(dstLocal, srcLocal, (half)2.0, 64, 8, { 1, 1, 8, 4 });
     
            outQueueDst.EnQue<float>(dstLocal);
            inQueueSrc.FreeTensor(srcLocal);
        }
        __aicore__ inline void CopyOut()
        {
            AscendC::LocalTensor<float> dstLocal = outQueueDst.DeQue<float>();
            AscendC::DataCopy(dstGlobal, dstLocal, 512);
            outQueueDst.FreeTensor(dstLocal);
        }
    private:
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
        AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
        AscendC::GlobalTensor<half> srcGlobal;
        AscendC::GlobalTensor<float> dstGlobal;
    };
    extern "C" __global__ __aicore__ void kernel_vec_ternary_scalar_Axpy_half_2_float(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm)
    {
        KernelAxpy op;
        op.Init(srcGm, dstGm);
        op.Process();
    }
    ```

    结果示例如下：

    ```
    输入数据(srcGm):
    [1. 1. 1. 1. 1. 1. ... 1.]
    输出数据(dstGm):
    [2. 2. 2. 2. 2. 2. ... 2.]
    ```

