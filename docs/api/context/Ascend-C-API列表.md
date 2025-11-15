# Ascend C API列表<a name="ZH-CN_TOPIC_0000001917094304"></a>

Ascend C提供一组类库API，开发者使用标准C++语法和类库API进行编程。Ascend C编程类库API示意图如下所示，分为：

-   Kernel API：用于实现算子核函数的API接口。包括：
    -   **基本数据结构**：kernel API中使用到的基本数据结构，比如GlobalTensor和LocalTensor。
    -   **基础API**：实现对硬件能力的抽象，开放芯片的能力，保证完备性和兼容性。标注为ISASI（Instruction Set Architecture Special Interface，硬件体系结构相关的接口）类别的API，不能保证跨硬件版本兼容。
    -   **高阶API**：实现一些常用的计算算法，用于提高编程开发效率，通常会调用多种基础API实现。高阶API包括数学库、Matmul、Softmax等API。高阶API可以保证兼容性。

-   **算子调测API**：用于算子调测的API，包括孪生调试，性能调测等。

进行Ascend C算子Host侧编程时，需要使用基础数据结构和API；完成算子开发后，需要使用Runtime API完成算子的调用。

![](figures/编程API.png)

## 基本数据结构列表<a name="section3119442205518"></a>

**表 1**  基本数据结构列表

<a name="table16268175715515"></a>
<table><thead align="left"><tr id="row11268657105518"><th class="cellrowborder" valign="top" width="40.37%" id="mcps1.2.3.1.1"><p id="p1026810577553"><a name="p1026810577553"></a><a name="p1026810577553"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="59.63%" id="mcps1.2.3.1.2"><p id="p226835735518"><a name="p226835735518"></a><a name="p226835735518"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row102681557105520"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p12251181311560"><a name="p12251181311560"></a><a name="p12251181311560"></a><a href="LocalTensor.md">LocalTensor</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p2025215484579"><a name="p2025215484579"></a><a name="p2025215484579"></a><span id="ph4396131516585"><a name="ph4396131516585"></a><a name="ph4396131516585"></a>LocalTensor用于存放AI Core中Local Memory（内部存储）的数据，支持逻辑位置<a href="TPosition.md">TPosition</a>为<span>VECIN、VECOUT、VECCALC、</span>A1<span>、</span>A2<span>、</span>B1<span>、</span>B2<span>、</span>CO1<span>、</span>CO2。</span></p>
</td>
</tr>
<tr id="row026925718557"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p142691057115518"><a name="p142691057115518"></a><a name="p142691057115518"></a><a href="GlobalTensor.md">GlobalTensor</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p625254835716"><a name="p625254835716"></a><a name="p625254835716"></a><span id="ph698815359590"><a name="ph698815359590"></a><a name="ph698815359590"></a>GlobalTensor用来存放Global Memory（外部存储）的全局数据。</span></p>
</td>
</tr>
<tr id="row132691757135514"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p12663337155617"><a name="p12663337155617"></a><a name="p12663337155617"></a><a href="ShapeInfo.md">ShapeInfo</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p17251164811573"><a name="p17251164811573"></a><a name="p17251164811573"></a><span id="ph2676113715594"><a name="ph2676113715594"></a><a name="ph2676113715594"></a>ShapeInfo用来存放LocalTensor或GlobalTensor的shape信息。</span></p>
</td>
</tr>
<tr id="row1726925715514"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p15191245165615"><a name="p15191245165615"></a><a name="p15191245165615"></a><a href="ListTensorDesc.md">ListTensorDesc</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p1925111480570"><a name="p1925111480570"></a><a name="p1925111480570"></a><span id="ph151811939145918"><a name="ph151811939145918"></a><a name="ph151811939145918"></a>ListTensorDesc用来解析符合以下内存排布格式的数据， 并在kernel侧根据索引获取储存对应数据的地址及shape信息。</span></p>
</td>
</tr>
<tr id="row826925718552"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p15871954155613"><a name="p15871954155613"></a><a name="p15871954155613"></a><a href="TensorDesc.md">TensorDesc</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p3250748125712"><a name="p3250748125712"></a><a name="p3250748125712"></a><span id="ph118431940145919"><a name="ph118431940145919"></a><a name="ph118431940145919"></a>TensorDesc用于储存<a href="ListTensorDesc.md">ListTensorDesc</a>.GetDesc()中根据index获取对应的Tensor描述信息。</span></p>
</td>
</tr>
<tr id="row32691573557"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p8269195705514"><a name="p8269195705514"></a><a name="p8269195705514"></a><a href="Coordinate.md">Coordinate</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p1825015488573"><a name="p1825015488573"></a><a name="p1825015488573"></a>Coordinate<span>本质上是一个元组（tuple），用于表示张量在不同维度的位置信息，即坐标值。</span></p>
</td>
</tr>
<tr id="row926915745510"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p192691757185519"><a name="p192691757185519"></a><a name="p192691757185519"></a><a href="Layout.md">Layout</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p1724954818576"><a name="p1724954818576"></a><a name="p1724954818576"></a><span id="ph198761643125913"><a name="ph198761643125913"></a><a name="ph198761643125913"></a>Layout&lt;Shape, Stride&gt;数据结构是描述多维张量内存布局的基础模板类，通过编译时的形状（Shape）和步长（Stride）信息，实现逻辑坐标空间到一维内存地址空间的映射，为复杂张量操作和硬件优化提供基础支持。</span></p>
</td>
</tr>
<tr id="row97101411165713"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p5710711115712"><a name="p5710711115712"></a><a name="p5710711115712"></a><a href="TensorTrait.md">TensorTrait</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p167101111155717"><a name="p167101111155717"></a><a name="p167101111155717"></a><span id="ph287618459597"><a name="ph287618459597"></a><a name="ph287618459597"></a>TensorTrait数据结构是描述Tensor相关信息的基础模板类，包含Tensor的数据类型、逻辑位置和Layout内存布局。</span></p>
</td>
</tr>
<tr id="row1462001485710"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p1726393614577"><a name="p1726393614577"></a><a name="p1726393614577"></a><a href="UnaryRepeatParams.md">UnaryRepeatParams</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p10620121414578"><a name="p10620121414578"></a><a name="p10620121414578"></a><span id="ph18941747105912"><a name="ph18941747105912"></a><a name="ph18941747105912"></a>UnaryRepeatParams为用于控制操作数地址步长的数据结构。结构体内包含操作数相邻迭代间相同<span id="zh-cn_topic_0000001487959374_ph1256166185416"><a name="zh-cn_topic_0000001487959374_ph1256166185416"></a><a name="zh-cn_topic_0000001487959374_ph1256166185416"></a>DataBlock</span>的地址步长，操作数同一迭代内不同<span id="zh-cn_topic_0000001487959374_ph763453417103"><a name="zh-cn_topic_0000001487959374_ph763453417103"></a><a name="zh-cn_topic_0000001487959374_ph763453417103"></a>DataBlock</span>的地址步长等参数。</span></p>
</td>
</tr>
<tr id="row15620181775713"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p16620417185713"><a name="p16620417185713"></a><a name="p16620417185713"></a><a href="BinaryRepeatParams.md">BinaryRepeatParams</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p156203177576"><a name="p156203177576"></a><a name="p156203177576"></a><span id="ph17969749125910"><a name="ph17969749125910"></a><a name="ph17969749125910"></a>BinaryRepeatParams为用于控制操作数地址步长的数据结构。结构体内包含操作数相邻迭代间相同<span id="zh-cn_topic_0000001533390357_ph1256166185416"><a name="zh-cn_topic_0000001533390357_ph1256166185416"></a><a name="zh-cn_topic_0000001533390357_ph1256166185416"></a>DataBlock</span>的地址步长，操作数同一迭代内不同<span id="zh-cn_topic_0000001533390357_ph1672554912125"><a name="zh-cn_topic_0000001533390357_ph1672554912125"></a><a name="zh-cn_topic_0000001533390357_ph1672554912125"></a>DataBlock</span>的地址步长等参数。</span></p>
</td>
</tr>
</tbody>
</table>

## 基础API<a name="section117632211201"></a>

**表 2**  标量计算API列表

<a name="table339023582010"></a>
<table><thead align="left"><tr id="row1539063572010"><th class="cellrowborder" valign="top" width="40.37%" id="mcps1.2.3.1.1"><p id="p13390235192015"><a name="p13390235192015"></a><a name="p13390235192015"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="59.63%" id="mcps1.2.3.1.2"><p id="p2390103519209"><a name="p2390103519209"></a><a name="p2390103519209"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row839013512016"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p136633102216"><a name="p136633102216"></a><a name="p136633102216"></a><a href="ScalarGetCountOfValue.md">ScalarGetCountOfValue</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p1739063514204"><a name="p1739063514204"></a><a name="p1739063514204"></a>获取一个uint64_t类型数字的二进制中0或者1的个数。</p>
</td>
</tr>
<tr id="row5390935132010"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p439033519205"><a name="p439033519205"></a><a name="p439033519205"></a><a href="ScalarCountLeadingZero.md">ScalarCountLeadingZero</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p739014352201"><a name="p739014352201"></a><a name="p739014352201"></a>计算一个uint64_t类型数字前导0的个数（二进制从最高位到第一个1一共有多少个0）。</p>
</td>
</tr>
<tr id="row187241145152620"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p19725445132613"><a name="p19725445132613"></a><a name="p19725445132613"></a><a href="ScalarCast.md">ScalarCast</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p772515454262"><a name="p772515454262"></a><a name="p772515454262"></a>将一个scalar的类型转换为指定的类型。</p>
</td>
</tr>
<tr id="row129095914341"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p15290059143412"><a name="p15290059143412"></a><a name="p15290059143412"></a><a href="CountBitsCntSameAsSignBit.md">CountBitsCntSameAsSignBit</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p05116051416"><a name="p05116051416"></a><a name="p05116051416"></a>计算一个uint64_t类型数字的二进制中，从最高数值位开始与符号位相同的连续比特位的个数。</p>
</td>
</tr>
<tr id="row944019598346"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p1062632319354"><a name="p1062632319354"></a><a name="p1062632319354"></a><a href="ScalarGetSFFValue.md">ScalarGetSFFValue</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p13440259173413"><a name="p13440259173413"></a><a name="p13440259173413"></a>获取一个uint64_t类型数字的二进制中第一个0或1出现的位置。</p>
</td>
</tr>
<tr id="row1566175913418"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p9566659173420"><a name="p9566659173420"></a><a name="p9566659173420"></a><a href="ToBfloat16.md">ToBfloat16</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p828219133366"><a name="p828219133366"></a><a name="p828219133366"></a>float类型标量数据转换成bfloat16_t类型标量数据。</p>
</td>
</tr>
<tr id="row13704105910340"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p16704195953414"><a name="p16704195953414"></a><a name="p16704195953414"></a><a href="ToFloat.md">ToFloat</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p15293151211360"><a name="p15293151211360"></a><a name="p15293151211360"></a>bfloat16_t类型标量数据转换成float类型标量数据。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  矢量计算API列表

<a name="table107281858237"></a>
<table><thead align="left"><tr id="row1372812592319"><th class="cellrowborder" valign="top" width="15.590000000000002%" id="mcps1.2.4.1.1"><p id="p28543193914"><a name="p28543193914"></a><a name="p28543193914"></a>分类</p>
</th>
<th class="cellrowborder" valign="top" width="24.64%" id="mcps1.2.4.1.2"><p id="p147285552316"><a name="p147285552316"></a><a name="p147285552316"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="59.77%" id="mcps1.2.4.1.3"><p id="p17281151239"><a name="p17281151239"></a><a name="p17281151239"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1972815510234"><td class="cellrowborder" rowspan="18" valign="top" width="15.590000000000002%" headers="mcps1.2.4.1.1 "><p id="p28542192920"><a name="p28542192920"></a><a name="p28542192920"></a>基础算术</p>
</td>
<td class="cellrowborder" valign="top" width="24.64%" headers="mcps1.2.4.1.2 "><p id="p472817542311"><a name="p472817542311"></a><a name="p472817542311"></a><a href="Exp.md">Exp</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.77%" headers="mcps1.2.4.1.3 "><p id="p14728115122318"><a name="p14728115122318"></a><a name="p14728115122318"></a>按元素取自然指数。</p>
</td>
</tr>
<tr id="row77297582318"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p74127324439"><a name="p74127324439"></a><a name="p74127324439"></a><a href="Ln.md">Ln</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p127291058234"><a name="p127291058234"></a><a name="p127291058234"></a>按元素取自然对数。</p>
</td>
</tr>
<tr id="row095531611435"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p441216325433"><a name="p441216325433"></a><a name="p441216325433"></a><a href="Abs.md">Abs</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p179557168437"><a name="p179557168437"></a><a name="p179557168437"></a>按元素取绝对值。</p>
</td>
</tr>
<tr id="row1698614591910"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p7412103212437"><a name="p7412103212437"></a><a name="p7412103212437"></a><a href="Reciprocal.md">Reciprocal</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p698612594920"><a name="p698612594920"></a><a name="p698612594920"></a>按元素取倒数。</p>
</td>
</tr>
<tr id="row204721719184318"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p7412103218431"><a name="p7412103218431"></a><a name="p7412103218431"></a><a href="Sqrt.md">Sqrt</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p144735190439"><a name="p144735190439"></a><a name="p144735190439"></a>按元素做开方。</p>
</td>
</tr>
<tr id="row1263518197432"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p134122326433"><a name="p134122326433"></a><a name="p134122326433"></a><a href="Rsqrt.md">Rsqrt</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1163651916435"><a name="p1163651916435"></a><a name="p1163651916435"></a>按元素做开方后取倒数。</p>
</td>
</tr>
<tr id="row11951420124314"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p8412143274317"><a name="p8412143274317"></a><a name="p8412143274317"></a><a href="Relu.md">Relu</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p19951020134315"><a name="p19951020134315"></a><a name="p19951020134315"></a>按元素做线性整流Relu。</p>
</td>
</tr>
<tr id="row687313439911"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p178741943997"><a name="p178741943997"></a><a name="p178741943997"></a><a href="Add.md">Add</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p58747431191"><a name="p58747431191"></a><a name="p58747431191"></a>按元素求和。</p>
</td>
</tr>
<tr id="row1874810244362"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1912104816385"><a name="p1912104816385"></a><a name="p1912104816385"></a><a href="Sub.md">Sub</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p107491245366"><a name="p107491245366"></a><a name="p107491245366"></a>按元素求差。</p>
</td>
</tr>
<tr id="row1291015249364"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p2121164811382"><a name="p2121164811382"></a><a name="p2121164811382"></a><a href="Mul.md">Mul</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p791052483612"><a name="p791052483612"></a><a name="p791052483612"></a>按元素求积。</p>
</td>
</tr>
<tr id="row81121256363"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p11121548153820"><a name="p11121548153820"></a><a name="p11121548153820"></a><a href="Div.md">Div</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p191131256368"><a name="p191131256368"></a><a name="p191131256368"></a>按元素求商。</p>
</td>
</tr>
<tr id="row152552025153613"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p191211748103818"><a name="p191211748103818"></a><a name="p191211748103818"></a><a href="Max.md">Max</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1325552515362"><a name="p1325552515362"></a><a name="p1325552515362"></a>按元素求最大值。</p>
</td>
</tr>
<tr id="row104017250363"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1312174833819"><a name="p1312174833819"></a><a name="p1312174833819"></a><a href="Min.md">Min</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p389445174112"><a name="p389445174112"></a><a name="p389445174112"></a>按元素求最小值。</p>
</td>
</tr>
<tr id="row14127182319571"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1448115232"><a name="p1448115232"></a><a name="p1448115232"></a><a href="Adds.md">Adds</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p884518405416"><a name="p884518405416"></a><a name="p884518405416"></a>矢量内每个元素与标量求和。</p>
</td>
</tr>
<tr id="row107601527165713"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p44471122316"><a name="p44471122316"></a><a name="p44471122316"></a><a href="Muls.md">Muls</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p577119572225"><a name="p577119572225"></a><a name="p577119572225"></a>矢量内每个元素与标量求积。</p>
</td>
</tr>
<tr id="row1277082912579"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p8441215238"><a name="p8441215238"></a><a name="p8441215238"></a><a href="Maxs.md">Maxs</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p18672613113517"><a name="p18672613113517"></a><a name="p18672613113517"></a>源操作数矢量内每个元素与标量相比，如果比标量大，则取源操作数值，比标量的值小，则取标量值。</p>
</td>
</tr>
<tr id="row33851439175710"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p9441515231"><a name="p9441515231"></a><a name="p9441515231"></a><a href="Mins.md">Mins</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p18276162693419"><a name="p18276162693419"></a><a name="p18276162693419"></a>源操作数矢量内每个元素与标量相比，如果比标量大，则取标量值，比标量的值小，则取源操作数值。</p>
</td>
</tr>
<tr id="row547912419575"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p5445162310"><a name="p5445162310"></a><a name="p5445162310"></a><a href="LeakyRelu.md">LeakyRelu</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p27096582222"><a name="p27096582222"></a><a name="p27096582222"></a>按元素做带泄露线性整流Leaky ReLU。</p>
</td>
</tr>
<tr id="row64233134354"><td class="cellrowborder" rowspan="5" valign="top" width="15.590000000000002%" headers="mcps1.2.4.1.1 "><p id="p771519716593"><a name="p771519716593"></a><a name="p771519716593"></a>逻辑计算</p>
</td>
<td class="cellrowborder" valign="top" width="24.64%" headers="mcps1.2.4.1.2 "><p id="p24122327439"><a name="p24122327439"></a><a name="p24122327439"></a><a href="Not.md">Not</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.77%" headers="mcps1.2.4.1.3 "><p id="p1694421944312"><a name="p1694421944312"></a><a name="p1694421944312"></a>按元素做按位取反。</p>
</td>
</tr>
<tr id="row855712515365"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p512110487389"><a name="p512110487389"></a><a name="p512110487389"></a><a href="And.md">And</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p555732515368"><a name="p555732515368"></a><a name="p555732515368"></a>针对每对元素执行按位与运算。</p>
</td>
</tr>
<tr id="row19695152573617"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p12122548183814"><a name="p12122548183814"></a><a name="p12122548183814"></a><a href="Or.md">Or</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p17348124922015"><a name="p17348124922015"></a><a name="p17348124922015"></a>针对每对元素执行按位或运算。</p>
</td>
</tr>
<tr id="row20711575584"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p644313232"><a name="p644313232"></a><a name="p644313232"></a><a href="ShiftLeft.md">ShiftLeft</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1676331375210"><a name="p1676331375210"></a><a name="p1676331375210"></a>对源操作数中的每个元素进行左移操作，左移的位数由输入参数scalarValue决定。</p>
</td>
</tr>
<tr id="row17751175445811"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p9447119234"><a name="p9447119234"></a><a name="p9447119234"></a><a href="ShiftRight.md">ShiftRight</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p97121651191211"><a name="p97121651191211"></a><a name="p97121651191211"></a>对源操作数中的每个元素进行右移操作，右移的位数由输入参数scalarValue决定。</p>
</td>
</tr>
<tr id="row1413722319810"><td class="cellrowborder" rowspan="11" valign="top" width="15.590000000000002%" headers="mcps1.2.4.1.1 "><p id="p12137102320816"><a name="p12137102320816"></a><a name="p12137102320816"></a>复合计算</p>
</td>
<td class="cellrowborder" valign="top" width="24.64%" headers="mcps1.2.4.1.2 "><p id="p1885912584229"><a name="p1885912584229"></a><a name="p1885912584229"></a><a href="Axpy.md">Axpy</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.77%" headers="mcps1.2.4.1.3 "><p id="p1485935816222"><a name="p1485935816222"></a><a name="p1485935816222"></a>源操作数中每个元素与标量求积后和目的操作数中的对应元素相加。</p>
</td>
</tr>
<tr id="row167541318782"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p8577141674313"><a name="p8577141674313"></a><a name="p8577141674313"></a><a href="CastDeq.md">CastDeq</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1924211214434"><a name="p1924211214434"></a><a name="p1924211214434"></a>对输入做量化并进行精度转换。</p>
</td>
</tr>
<tr id="row15953122533617"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p19122184810389"><a name="p19122184810389"></a><a name="p19122184810389"></a><a href="AddRelu.md">AddRelu</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p39535252363"><a name="p39535252363"></a><a name="p39535252363"></a>按元素求和，结果和0对比取较大值。</p>
</td>
</tr>
<tr id="row16141192653616"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p112214815384"><a name="p112214815384"></a><a name="p112214815384"></a><a href="AddReluCast.md">AddReluCast</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p514172623613"><a name="p514172623613"></a><a name="p514172623613"></a>按元素求和，结果和0对比取较大值，并根据源操作数和目的操作数Tensor的数据类型进行精度转换。</p>
</td>
</tr>
<tr id="row977016176378"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1312214873810"><a name="p1312214873810"></a><a name="p1312214873810"></a><a href="AddDeqRelu.md">AddDeqRelu</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1477071717372"><a name="p1477071717372"></a><a name="p1477071717372"></a>依次计算按元素求和、结果进行deq量化后再进行relu计算（结果和0对比取较大值）。</p>
</td>
</tr>
<tr id="row17935161713713"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1012244843817"><a name="p1012244843817"></a><a name="p1012244843817"></a><a href="SubRelu.md">SubRelu</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p149363171378"><a name="p149363171378"></a><a name="p149363171378"></a>按元素求差，结果和0对比取较大值。</p>
</td>
</tr>
<tr id="row138841816370"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p51221848173810"><a name="p51221848173810"></a><a name="p51221848173810"></a><a href="SubReluCast.md">SubReluCast</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p158818183378"><a name="p158818183378"></a><a name="p158818183378"></a>按元素求差，结果和0对比取较大值，并根据源操作数和目的操作数Tensor的数据类型进行精度转换。</p>
</td>
</tr>
<tr id="row132241218193712"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p212217485389"><a name="p212217485389"></a><a name="p212217485389"></a><a href="MulAddDst.md">MulAddDst</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p5224201818378"><a name="p5224201818378"></a><a name="p5224201818378"></a>按元素将src0Local和src1Local相乘并和dstLocal相加，将最终结果存放进dstLocal中。</p>
</td>
</tr>
<tr id="row117041522154616"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p19705182219464"><a name="p19705182219464"></a><a name="p19705182219464"></a><a href="MulCast.md">MulCast</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p12705182217466"><a name="p12705182217466"></a><a name="p12705182217466"></a>按元素求积，并根据源操作数和目的操作数Tensor的数据类型进行精度转换。</p>
</td>
</tr>
<tr id="row193701818163720"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p91222048133815"><a name="p91222048133815"></a><a name="p91222048133815"></a><a href="FusedMulAdd.md">FusedMulAdd</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1037019189371"><a name="p1037019189371"></a><a name="p1037019189371"></a>按元素将src0Local和dstLocal相乘并加上src1Local，最终结果存放入dstLocal。</p>
</td>
</tr>
<tr id="row1075883983914"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p181228486382"><a name="p181228486382"></a><a name="p181228486382"></a><a href="FusedMulAddRelu.md">FusedMulAddRelu</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p475993916396"><a name="p475993916396"></a><a name="p475993916396"></a>按元素将src0Local和dstLocal相乘并加上src1Local，将结果和0作比较，取较大值，最终结果存放进dstLocal中。</p>
</td>
</tr>
<tr id="row187959172211"><td class="cellrowborder" rowspan="5" valign="top" width="15.590000000000002%" headers="mcps1.2.4.1.1 "><p id="p147155915229"><a name="p147155915229"></a><a name="p147155915229"></a>比较与选择</p>
<p id="p19980131874015"><a name="p19980131874015"></a><a name="p19980131874015"></a></p>
</td>
<td class="cellrowborder" valign="top" width="24.64%" headers="mcps1.2.4.1.2 "><p id="p157105910222"><a name="p157105910222"></a><a name="p157105910222"></a><a href="Compare.md">Compare</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.77%" headers="mcps1.2.4.1.3 "><p id="p1745910221"><a name="p1745910221"></a><a name="p1745910221"></a>逐元素比较两个tensor大小，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0。</p>
</td>
</tr>
<tr id="row1314575919229"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1314535915221"><a name="p1314535915221"></a><a name="p1314535915221"></a><a href="Compare（结果存入寄存器）.md">Compare（结果存放入寄存器）</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p177959415503"><a name="p177959415503"></a><a name="p177959415503"></a>逐元素比较两个tensor大小，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0。Compare接口需要mask参数时，可以使用此接口。计算结果存放入寄存器中。</p>
</td>
</tr>
<tr id="row31514186404"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p7151171817406"><a name="p7151171817406"></a><a name="p7151171817406"></a><a href="CompareScalar.md">CompareScalar</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p7227113135415"><a name="p7227113135415"></a><a name="p7227113135415"></a>逐元素比较一个tensor中的元素和另一个Scalar的大小，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0。</p>
</td>
</tr>
<tr id="row1798081810408"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p61901953124213"><a name="p61901953124213"></a><a name="p61901953124213"></a><a href="Select.md">Select</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p298081819406"><a name="p298081819406"></a><a name="p298081819406"></a>给定两个源操作数src0和src1，根据selMask（用于选择的Mask掩码）的比特位值选取元素，得到目的操作数dst。选择的规则为：当selMask的比特位是1时，从src0中选取，比特位是0时从src1选取。</p>
</td>
</tr>
<tr id="row1214111974012"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1719055394219"><a name="p1719055394219"></a><a name="p1719055394219"></a><a href="GatherMask.md">GatherMask</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p182166215152"><a name="p182166215152"></a><a name="p182166215152"></a>以内置固定模式对应的二进制或者用户自定义输入的Tensor数值对应的二进制为gather mask（数据收集的掩码），从源操作数中选取元素写入目的操作数中。</p>
</td>
</tr>
<tr id="row1095112164319"><td class="cellrowborder" valign="top" width="15.590000000000002%" headers="mcps1.2.4.1.1 "><p id="p1357761654319"><a name="p1357761654319"></a><a name="p1357761654319"></a>精度转换指令</p>
</td>
<td class="cellrowborder" valign="top" width="24.64%" headers="mcps1.2.4.1.2 "><p id="p757710162439"><a name="p757710162439"></a><a name="p757710162439"></a><a href="Cast.md">Cast</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.77%" headers="mcps1.2.4.1.3 "><p id="p5969213430"><a name="p5969213430"></a><a name="p5969213430"></a>根据源操作数和目的操作数Tensor的数据类型进行精度转换。</p>
</td>
</tr>
<tr id="row1651510211434"><td class="cellrowborder" rowspan="11" valign="top" width="15.590000000000002%" headers="mcps1.2.4.1.1 "><p id="p1757701610436"><a name="p1757701610436"></a><a name="p1757701610436"></a>归约计算</p>
</td>
<td class="cellrowborder" valign="top" width="24.64%" headers="mcps1.2.4.1.2 "><p id="p1357731618438"><a name="p1357731618438"></a><a name="p1357731618438"></a><a href="ReduceMax.md">ReduceMax</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.77%" headers="mcps1.2.4.1.3 "><p id="p1351515264314"><a name="p1351515264314"></a><a name="p1351515264314"></a>在所有的输入数据中找出最大值及最大值对应的索引位置。</p>
</td>
</tr>
<tr id="row196635294315"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p457871613433"><a name="p457871613433"></a><a name="p457871613433"></a><a href="ReduceMin.md">ReduceMin</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p6853520201916"><a name="p6853520201916"></a><a name="p6853520201916"></a>在所有的输入数据中找出最小值及最小值对应的索引位置。</p>
</td>
</tr>
<tr id="row77918244313"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p35781716164313"><a name="p35781716164313"></a><a name="p35781716164313"></a><a href="ReduceSum.md">ReduceSum</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1546312559583"><a name="p1546312559583"></a><a name="p1546312559583"></a>对所有的输入数据求和。</p>
</td>
</tr>
<tr id="row79421217439"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p65788161435"><a name="p65788161435"></a><a name="p65788161435"></a><a href="WholeReduceMax.md">WholeReduceMax</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p193962509594"><a name="p193962509594"></a><a name="p193962509594"></a>每个repeat内所有数据求最大值以及其索引index。</p>
</td>
</tr>
<tr id="row1961802720599"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p961972710592"><a name="p961972710592"></a><a name="p961972710592"></a><a href="WholeReduceMin.md">WholeReduceMin</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p0619192712592"><a name="p0619192712592"></a><a name="p0619192712592"></a>每个repeat内所有数据求最小值以及其索引index。</p>
</td>
</tr>
<tr id="row2928152965915"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1492892975917"><a name="p1492892975917"></a><a name="p1492892975917"></a><a href="WholeReduceSum.md">WholeReduceSum</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p199283290594"><a name="p199283290594"></a><a name="p199283290594"></a>每个repeat内所有数据求和。</p>
</td>
</tr>
<tr id="row157712324319"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p3578151614430"><a name="p3578151614430"></a><a name="p3578151614430"></a><a href="BlockReduceMax.md">BlockReduceMax</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p177391045819"><a name="p177391045819"></a><a name="p177391045819"></a>对每个repeat内所有元素求最大值。</p>
</td>
</tr>
<tr id="row17211133134315"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1057841619435"><a name="p1057841619435"></a><a name="p1057841619435"></a><a href="BlockReduceMin.md">BlockReduceMin</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p2021183144310"><a name="p2021183144310"></a><a name="p2021183144310"></a>对每个repeat内所有元素求最小值。</p>
</td>
</tr>
<tr id="row193401338432"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1557814168438"><a name="p1557814168438"></a><a name="p1557814168438"></a><a href="BlockReduceSum.md">BlockReduceSum</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p998114445293"><a name="p998114445293"></a><a name="p998114445293"></a>对每个repeat内所有元素求和。源操作数相加采用二叉树方式，两两相加。</p>
</td>
</tr>
<tr id="row250213184316"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p957811615437"><a name="p957811615437"></a><a name="p957811615437"></a><a href="PairReduceSum.md">PairReduceSum</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p6502183184316"><a name="p6502183184316"></a><a name="p6502183184316"></a>PairReduceSum：相邻两个（奇偶）元素求和。</p>
</td>
</tr>
<tr id="row5812182116350"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p198129212351"><a name="p198129212351"></a><a name="p198129212351"></a><a href="RepeatReduceSum.md">RepeatReduceSum</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1481215214350"><a name="p1481215214350"></a><a name="p1481215214350"></a>每个repeat内所有数据求和。和<a href="WholeReduceSum.md">WholeReduceSum</a>接口相比，不支持mask逐bit模式。建议使用功能更全面的<a href="WholeReduceSum.md">WholeReduceSum</a>接口。</p>
</td>
</tr>
<tr id="row109261892541"><td class="cellrowborder" rowspan="2" valign="top" width="15.590000000000002%" headers="mcps1.2.4.1.1 "><p id="p629093119547"><a name="p629093119547"></a><a name="p629093119547"></a>数据转换</p>
</td>
<td class="cellrowborder" valign="top" width="24.64%" headers="mcps1.2.4.1.2 "><p id="p42901731145414"><a name="p42901731145414"></a><a name="p42901731145414"></a><a href="Transpose.md">Transpose</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.77%" headers="mcps1.2.4.1.3 "><p id="p1429010318547"><a name="p1429010318547"></a><a name="p1429010318547"></a>可实现16*16的二维矩阵数据块的转置和[N,C,H,W]与[N,H,W,C]互相转换。</p>
</td>
</tr>
<tr id="row476616718548"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p629033110540"><a name="p629033110540"></a><a name="p629033110540"></a><a href="TransDataTo5HD.md">TransDataTo5HD</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p162901631125410"><a name="p162901631125410"></a><a name="p162901631125410"></a>数据格式转换，一般用于将NCHW格式转换成NC1HWC0格式。特别的，也可以用于二维矩阵数据块的转置。</p>
</td>
</tr>
<tr id="row86261631104316"><td class="cellrowborder" rowspan="3" valign="top" width="15.590000000000002%" headers="mcps1.2.4.1.1 "><p id="p203231339194317"><a name="p203231339194317"></a><a name="p203231339194317"></a>数据填充</p>
</td>
<td class="cellrowborder" valign="top" width="24.64%" headers="mcps1.2.4.1.2 "><p id="p432393920436"><a name="p432393920436"></a><a name="p432393920436"></a><a href="Duplicate.md">Duplicate</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.77%" headers="mcps1.2.4.1.3 "><p id="p9626431124314"><a name="p9626431124314"></a><a name="p9626431124314"></a>将一个变量或一个立即数，复制多次并填充到向量。</p>
</td>
</tr>
<tr id="row17641318431"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p232383912436"><a name="p232383912436"></a><a name="p232383912436"></a><a href="Brcb.md">Brcb</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p911mcpsimp"><a name="p911mcpsimp"></a><a name="p911mcpsimp"></a>给定一个输入张量，每一次取输入张量中的8个数填充到结果张量的8个datablock（32Bytes）中去，每个数对应一个datablock。</p>
</td>
</tr>
<tr id="row173409456360"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p24100384369"><a name="p24100384369"></a><a name="p24100384369"></a><a href="CreateVecIndex.md">CreateVecIndex</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p234064518368"><a name="p234064518368"></a><a name="p234064518368"></a>以firstValue为起始值创建向量索引。</p>
</td>
</tr>
<tr id="row161901532124317"><td class="cellrowborder" valign="top" width="15.590000000000002%" headers="mcps1.2.4.1.1 "><p id="p2323163915431"><a name="p2323163915431"></a><a name="p2323163915431"></a>数据分散/数据收集</p>
</td>
<td class="cellrowborder" valign="top" width="24.64%" headers="mcps1.2.4.1.2 "><p id="p1232318396431"><a name="p1232318396431"></a><a name="p1232318396431"></a><a href="Gather.md">Gather</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.77%" headers="mcps1.2.4.1.3 "><p id="p161901632134314"><a name="p161901632134314"></a><a name="p161901632134314"></a>给定输入的张量和一个地址偏移张量，Gather指令根据偏移地址将输入张量按元素收集到结果张量中。</p>
</td>
</tr>
<tr id="row16331932164312"><td class="cellrowborder" rowspan="4" valign="top" width="15.590000000000002%" headers="mcps1.2.4.1.1 "><p id="p1632373914312"><a name="p1632373914312"></a><a name="p1632373914312"></a>掩码操作</p>
</td>
<td class="cellrowborder" valign="top" width="24.64%" headers="mcps1.2.4.1.2 "><p id="p43241739154316"><a name="p43241739154316"></a><a name="p43241739154316"></a><a href="SetMaskCount.md">SetMaskCount</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.77%" headers="mcps1.2.4.1.3 "><p id="p9633133216430"><a name="p9633133216430"></a><a name="p9633133216430"></a>设置mask模式为Counter模式。该模式下，不需要开发者去感知迭代次数、处理非对齐的尾块等操作，可直接传入计算数据量，实际迭代次数由Vector计算单元自动推断。</p>
</td>
</tr>
<tr id="row1678513323437"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p15324339114312"><a name="p15324339114312"></a><a name="p15324339114312"></a><a href="SetMaskNorm.md">SetMaskNorm</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1785143218431"><a name="p1785143218431"></a><a name="p1785143218431"></a>设置mask模式为Normal模式。该模式为系统默认模式，支持开发者配置迭代次数。</p>
</td>
</tr>
<tr id="row1393713216436"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p23247397438"><a name="p23247397438"></a><a name="p23247397438"></a><a href="SetVectorMask.md">SetVectorMask</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p993873224313"><a name="p993873224313"></a><a name="p993873224313"></a>用于在矢量计算时设置mask。</p>
</td>
</tr>
<tr id="row15756339433"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p3324039144317"><a name="p3324039144317"></a><a name="p3324039144317"></a><a href="ResetMask.md">ResetMask</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p57533324312"><a name="p57533324312"></a><a name="p57533324312"></a>恢复mask的值为默认值（全1），表示矢量计算中每次迭代内的所有元素都将参与运算。</p>
</td>
</tr>
<tr id="row743516194407"><td class="cellrowborder" valign="top" width="15.590000000000002%" headers="mcps1.2.4.1.1 "><p id="p23240390438"><a name="p23240390438"></a><a name="p23240390438"></a>量化设置</p>
</td>
<td class="cellrowborder" valign="top" width="24.64%" headers="mcps1.2.4.1.2 "><p id="p10324173918436"><a name="p10324173918436"></a><a name="p10324173918436"></a><a href="SetDeqScale.md">SetDeqScale</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.77%" headers="mcps1.2.4.1.3 "><p id="p1238122924318"><a name="p1238122924318"></a><a name="p1238122924318"></a>设置DEQSCALE寄存器的值。</p>
</td>
</tr>
</tbody>
</table>

**表 4**  数据搬运API列表

<a name="table1199372172410"></a>
<table><thead align="left"><tr id="row69936217246"><th class="cellrowborder" valign="top" width="40.37%" id="mcps1.2.3.1.1"><p id="p1799422162414"><a name="p1799422162414"></a><a name="p1799422162414"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="59.63%" id="mcps1.2.3.1.2"><p id="p89941221202417"><a name="p89941221202417"></a><a name="p89941221202417"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row19994142132410"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p119488415242"><a name="p119488415242"></a><a name="p119488415242"></a><a href="DataCopy.md">DataCopy</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p1188623312246"><a name="p1188623312246"></a><a name="p1188623312246"></a>数据搬运接口，包括普通数据搬运、增强数据搬运、切片数据搬运、随路格式转换。</p>
</td>
</tr>
<tr id="row29942216241"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p138851338243"><a name="p138851338243"></a><a name="p138851338243"></a><a href="Copy.md">Copy</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p1488517337242"><a name="p1488517337242"></a><a name="p1488517337242"></a>VECIN、VECCALC、VECOUT之间的搬运指令，支持mask操作和<span id="ph1256166185416"><a name="ph1256166185416"></a><a name="ph1256166185416"></a>DataBlock</span>间隔操作。</p>
</td>
</tr>
</tbody>
</table>

**表 5**  资源管理API列表

<a name="table1267664316264"></a>
<table><thead align="left"><tr id="row15676154310267"><th class="cellrowborder" valign="top" width="40.37%" id="mcps1.2.3.1.1"><p id="p6676543192617"><a name="p6676543192617"></a><a name="p6676543192617"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="59.63%" id="mcps1.2.3.1.2"><p id="p146761434266"><a name="p146761434266"></a><a name="p146761434266"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row367664312619"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p667654312618"><a name="p667654312618"></a><a name="p667654312618"></a><a href="TPipe.md">TPipe</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p6428345172717"><a name="p6428345172717"></a><a name="p6428345172717"></a>TPipe是用来管理全局内存等资源的框架。通过TPipe类提供的接口可以完成内存等资源的分配管理操作。</p>
</td>
</tr>
<tr id="row1867604312262"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p616062319285"><a name="p616062319285"></a><a name="p616062319285"></a><a href="GetTPipePtr.md">GetTPipePtr</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p26771543172611"><a name="p26771543172611"></a><a name="p26771543172611"></a>获取框架当前管理全局内存的TPipe指针，用户获取指针后，可进行TPipe相关的操作。</p>
</td>
</tr>
<tr id="row1381344122816"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p681124418285"><a name="p681124418285"></a><a name="p681124418285"></a><a href="TBufPool.md">TBufPool</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p1062105494410"><a name="p1062105494410"></a><a name="p1062105494410"></a>TPipe可以管理全局内存资源，而TBufPool可以手动管理或复用<span id="ph1088254310583"><a name="ph1088254310583"></a><a name="ph1088254310583"></a>Unified Buffer</span>/<span id="ph1535518221316"><a name="ph1535518221316"></a><a name="ph1535518221316"></a>L1 Buffer</span>物理内存，主要用于多个stage计算中<span id="ph791414174415"><a name="ph791414174415"></a><a name="ph791414174415"></a>Unified Buffer</span>/<span id="ph7621654184416"><a name="ph7621654184416"></a><a name="ph7621654184416"></a>L1 Buffer</span>物理内存不足的场景。</p>
</td>
</tr>
<tr id="row1032714367308"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p73271836193011"><a name="p73271836193011"></a><a name="p73271836193011"></a><a href="TQue.md">TQue</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p13327163673017"><a name="p13327163673017"></a><a name="p13327163673017"></a>提供入队出队等接口，通过队列（Queue）完成任务间同步。</p>
</td>
</tr>
<tr id="row6569103614302"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p165691368309"><a name="p165691368309"></a><a name="p165691368309"></a><a href="TQueBind.md">TQueBind</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p9569203683010"><a name="p9569203683010"></a><a name="p9569203683010"></a>TQueBind绑定源逻辑位置和目的逻辑位置，根据源位置和目的位置，来确定内存分配的位置 、插入对应的同步事件，帮助开发者解决内存分配和管理、同步等问题。</p>
</td>
</tr>
<tr id="row106614368306"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p1866114362309"><a name="p1866114362309"></a><a name="p1866114362309"></a><a href="TBuf.md">TBuf</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p12661203618308"><a name="p12661203618308"></a><a name="p12661203618308"></a>使用<span id="ph438819594207"><a name="ph438819594207"></a><a name="ph438819594207"></a>Ascend C</span>编程的过程中，可能会用到一些临时变量。这些临时变量占用的内存可以使用TBuf数据结构来管理。</p>
</td>
</tr>
<tr id="row158561936163012"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p545354010463"><a name="p545354010463"></a><a name="p545354010463"></a><a href="InitSpmBuffer.md">InitSpmBuffer</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p16856103663012"><a name="p16856103663012"></a><a name="p16856103663012"></a>初始化SPM Buffer。</p>
</td>
</tr>
<tr id="row324173716309"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p1245314074616"><a name="p1245314074616"></a><a name="p1245314074616"></a><a href="WriteSpmBuffer.md">WriteSpmBuffer</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p17575330684"><a name="p17575330684"></a><a name="p17575330684"></a>将需要溢出暂存的数据拷贝到SPM Buffer中。</p>
</td>
</tr>
<tr id="row1515119377302"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p19453154018462"><a name="p19453154018462"></a><a name="p19453154018462"></a><a href="ReadSpmBuffer.md">ReadSpmBuffer</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p1881419176258"><a name="p1881419176258"></a><a name="p1881419176258"></a>从SPM Buffer读回到local数据中。</p>
</td>
</tr>
<tr id="row1550403614616"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p4453154014462"><a name="p4453154014462"></a><a name="p4453154014462"></a><a href="GetUserWorkspace.md">GetUserWorkspace</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p19504103614618"><a name="p19504103614618"></a><a name="p19504103614618"></a>获取用户使用的workspace指针。</p>
</td>
</tr>
<tr id="row15918536184616"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p54539406461"><a name="p54539406461"></a><a name="p54539406461"></a><a href="SetSysWorkSpace.md">SetSysWorkSpace</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p4918163654611"><a name="p4918163654611"></a><a name="p4918163654611"></a>在进行融合算子编程时，由于框架通信机制需要使用到workspace，也就是系统workspace，所以在该场景下，开发者要调用该接口，设置系统workspace的指针。</p>
</td>
</tr>
<tr id="row196011374465"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p2045304004615"><a name="p2045304004615"></a><a name="p2045304004615"></a><a href="GetSysWorkSpacePtr.md">GetSysWorkSpacePtr</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p86018377467"><a name="p86018377467"></a><a name="p86018377467"></a>获取系统workspace指针。</p>
</td>
</tr>
</tbody>
</table>

**表 6**  同步API列表

<a name="table921112251162"></a>
<table><thead align="left"><tr id="row975619311161"><th class="cellrowborder" valign="top" width="40.37%" id="mcps1.2.3.1.1"><p id="p8307249121617"><a name="p8307249121617"></a><a name="p8307249121617"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="59.63%" id="mcps1.2.3.1.2"><p id="p0308184961620"><a name="p0308184961620"></a><a name="p0308184961620"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1021262515169"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p62931833203117"><a name="p62931833203117"></a><a name="p62931833203117"></a><a href="TQueSync.md">TQueSync</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p82931433203116"><a name="p82931433203116"></a><a name="p82931433203116"></a>TQueSync类提供同步控制接口，开发者可以使用这类API来自行完成同步控制。</p>
</td>
</tr>
<tr id="row1821272513168"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p1190314213216"><a name="p1190314213216"></a><a name="p1190314213216"></a><a href="IBSet.md">IBSet</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p15692228145311"><a name="p15692228145311"></a><a name="p15692228145311"></a>当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。调用IBSet设置某一个核的标志位，与IBWait成对出现配合使用，表示核之间的同步等待指令，等待某一个核操作完成。</p>
</td>
</tr>
<tr id="row121219258161"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p13903164214324"><a name="p13903164214324"></a><a name="p13903164214324"></a><a href="IBWait.md">IBWait</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p27141049175319"><a name="p27141049175319"></a><a name="p27141049175319"></a>当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。IBWait与IBSet成对出现配合使用，表示核之间的同步等待指令，等待某一个核操作完成。</p>
</td>
</tr>
<tr id="row1121220254168"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p8904942143216"><a name="p8904942143216"></a><a name="p8904942143216"></a><a href="SyncAll.md">SyncAll</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p12782512105416"><a name="p12782512105416"></a><a name="p12782512105416"></a>当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。目前多核同步分为硬同步和软同步，硬件同步是利用硬件自带的全核同步指令由硬件保证多核同步，软件同步是使用软件算法模拟实现。</p>
</td>
</tr>
<tr id="row92121225191611"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p29094917345"><a name="p29094917345"></a><a name="p29094917345"></a><a href="InitDetermineComputeWorkspace.md">InitDetermineComputeWorkspace</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p9909129113414"><a name="p9909129113414"></a><a name="p9909129113414"></a>初始化GM共享内存的值，完成初始化后才可以调用<a href="WaitPreBlock.md">WaitPreBlock</a>和<a href="NotifyNextBlock.md">NotifyNextBlock</a>。</p>
</td>
</tr>
<tr id="row16213425171615"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p390910913341"><a name="p390910913341"></a><a name="p390910913341"></a><a href="WaitPreBlock.md">WaitPreBlock</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p49091792347"><a name="p49091792347"></a><a name="p49091792347"></a>通过读GM地址中的值，确认是否需要继续等待，当GM的值满足当前核的等待条件时，该核即可往下执行，进行下一步操作。</p>
</td>
</tr>
<tr id="row152131825131610"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p29091594341"><a name="p29091594341"></a><a name="p29091594341"></a><a href="NotifyNextBlock.md">NotifyNextBlock</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p6909899349"><a name="p6909899349"></a><a name="p6909899349"></a>通过写GM地址，通知下一个核当前核的操作已完成，下一个核可以进行操作。</p>
</td>
</tr>
<tr id="row52131925101618"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p39841259102515"><a name="p39841259102515"></a><a name="p39841259102515"></a><a href="SetNextTaskStart.md">SetNextTaskStart</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p12934317236"><a name="p12934317236"></a><a name="p12934317236"></a><span id="ph1882043515233"><a name="ph1882043515233"></a><a name="ph1882043515233"></a>在SuperKernel的子Kernel中调用，调用后的指令可以和后续其他的子Kernel实现并行，提升整体性能。</span></p>
</td>
</tr>
<tr id="row162135257162"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p288919718266"><a name="p288919718266"></a><a name="p288919718266"></a><a href="WaitPreTaskEnd.md">WaitPreTaskEnd</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p632496172313"><a name="p632496172313"></a><a name="p632496172313"></a><span id="ph1925517378244"><a name="ph1925517378244"></a><a name="ph1925517378244"></a>在SuperKernel的子Kernel中调用，调用前的指令可以和前序其他的子Kernel实现并行，提升整体性能。</span></p>
</td>
</tr>
</tbody>
</table>

**表 7**  缓存处理API列表

<a name="table5254131810573"></a>
<table><thead align="left"><tr id="row32541018135711"><th class="cellrowborder" valign="top" width="40.27%" id="mcps1.2.3.1.1"><p id="p17682103125810"><a name="p17682103125810"></a><a name="p17682103125810"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="59.730000000000004%" id="mcps1.2.3.1.2"><p id="p26828319585"><a name="p26828319585"></a><a name="p26828319585"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1325501835712"><td class="cellrowborder" valign="top" width="40.27%" headers="mcps1.2.3.1.1 "><p id="p83441336473"><a name="p83441336473"></a><a name="p83441336473"></a><a href="DataCachePreload.md">DataCachePreload</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.730000000000004%" headers="mcps1.2.3.1.2 "><p id="p18111100154719"><a name="p18111100154719"></a><a name="p18111100154719"></a>从源地址所在的特定DDR地址预加载数据到data cache中。</p>
</td>
</tr>
<tr id="row5255181815720"><td class="cellrowborder" valign="top" width="40.27%" headers="mcps1.2.3.1.1 "><p id="p1134473164714"><a name="p1134473164714"></a><a name="p1134473164714"></a><a href="DataCacheCleanAndInvalid.md">DataCacheCleanAndInvalid</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.730000000000004%" headers="mcps1.2.3.1.2 "><p id="p647301034415"><a name="p647301034415"></a><a name="p647301034415"></a>该接口用来刷新Cache，保证Cache的一致性。</p>
</td>
</tr>
</tbody>
</table>

**表 8**  系统变量访问API列表

<a name="table26716458301"></a>
<table><thead align="left"><tr id="row15672134518304"><th class="cellrowborder" valign="top" width="40.37%" id="mcps1.2.3.1.1"><p id="p18672144511306"><a name="p18672144511306"></a><a name="p18672144511306"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="59.63%" id="mcps1.2.3.1.2"><p id="p56726451306"><a name="p56726451306"></a><a name="p56726451306"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row967212454304"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p1194732015475"><a name="p1194732015475"></a><a name="p1194732015475"></a><a href="GetBlockNum.md">GetBlockNum</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p585381495015"><a name="p585381495015"></a><a name="p585381495015"></a>获取当前任务配置的Block数，用于代码内部的多核逻辑控制等。</p>
</td>
</tr>
<tr id="row1967224513011"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p11947620104710"><a name="p11947620104710"></a><a name="p11947620104710"></a><a href="GetBlockIdx.md">GetBlockIdx</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p54921825165017"><a name="p54921825165017"></a><a name="p54921825165017"></a>获取当前core的index，用于代码内部的多核逻辑控制及多核偏移量计算等。</p>
</td>
</tr>
<tr id="row11672104593014"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p17947182012472"><a name="p17947182012472"></a><a name="p17947182012472"></a><a href="GetDataBlockSizeInBytes.md">GetDataBlockSizeInBytes</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p14250113816503"><a name="p14250113816503"></a><a name="p14250113816503"></a>获取当前芯片版本一个datablock的大小，单位为byte。开发者根据datablock的大小来计算API指令中待传入的<span>repeatTime</span> 、<span id="ph06845043917"><a name="ph06845043917"></a><a name="ph06845043917"></a>DataBlock Stride</span>、<span id="ph13946527113916"><a name="ph13946527113916"></a><a name="ph13946527113916"></a>Repeat Stride</span><span>等</span>参数值。</p>
</td>
</tr>
<tr id="row3672445153011"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p1594715207476"><a name="p1594715207476"></a><a name="p1594715207476"></a><a href="GetArchVersion.md">GetArchVersion</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p764334516501"><a name="p764334516501"></a><a name="p764334516501"></a>获取当前AI处理器架构版本号。</p>
</td>
</tr>
</tbody>
</table>

**表 9**  原子操作接口列表

<a name="table1395854383210"></a>
<table><thead align="left"><tr id="row12958043173211"><th class="cellrowborder" valign="top" width="40.089999999999996%" id="mcps1.2.3.1.1"><p id="p2958104363212"><a name="p2958104363212"></a><a name="p2958104363212"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="59.91%" id="mcps1.2.3.1.2"><p id="p0958174353211"><a name="p0958174353211"></a><a name="p0958174353211"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1077235835018"><td class="cellrowborder" valign="top" width="40.089999999999996%" headers="mcps1.2.3.1.1 "><p id="p10729145920507"><a name="p10729145920507"></a><a name="p10729145920507"></a><a href="SetAtomicAdd.md">SetAtomicAdd</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.91%" headers="mcps1.2.3.1.2 "><p id="p20143125311516"><a name="p20143125311516"></a><a name="p20143125311516"></a>设置接下来从VECOUT到GM，L0C到GM，L1到GM的数据传输是否进行原子累加，可根据参数不同设定不同的累加数据类型。</p>
</td>
</tr>
<tr id="row139588436327"><td class="cellrowborder" valign="top" width="40.089999999999996%" headers="mcps1.2.3.1.1 "><p id="p1772985912505"><a name="p1772985912505"></a><a name="p1772985912505"></a><a href="SetAtomicType.md">SetAtomicType</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.91%" headers="mcps1.2.3.1.2 "><p id="p38701310125517"><a name="p38701310125517"></a><a name="p38701310125517"></a>通过设置模板参数来设定原子操作不同的数据类型。</p>
</td>
</tr>
<tr id="row595874313218"><td class="cellrowborder" valign="top" width="40.089999999999996%" headers="mcps1.2.3.1.1 "><p id="p773085925015"><a name="p773085925015"></a><a name="p773085925015"></a><a href="SetAtomicNone.md">SetAtomicNone</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.91%" headers="mcps1.2.3.1.2 "><p id="p1214271020615"><a name="p1214271020615"></a><a name="p1214271020615"></a>原子操作函数，清空原子操作的状态。</p>
</td>
</tr>
</tbody>
</table>

**表 10**  调试接口列表

<a name="table18295429109"></a>
<table><thead align="left"><tr id="row530154201012"><th class="cellrowborder" valign="top" width="37.71%" id="mcps1.2.3.1.1"><p id="p113014281016"><a name="p113014281016"></a><a name="p113014281016"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.29%" id="mcps1.2.3.1.2"><p id="p1230154221011"><a name="p1230154221011"></a><a name="p1230154221011"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row4220222125515"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p741723545516"><a name="p741723545516"></a><a name="p741723545516"></a><a href="DumpTensor.md">DumpTensor</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1541723514559"><a name="p1541723514559"></a><a name="p1541723514559"></a>基于算子工程开发的算子，可以使用该接口Dump指定Tensor的内容。</p>
</td>
</tr>
<tr id="row153018215552"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p144177356555"><a name="p144177356555"></a><a name="p144177356555"></a><a href="printf.md">printf</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1941711350553"><a name="p1941711350553"></a><a name="p1941711350553"></a>基于算子工程开发的算子，可以使用该接口实现CPU侧/NPU侧调试场景下的格式化输出功能。</p>
</td>
</tr>
<tr id="row18839203119378"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p17828375379"><a name="p17828375379"></a><a name="p17828375379"></a><a href="ascendc_assert.md">ascendc_assert</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p138203733710"><a name="p138203733710"></a><a name="p138203733710"></a><span id="ph5884122310382"><a name="ph5884122310382"></a><a name="ph5884122310382"></a>ascendc_assert<span>提供了一种在CPU/NPU域实现断言功能的接口。当断言条件不满足时，系统会输出断言信息并格式化打印在屏幕上。</span></span></p>
</td>
</tr>
<tr id="row8256721175511"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p34186352551"><a name="p34186352551"></a><a name="p34186352551"></a><a href="assert.md">assert</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1141883535519"><a name="p1141883535519"></a><a name="p1141883535519"></a>基于算子工程开发的算子，可以使用该接口实现CPU/NPU域assert断言功能。</p>
</td>
</tr>
<tr id="row820972155519"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p0418143555511"><a name="p0418143555511"></a><a name="p0418143555511"></a><a href="DumpAccChkPoint.md">DumpAccChkPoint</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p9418735175516"><a name="p9418735175516"></a><a name="p9418735175516"></a>基于算子工程开发的算子，可以使用该接口Dump指定Tensor的内容。该接口可以支持指定偏移位置的Tensor打印。</p>
</td>
</tr>
<tr id="row14208102016559"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p12418123512551"><a name="p12418123512551"></a><a name="p12418123512551"></a><a href="Trap.md">Trap</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1941853512557"><a name="p1941853512557"></a><a name="p1941853512557"></a>当软件产生异常后，使用该指令使kernel中止运行。</p>
</td>
</tr>
<tr id="row830164241016"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p13406234125916"><a name="p13406234125916"></a><a name="p13406234125916"></a><a href="GmAlloc.md">GmAlloc</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001963639306_zh-cn_topic_0000001541764188_p1531015912615"><a name="zh-cn_topic_0000001963639306_zh-cn_topic_0000001541764188_p1531015912615"></a><a name="zh-cn_topic_0000001963639306_zh-cn_topic_0000001541764188_p1531015912615"></a>进行核函数的CPU侧运行验证时，用于创建共享内存：在/tmp目录下创建一个共享文件，并返回该文件的映射指针。</p>
</td>
</tr>
<tr id="row88415865818"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p161386446595"><a name="p161386446595"></a><a name="p161386446595"></a><a href="ICPU_RUN_KF.md">ICPU_RUN_KF</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p6851188585"><a name="p6851188585"></a><a name="p6851188585"></a>进行核函数的CPU侧运行验证时，CPU调测总入口，完成CPU侧的算子程序调用。</p>
</td>
</tr>
<tr id="row7702750145910"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p19702850185918"><a name="p19702850185918"></a><a name="p19702850185918"></a><a href="ICPU_SET_TILING_KEY.md">ICPU_SET_TILING_KEY</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p10702750175919"><a name="p10702750175919"></a><a name="p10702750175919"></a>用于指定本次CPU调测使用的tilingKey。调测执行时，将只执行算子核函数中该tilingKey对应的分支。</p>
</td>
</tr>
<tr id="row10114145135910"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p31141451185911"><a name="p31141451185911"></a><a name="p31141451185911"></a><a href="GmFree.md">GmFree</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p311495195915"><a name="p311495195915"></a><a name="p311495195915"></a>进行核函数的CPU侧运行验证时，用于释放通过GmAlloc申请的共享内存。</p>
</td>
</tr>
<tr id="row1125615511592"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p16416351306"><a name="p16416351306"></a><a name="p16416351306"></a><a href="SetKernelMode.md">SetKernelMode</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_p1118165416116"><a name="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_p1118165416116"></a><a name="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_p1118165416116"></a>CPU调测时，设置内核模式为单AIV模式，单AIC模式或者MIX模式，以分别支持单AIV矢量算子，单AIC矩阵算子，MIX混合算子的CPU调试。</p>
</td>
</tr>
<tr id="row5399851115914"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p18391515913"><a name="p18391515913"></a><a name="p18391515913"></a><a href="TRACE_START.md">TRACE_START</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p133991451205919"><a name="p133991451205919"></a><a name="p133991451205919"></a>通过CAModel进行算子性能仿真时，可对算子任意运行阶段打点，从而分析不同指令的流水图，以便进一步性能调优。</p>
<p id="p3807143716119"><a name="p3807143716119"></a><a name="p3807143716119"></a>用于表示起始位置打点，一般与<a href="TRACE_STOP.md">TRACE_STOP</a>配套使用。</p>
</td>
</tr>
<tr id="row125261851195913"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p66051550125"><a name="p66051550125"></a><a name="p66051550125"></a><a href="TRACE_STOP.md">TRACE_STOP</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002000199857_p15636113710169"><a name="zh-cn_topic_0000002000199857_p15636113710169"></a><a name="zh-cn_topic_0000002000199857_p15636113710169"></a>通过CAModel进行算子性能仿真时，可对算子任意运行阶段打点，从而分析不同指令的流水图，以便进一步性能调优。</p>
<p id="zh-cn_topic_0000002000199857_p868234414164"><a name="zh-cn_topic_0000002000199857_p868234414164"></a><a name="zh-cn_topic_0000002000199857_p868234414164"></a>用于表示终止位置打点，一般与<a href="TRACE_START.md">TRACE_START</a>配套使用。</p>
</td>
</tr>
<tr id="row368519581124"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1468611581226"><a name="p1468611581226"></a><a name="p1468611581226"></a><a href="MetricsProfStart.md">MetricsProfStart</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p16869582024"><a name="p16869582024"></a><a name="p16869582024"></a>用于设置性能数据采集信号启动，和MetricsProfStop配合使用。使用工具进行算子上板调优时，可在kernel侧代码段前后分别调用MetricsProfStart和MetricsProfStop来指定需要调优的代码段范围。</p>
</td>
</tr>
<tr id="row2143959125"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p4507261733"><a name="p4507261733"></a><a name="p4507261733"></a><a href="MetricsProfStop.md">MetricsProfStop</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002000280001_zh-cn_topic_0000001960477980_p279125418620"><a name="zh-cn_topic_0000002000280001_zh-cn_topic_0000001960477980_p279125418620"></a><a name="zh-cn_topic_0000002000280001_zh-cn_topic_0000001960477980_p279125418620"></a>设置性能数据采集信号停止，和MetricsProfStart配合使用。使用工具进行算子上板调优时，可在kernel侧代码段前后分别调用MetricsProfStart和MetricsProfStop来指定需要调优的代码段范围。</p>
</td>
</tr>
</tbody>
</table>

**表 11**  工具类接口列表

<a name="table9496143191816"></a>
<table><thead align="left"><tr id="row14962043181812"><th class="cellrowborder" valign="top" width="40.089999999999996%" id="mcps1.2.3.1.1"><p id="p449784310180"><a name="p449784310180"></a><a name="p449784310180"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="59.91%" id="mcps1.2.3.1.2"><p id="p17497204317180"><a name="p17497204317180"></a><a name="p17497204317180"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row649710437184"><td class="cellrowborder" valign="top" width="40.089999999999996%" headers="mcps1.2.3.1.1 "><p id="p149744319183"><a name="p149744319183"></a><a name="p149744319183"></a><a href="Async.md">Async</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.91%" headers="mcps1.2.3.1.2 "><p id="p816712203211"><a name="p816712203211"></a><a name="p816712203211"></a><span id="ph585624314195"><a name="ph585624314195"></a><a name="ph585624314195"></a>Async提供了一个统一的接口，用于在不同模式下（AIC或AIV）执行特定函数，从而避免代码中直接的硬件条件判断（如使用ASCEND_IS_AIV或ASCEND_IS_AIC）。</span></p>
</td>
</tr>
<tr id="row154197811816"><td class="cellrowborder" valign="top" width="40.089999999999996%" headers="mcps1.2.3.1.1 "><p id="p2094712054713"><a name="p2094712054713"></a><a name="p2094712054713"></a><a href="GetTaskRatio.md">GetTaskRatio</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.91%" headers="mcps1.2.3.1.2 "><p id="p29729544501"><a name="p29729544501"></a><a name="p29729544501"></a>适用于Cube/Vector分离模式，用来获取Cube/Vector的配比。</p>
</td>
</tr>
</tbody>
</table>

**表 12**  Kernel Tiling接口列表

<a name="table2017815711517"></a>
<table><thead align="left"><tr id="row11179357358"><th class="cellrowborder" valign="top" width="39.900000000000006%" id="mcps1.2.3.1.1"><p id="p417975719517"><a name="p417975719517"></a><a name="p417975719517"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60.099999999999994%" id="mcps1.2.3.1.2"><p id="p1817910573513"><a name="p1817910573513"></a><a name="p1817910573513"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row14180457256"><td class="cellrowborder" valign="top" width="39.900000000000006%" headers="mcps1.2.3.1.1 "><p id="p1518025711512"><a name="p1518025711512"></a><a name="p1518025711512"></a><a href="GET_TILING_DATA.md">GET_TILING_DATA</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.099999999999994%" headers="mcps1.2.3.1.2 "><p id="p818095714515"><a name="p818095714515"></a><a name="p818095714515"></a>用于获取算子kernel入口函数传入的tiling信息，并填入注册的Tiling结构体中，此函数会以宏展开的方式进行编译。如果用户注册了多个TilingData结构体，使用该接口返回默认注册的结构体。</p>
</td>
</tr>
<tr id="row165593514011"><td class="cellrowborder" valign="top" width="39.900000000000006%" headers="mcps1.2.3.1.1 "><p id="p155603514401"><a name="p155603514401"></a><a name="p155603514401"></a><a href="GET_TILING_DATA_WITH_STRUCT.md">GET_TILING_DATA_WITH_STRUCT</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.099999999999994%" headers="mcps1.2.3.1.2 "><p id="p10569356405"><a name="p10569356405"></a><a name="p10569356405"></a>使用该接口指定结构体名称，可获取指定的tiling信息，并填入对应的Tiling结构体中，此函数会以宏展开的方式进行编译。</p>
</td>
</tr>
<tr id="row146553974019"><td class="cellrowborder" valign="top" width="39.900000000000006%" headers="mcps1.2.3.1.1 "><p id="p246512399407"><a name="p246512399407"></a><a name="p246512399407"></a><a href="GET_TILING_DATA_MEMBER.md">GET_TILING_DATA_MEMBER</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.099999999999994%" headers="mcps1.2.3.1.2 "><p id="p18466133917406"><a name="p18466133917406"></a><a name="p18466133917406"></a>用于获取tiling结构体的成员变量。</p>
</td>
</tr>
<tr id="row818045715513"><td class="cellrowborder" valign="top" width="39.900000000000006%" headers="mcps1.2.3.1.1 "><p id="p1318010571658"><a name="p1318010571658"></a><a name="p1318010571658"></a><a href="TILING_KEY_IS.md">TILING_KEY_IS</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.099999999999994%" headers="mcps1.2.3.1.2 "><p id="p41805570511"><a name="p41805570511"></a><a name="p41805570511"></a>在核函数中判断本次执行时的tiling_key是否等于某个key，从而标识tiling_key==key的一条kernel分支。</p>
</td>
</tr>
<tr id="row930185314421"><td class="cellrowborder" valign="top" width="39.900000000000006%" headers="mcps1.2.3.1.1 "><p id="p1830185311424"><a name="p1830185311424"></a><a name="p1830185311424"></a><a href="REGISTER_TILING_DEFAULT.md">REGISTER_TILING_DEFAULT</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.099999999999994%" headers="mcps1.2.3.1.2 "><p id="p63075319421"><a name="p63075319421"></a><a name="p63075319421"></a>用于在kernel侧注册用户使用标准C++语法自定义的默认TilingData结构体。</p>
</td>
</tr>
<tr id="row2047417552427"><td class="cellrowborder" valign="top" width="39.900000000000006%" headers="mcps1.2.3.1.1 "><p id="p1147415594219"><a name="p1147415594219"></a><a name="p1147415594219"></a><a href="REGISTER_TILING_FOR_TILINGKEY.md">REGISTER_TILING_FOR_TILINGKEY</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.099999999999994%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001526206862_p0824144014589"><a name="zh-cn_topic_0000001526206862_p0824144014589"></a><a name="zh-cn_topic_0000001526206862_p0824144014589"></a>用于在kernel侧注册与TilingKey相匹配的TilingData自定义结构体；该接口需提供一个逻辑表达式，逻辑表达式以字符串“TILING_KEY_VAR”代指实际TilingKey，表达TIlingKey所满足的范围。</p>
</td>
</tr>
<tr id="row8142114933417"><td class="cellrowborder" valign="top" width="39.900000000000006%" headers="mcps1.2.3.1.1 "><p id="p187745415347"><a name="p187745415347"></a><a name="p187745415347"></a><a href="REGISTER_NONE_TILING.md">REGISTER_NONE_TILING</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.099999999999994%" headers="mcps1.2.3.1.2 "><p id="p1131075619124"><a name="p1131075619124"></a><a name="p1131075619124"></a><span>在Kernel侧使用标准C++语法自定义的TilingData结构体时，若用户不确定需要注册哪些结构体，可使用该接口</span>告知框架侧需使用未注册的标准C++语法来定义TilingData，并配套<a href="GET_TILING_DATA_WITH_STRUCT.md">GET_TILING_DATA_WITH_STRUCT</a>，<a href="GET_TILING_DATA_MEMBER.md">GET_TILING_DATA_MEMBER</a>，<a href="GET_TILING_DATA_PTR_WITH_STRUCT.md">GET_TILING_DATA_PTR_WITH_STRUCT</a>来获取对应的TilingData。</p>
</td>
</tr>
<tr id="row131802571959"><td class="cellrowborder" valign="top" width="39.900000000000006%" headers="mcps1.2.3.1.1 "><p id="p51807572055"><a name="p51807572055"></a><a name="p51807572055"></a><a href="设置Kernel类型.md">KERNEL_TASK_TYPE_DEFAULT</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.099999999999994%" headers="mcps1.2.3.1.2 "><p id="p1618045718520"><a name="p1618045718520"></a><a name="p1618045718520"></a>设置全局默认的kernel type，对所有的tiling key生效。</p>
</td>
</tr>
<tr id="row21801457154"><td class="cellrowborder" valign="top" width="39.900000000000006%" headers="mcps1.2.3.1.1 "><p id="p0180185713514"><a name="p0180185713514"></a><a name="p0180185713514"></a><a href="设置Kernel类型.md">KERNEL_TASK_TYPE</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.099999999999994%" headers="mcps1.2.3.1.2 "><p id="p201806571756"><a name="p201806571756"></a><a name="p201806571756"></a>设置某一个具体的tiling key对应的kernel type。</p>
</td>
</tr>
</tbody>
</table>

**表 13**  ISASI接口列表

<a name="table19526741203211"></a>
<table><thead align="left"><tr id="row352624118322"><th class="cellrowborder" valign="top" width="12.379999999999999%" id="mcps1.2.4.1.1"><p id="p88065174816"><a name="p88065174816"></a><a name="p88065174816"></a>分类</p>
</th>
<th class="cellrowborder" valign="top" width="27.63%" id="mcps1.2.4.1.2"><p id="p14526241173218"><a name="p14526241173218"></a><a name="p14526241173218"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="59.99%" id="mcps1.2.4.1.3"><p id="p145261141203210"><a name="p145261141203210"></a><a name="p145261141203210"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row154401144114718"><td class="cellrowborder" rowspan="15" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.1 "><p id="p12441154416479"><a name="p12441154416479"></a><a name="p12441154416479"></a>矢量计算</p>
</td>
<td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.2 "><p id="p20441544114710"><a name="p20441544114710"></a><a name="p20441544114710"></a><a href="VectorPadding(ISASI).md">VectorPadding</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.99%" headers="mcps1.2.4.1.3 "><p id="p244134434718"><a name="p244134434718"></a><a name="p244134434718"></a>根据padMode（pad模式）与padSide（pad方向）对源操作数按照datablock进行填充操作。</p>
</td>
</tr>
<tr id="row10526441173211"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1994405814488"><a name="p1994405814488"></a><a name="p1994405814488"></a><a href="BilinearInterpolation(ISASI).md">BilinearInterpolation</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1526144123212"><a name="p1526144123212"></a><a name="p1526144123212"></a><span>双线性插值操作，分为垂直迭代和水平迭代。</span></p>
</td>
</tr>
<tr id="row1952734143214"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p99449586481"><a name="p99449586481"></a><a name="p99449586481"></a><a href="GetCmpMask(ISASI).md">GetCmpMask</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1152720414327"><a name="p1152720414327"></a><a name="p1152720414327"></a>获取<a href="Compare（结果存入寄存器）.md">Compare（结果存入寄存器）</a>指令的比较结果。</p>
</td>
</tr>
<tr id="row946564134915"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p5465104110496"><a name="p5465104110496"></a><a name="p5465104110496"></a><a href="SetCmpMask(ISASI).md">SetCmpMask</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p184655418496"><a name="p184655418496"></a><a name="p184655418496"></a>为<a href="Select.md">Select</a>不传入mask参数的接口设置比较寄存器。</p>
</td>
</tr>
<tr id="row1952774119326"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p194435804818"><a name="p194435804818"></a><a name="p194435804818"></a><a href="GetAccVal(ISASI).md">GetAccVal</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p165272418326"><a name="p165272418326"></a><a name="p165272418326"></a>获取<a href="ReduceSum.md">ReduceSum</a>（针对tensor前n个数据计算）接口的计算结果。</p>
</td>
</tr>
<tr id="row125271041103210"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p294415583488"><a name="p294415583488"></a><a name="p294415583488"></a><a href="GetReduceMaxMinCount(ISASI).md">GetReduceMaxMinCount</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p195452267225"><a name="p195452267225"></a><a name="p195452267225"></a>获取<a href="ReduceMax.md">ReduceMax</a>、<a href="ReduceMin.md">ReduceMin</a>连续场景下的最大/最小值以及相应的索引值。</p>
</td>
</tr>
<tr id="row185271941123212"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p169441258134818"><a name="p169441258134818"></a><a name="p169441258134818"></a><a href="ProposalConcat.md">ProposalConcat</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p14527241173218"><a name="p14527241173218"></a><a name="p14527241173218"></a>将连续元素合入Region Proposal内对应位置，每次迭代会将16个连续元素合入到16个Region Proposals的对应位置里。</p>
</td>
</tr>
<tr id="row13527134173211"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p39441158184816"><a name="p39441158184816"></a><a name="p39441158184816"></a><a href="ProposalExtract.md">ProposalExtract</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p10462291478"><a name="p10462291478"></a><a name="p10462291478"></a>与ProposalConcat功能相反，从Region Proposals内将相应位置的单个元素抽取后重排，每次迭代处理16个Region Proposals，抽取16个元素后连续排列。</p>
</td>
</tr>
<tr id="row452716417321"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p6944165817489"><a name="p6944165817489"></a><a name="p6944165817489"></a><a href="RpSort16.md">RpSort16</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p773018265170"><a name="p773018265170"></a><a name="p773018265170"></a>根据Region Proposals中的score域对其进行排序（score大的排前面），每次排16个Region Proposals。</p>
</td>
</tr>
<tr id="row745884215480"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p139441358114819"><a name="p139441358114819"></a><a name="p139441358114819"></a><a href="MrgSort4.md">MrgSort4</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p101781237141713"><a name="p101781237141713"></a><a name="p101781237141713"></a>将已经排好序的最多4 条region proposals队列，排列并合并成1条队列，结果按照score域由大到小排序。</p>
</td>
</tr>
<tr id="row1019917563485"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p594585811484"><a name="p594585811484"></a><a name="p594585811484"></a><a href="Sort32.md">Sort32</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p151991556204813"><a name="p151991556204813"></a><a name="p151991556204813"></a>排序函数，一次迭代可以完成32个数的排序。</p>
</td>
</tr>
<tr id="row18337125610483"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p10945175864811"><a name="p10945175864811"></a><a name="p10945175864811"></a><a href="MrgSort.md">MrgSort</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p78578171182"><a name="p78578171182"></a><a name="p78578171182"></a>将已经排好序的最多4 条队列，合并排列成 1 条队列，结果按照score域由大到小排序。</p>
</td>
</tr>
<tr id="row8465195684815"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p139456584485"><a name="p139456584485"></a><a name="p139456584485"></a><a href="GetMrgSortResult.md">GetMrgSortResult</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p101531828111815"><a name="p101531828111815"></a><a name="p101531828111815"></a>获取<a href="MrgSort.md">MrgSort</a>或<a href="MrgSort4.md">MrgSort4</a>已经处理过的队列里的Region Proposal个数，并依次存储在四个List入参中。</p>
</td>
</tr>
<tr id="row46098568481"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1794575816483"><a name="p1794575816483"></a><a name="p1794575816483"></a><a href="Gatherb(ISASI).md">Gatherb</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p4609556174812"><a name="p4609556174812"></a><a name="p4609556174812"></a>给定一个输入的张量和一个地址偏移张量，Gatherb指令根据偏移地址将输入张量收集到结果张量中。</p>
</td>
</tr>
<tr id="row576405615486"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1945158154815"><a name="p1945158154815"></a><a name="p1945158154815"></a><a href="Scatter(ISASI).md">Scatter</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1836365951815"><a name="p1836365951815"></a><a name="p1836365951815"></a>给定一个连续的输入张量和一个目的地址偏移张量，Scatter指令根据偏移地址生成新的结果张量后将输入张量分散到结果张量中。</p>
</td>
</tr>
<tr id="row224111331412"><td class="cellrowborder" rowspan="2" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.1 "><p id="p711814301143"><a name="p711814301143"></a><a name="p711814301143"></a>数据搬运</p>
</td>
<td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.2 "><p id="p148821245184915"><a name="p148821245184915"></a><a name="p148821245184915"></a><a href="DataCopyPad(ISASI).md">DataCopyPad</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.99%" headers="mcps1.2.4.1.3 "><p id="p019795716484"><a name="p019795716484"></a><a name="p019795716484"></a>该接口提供数据非对齐搬运的功能。</p>
</td>
</tr>
<tr id="row7375515151412"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p148821245194911"><a name="p148821245194911"></a><a name="p148821245194911"></a><a href="SetPadValue(ISASI).md">SetPadValue</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p6689192111598"><a name="p6689192111598"></a><a name="p6689192111598"></a>设置DataCopyPad接口填充的数值。</p>
</td>
</tr>
<tr id="row154051221164915"><td class="cellrowborder" rowspan="25" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.1 "><p id="p118921556154813"><a name="p118921556154813"></a><a name="p118921556154813"></a>矩阵计算</p>
</td>
<td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.2 "><p id="p1255212302496"><a name="p1255212302496"></a><a name="p1255212302496"></a><a href="Mmad.md">Mmad</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.99%" headers="mcps1.2.4.1.3 "><p id="p24861371569"><a name="p24861371569"></a><a name="p24861371569"></a>完成矩阵乘加操作。</p>
</td>
</tr>
<tr id="row12765526165715"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1176511267573"><a name="p1176511267573"></a><a name="p1176511267573"></a><a href="MmadWithSparse.md">MmadWithSparse</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p47651826105712"><a name="p47651826105712"></a><a name="p47651826105712"></a>完成矩阵乘加操作，传入的左矩阵A为稀疏矩阵， 右矩阵B为稠密矩阵 。</p>
</td>
</tr>
<tr id="row0435522154918"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p8552203094910"><a name="p8552203094910"></a><a name="p8552203094910"></a><a href="SetHF32Mode.md">SetHF32Mode</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p64161631122120"><a name="p64161631122120"></a><a name="p64161631122120"></a>此接口同<a href="SetHF32TransMode.md">SetHF32TransMode</a>与<a href="SetMMLayoutTransform.md">SetMMLayoutTransform</a>一样，都用于设置寄存器的值。SetHF32Mode接口用于设置MMAD的HF32模式。</p>
</td>
</tr>
<tr id="row1858982214918"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p165526308490"><a name="p165526308490"></a><a name="p165526308490"></a><a href="SetHF32TransMode.md">SetHF32TransMode</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1478477185019"><a name="p1478477185019"></a><a name="p1478477185019"></a>此接口同<a href="SetHF32Mode.md">SetHF32Mode</a>与<a href="SetMMLayoutTransform.md">SetMMLayoutTransform</a>一样，都用于设置寄存器的值。SetHF32TransMode用于设置MMAD的HF32取整模式，仅在MMAD的HF32模式生效时有效。</p>
</td>
</tr>
<tr id="row1374212216499"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p145521430154913"><a name="p145521430154913"></a><a name="p145521430154913"></a><a href="SetMMLayoutTransform.md">SetMMLayoutTransform</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p87421422144917"><a name="p87421422144917"></a><a name="p87421422144917"></a>此接口同<a href="SetHF32Mode.md">SetHF32Mode</a>与<a href="SetHF32TransMode.md">SetHF32TransMode</a>一样，都用于设置寄存器的值，其中SetMMLayoutTransform接口用于设置MMAD的M/N方向。</p>
</td>
</tr>
<tr id="row19411232491"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p20552113020499"><a name="p20552113020499"></a><a name="p20552113020499"></a><a href="Conv2D（废弃）.md">Conv2D</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p10261152153414"><a name="p10261152153414"></a><a name="p10261152153414"></a>计算给定输入张量和权重张量的2-D卷积，输出结果张量。Conv2d卷积层多用于图像识别，使用过滤器提取图像中的特征。</p>
</td>
</tr>
<tr id="row144112573483"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p9552930204920"><a name="p9552930204920"></a><a name="p9552930204920"></a><a href="Gemm（废弃）.md">Gemm</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p14116577483"><a name="p14116577483"></a><a name="p14116577483"></a>根据输入的切分规则，将给定的两个输入张量做矩阵乘，输出至结果张量。将A和B两个输入矩阵乘法在一起，得到一个输出矩阵C。</p>
</td>
</tr>
<tr id="row147451115151916"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p45523300491"><a name="p45523300491"></a><a name="p45523300491"></a><a href="SetFixPipeConfig(ISASI).md">SetFixPipeConfig</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p17851154862112"><a name="p17851154862112"></a><a name="p17851154862112"></a><a href="随路量化激活搬运.md">DataCopy</a>（CO1-&gt;GM、CO1-&gt;A1）过程中进行随路量化时，通过调用该接口设置量化流程中tensor量化参数。</p>
</td>
</tr>
<tr id="row294174016209"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p65522030174916"><a name="p65522030174916"></a><a name="p65522030174916"></a><a href="SetFixpipeNz2ndFlag(ISASI).md">SetFixpipeNz2ndFlag</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p2898124417263"><a name="p2898124417263"></a><a name="p2898124417263"></a><a href="随路量化激活搬运.md">DataCopy</a>（CO1-&gt;GM、CO1-&gt;A1）过程中进行随路格式转换（NZ2ND）时，通过调用该接口设置NZ2ND相关配置。</p>
</td>
</tr>
<tr id="row11129104162017"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p165521930144913"><a name="p165521930144913"></a><a name="p165521930144913"></a><a href="SetFixpipePreQuantFlag(ISASI).md">SetFixpipePreQuantFlag</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p16332111515234"><a name="p16332111515234"></a><a name="p16332111515234"></a><a href="随路量化激活搬运.md">DataCopy</a>（CO1-&gt;GM、CO1-&gt;A1）过程中进行随路量化时，通过调用该接口设置量化流程中scalar量化参数。</p>
</td>
</tr>
<tr id="row19331114117200"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p278152219274"><a name="p278152219274"></a><a name="p278152219274"></a><a href="SetFixPipeClipRelu(ISASI).md">SetFixPipeClipRelu</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p10787226279"><a name="p10787226279"></a><a name="p10787226279"></a><a href="随路量化激活搬运.md">DataCopy</a>（CO1-&gt;GM）过程中进行随路量化后，通过调用该接口设置ClipRelu操作的最大值。</p>
</td>
</tr>
<tr id="row0541441152019"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1674634112715"><a name="p1674634112715"></a><a name="p1674634112715"></a><a href="SetFixPipeAddr(ISASI).md">SetFixPipeAddr</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p71911024132712"><a name="p71911024132712"></a><a name="p71911024132712"></a><a href="随路量化激活搬运.md">DataCopy</a>（CO1-&gt;GM）过程中进行随路量化后，通过调用该接口设置element-wise操作时LocalTensor的地址。</p>
</td>
</tr>
<tr id="row08411461812"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1955143017499"><a name="p1955143017499"></a><a name="p1955143017499"></a><a href="InitConstValue(ISASI).md">InitConstValue</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1489215634815"><a name="p1489215634815"></a><a name="p1489215634815"></a>初始化LocalTensor（TPosition为A1/A2/B1/B2）为某一个具体的数值。</p>
</td>
</tr>
<tr id="row696114470313"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p455133074917"><a name="p455133074917"></a><a name="p455133074917"></a><a href="LoadData(ISASI).md">LoadData</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p3448142813492"><a name="p3448142813492"></a><a name="p3448142813492"></a>LoadData包括Load2D和Load3D数据加载功能。</p>
</td>
</tr>
<tr id="row8181746635"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p20551143074919"><a name="p20551143074919"></a><a name="p20551143074919"></a><a href="LoadDataWithTranspose(ISASI).md">LoadDataWithTranspose</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p124281245667"><a name="p124281245667"></a><a name="p124281245667"></a>该接口实现带转置的2D格式数据从A1/B1到A2/B2的加载。</p>
</td>
</tr>
<tr id="row194791339636"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p12211416165318"><a name="p12211416165318"></a><a name="p12211416165318"></a><a href="SetAippFunctions(ISASI).md">SetAippFunctions</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p192129164538"><a name="p192129164538"></a><a name="p192129164538"></a>设置图片预处理（AIPP，AI core pre-process）相关参数。</p>
</td>
</tr>
<tr id="row1948463715319"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1061103015541"><a name="p1061103015541"></a><a name="p1061103015541"></a><a href="LoadImageToLocal(ISASI).md">LoadImageToLocal</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p792614613548"><a name="p792614613548"></a><a name="p792614613548"></a>将图像数据从GM搬运到A1/B1。 搬运过程中可以完成图像预处理操作：包括图像翻转，改变图像尺寸（抠图，裁边，缩放，伸展），以及色域转换，类型转换等。</p>
</td>
</tr>
<tr id="row0248535339"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p16995181110545"><a name="p16995181110545"></a><a name="p16995181110545"></a><a href="LoadUnzipIndex(ISASI).md">LoadUnZipIndex</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p20995111113541"><a name="p20995111113541"></a><a name="p20995111113541"></a>加载GM上的压缩索引表到内部寄存器。</p>
</td>
</tr>
<tr id="row132838331033"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1434981445411"><a name="p1434981445411"></a><a name="p1434981445411"></a><a href="LoadDataUnzip(ISASI).md">LoadDataUnzip</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p2349121415547"><a name="p2349121415547"></a><a name="p2349121415547"></a>将GM上的数据解压并搬运到A1/B1/B2上。</p>
</td>
</tr>
<tr id="row7838730733"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1418115368566"><a name="p1418115368566"></a><a name="p1418115368566"></a><a href="LoadDataWithSparse(ISASI).md">LoadDataWithSparse</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p7263112805617"><a name="p7263112805617"></a><a name="p7263112805617"></a>用于搬运存放在B1里的512B的稠密权重矩阵到B2里，同时读取128B的索引矩阵用于稠密矩阵的稀疏化。</p>
</td>
</tr>
<tr id="row66591928736"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1552163012493"><a name="p1552163012493"></a><a name="p1552163012493"></a><a href="SetFmatrix(ISASI).md">SetFmatrix</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1365632015497"><a name="p1365632015497"></a><a name="p1365632015497"></a>用于调用<a href="LoadData(ISASI).md">Load3Dv1/Load3Dv2</a>时设置FeatureMap的属性描述。</p>
</td>
</tr>
<tr id="row07692515218"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p055233018498"><a name="p055233018498"></a><a name="p055233018498"></a><a href="SetLoadDataBoundary(ISASI).md">SetLoadDataBoundary</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p37801058115614"><a name="p37801058115614"></a><a name="p37801058115614"></a>设置<a href="LoadData(ISASI).md">Load3D</a>时A1/B1边界值。</p>
</td>
</tr>
<tr id="row3787671224"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1155273074915"><a name="p1155273074915"></a><a name="p1155273074915"></a><a href="SetLoadDataRepeat(ISASI).md">SetLoadDataRepeat</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p532619442559"><a name="p532619442559"></a><a name="p532619442559"></a>用于设置Load3Dv2接口的repeat参数。设置repeat参数后，可以通过调用一次Load3Dv2接口完成多个迭代的数据搬运。</p>
</td>
</tr>
<tr id="row4662691428"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p18552183094913"><a name="p18552183094913"></a><a name="p18552183094913"></a><a href="SetLoadDataPaddingValue(ISASI).md">SetLoadDataPaddingValue</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p53183549554"><a name="p53183549554"></a><a name="p53183549554"></a>设置padValue，用于Load3Dv1/Load3Dv2。</p>
</td>
</tr>
<tr id="row165019119218"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p12552123034913"><a name="p12552123034913"></a><a name="p12552123034913"></a><a href="Fixpipe(ISASI).md">Fixpipe</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p7194151162615"><a name="p7194151162615"></a><a name="p7194151162615"></a>矩阵计算完成后，对结果进行处理，例如对计算结果进行量化操作，并把数据从CO1搬迁到Global Memory中。</p>
</td>
</tr>
<tr id="row188711975503"><td class="cellrowborder" rowspan="5" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.1 "><p id="p13495131445016"><a name="p13495131445016"></a><a name="p13495131445016"></a>同步控制</p>
</td>
<td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.2 "><p id="p14495181465015"><a name="p14495181465015"></a><a name="p14495181465015"></a><a href="SetFlag-WaitFlag(ISASI).md">SetFlag/WaitFlag</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.99%" headers="mcps1.2.4.1.3 "><p id="p158718715503"><a name="p158718715503"></a><a name="p158718715503"></a>同一核内不同流水线之间的同步指令。具有数据依赖的不同流水指令之间需要插此同步。</p>
</td>
</tr>
<tr id="row181116835017"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p16495161412505"><a name="p16495161412505"></a><a name="p16495161412505"></a><a href="PipeBarrier(ISASI).md">PipeBarrier</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p21128145011"><a name="p21128145011"></a><a name="p21128145011"></a>阻塞相同流水，具有数据依赖的相同流水之间需要插此同步。</p>
</td>
</tr>
<tr id="row653814222592"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1289326165918"><a name="p1289326165918"></a><a name="p1289326165918"></a><a href="DataSyncBarrier(ISASI).md">DataSyncBarrier</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p8600163212594"><a name="p8600163212594"></a><a name="p8600163212594"></a>用于阻塞后续的指令执行，直到所有之前的内存访问指令（需要等待的内存位置可通过参数控制）执行结束。</p>
</td>
</tr>
<tr id="row114551284506"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p5495141475017"><a name="p5495141475017"></a><a name="p5495141475017"></a><a href="CrossCoreSetFlag(ISASI).md">CrossCoreSetFlag</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p20555155242412"><a name="p20555155242412"></a><a name="p20555155242412"></a>针对分离模式，AI Core上的Cube核（AIC）与Vector核（AIV）之间的同步设置指令。</p>
</td>
</tr>
<tr id="row1169128105014"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1495141410509"><a name="p1495141410509"></a><a name="p1495141410509"></a><a href="CrossCoreWaitFlag(ISASI).md">CrossCoreWaitFlag</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1453712312015"><a name="p1453712312015"></a><a name="p1453712312015"></a>针对分离模式，AI Core上的Cube核（AIC）与Vector核（AIV）之间的同步等待指令。</p>
</td>
</tr>
<tr id="row329139125011"><td class="cellrowborder" rowspan="2" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.1 "><p id="p164958141507"><a name="p164958141507"></a><a name="p164958141507"></a>缓存处理</p>
</td>
<td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.2 "><p id="p124961914135016"><a name="p124961914135016"></a><a name="p124961914135016"></a><a href="ICachePreLoad(ISASI).md">ICachePreLoad</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.99%" headers="mcps1.2.4.1.3 "><p id="p16327940182218"><a name="p16327940182218"></a><a name="p16327940182218"></a>从指令所在DDR地址预加载指令到ICache中。</p>
</td>
</tr>
<tr id="row19196799505"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p4496131419504"><a name="p4496131419504"></a><a name="p4496131419504"></a><a href="GetICachePreloadStatus(ISASI).md">GetICachePreloadStatus</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p319669165012"><a name="p319669165012"></a><a name="p319669165012"></a>获取ICACHE的PreLoad的状态。</p>
</td>
</tr>
<tr id="row2493696509"><td class="cellrowborder" rowspan="4" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.1 "><p id="p649661485013"><a name="p649661485013"></a><a name="p649661485013"></a>系统变量访问</p>
</td>
<td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.2 "><p id="p1649651485012"><a name="p1649651485012"></a><a name="p1649651485012"></a><a href="GetProgramCounter(ISASI).md">GetProgramCounter</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.99%" headers="mcps1.2.4.1.3 "><p id="p1349315905014"><a name="p1349315905014"></a><a name="p1349315905014"></a>获取程序计数器的指针，程序计数器用于记录当前程序执行的位置。</p>
</td>
</tr>
<tr id="row19631896502"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p74961814175010"><a name="p74961814175010"></a><a name="p74961814175010"></a><a href="GetSubBlockNum(ISASI).md">GetSubBlockNum</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p966612452211"><a name="p966612452211"></a><a name="p966612452211"></a>获取AI Core上Vector核的数量。</p>
</td>
</tr>
<tr id="row278314916501"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p2496314115019"><a name="p2496314115019"></a><a name="p2496314115019"></a><a href="GetSubBlockIdx(ISASI).md">GetSubBlockIdx</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p7322151817114"><a name="p7322151817114"></a><a name="p7322151817114"></a>获取AI Core上Vector核的ID。</p>
</td>
</tr>
<tr id="row591914913509"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p449621416508"><a name="p449621416508"></a><a name="p449621416508"></a><a href="GetSystemCycle(ISASI).md">GetSystemCycle</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p318519518103"><a name="p318519518103"></a><a name="p318519518103"></a>获取当前系统cycle数，若换算成时间需要按照50MHz的频率，时间单位为us，换算公式为：time = (cycle数/50) us 。</p>
</td>
</tr>
<tr id="row20229161014506"><td class="cellrowborder" rowspan="4" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.1 "><p id="p649691414505"><a name="p649691414505"></a><a name="p649691414505"></a>原子操作</p>
</td>
<td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.2 "><p id="p114961614195013"><a name="p114961614195013"></a><a name="p114961614195013"></a><a href="SetAtomicMax(ISASI).md">SetAtomicMax</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.99%" headers="mcps1.2.4.1.3 "><p id="p1555107145615"><a name="p1555107145615"></a><a name="p1555107145615"></a>原子操作函数，设置后续从VECOUT传输到GM的数据是否执行原子比较，将待拷贝的内容和GM已有内容进行比较，将最大值写入GM。</p>
</td>
</tr>
<tr id="row237920103503"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p18496141413508"><a name="p18496141413508"></a><a name="p18496141413508"></a><a href="SetAtomicMin(ISASI).md">SetAtomicMin</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p15379191018502"><a name="p15379191018502"></a><a name="p15379191018502"></a>原子操作函数，设置后续从VECOUT传输到GM的数据是否执行原子比较，将待拷贝的内容和GM已有内容进行比较，将最小值写入GM。</p>
</td>
</tr>
<tr id="row5515201016501"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p4496814165013"><a name="p4496814165013"></a><a name="p4496814165013"></a><a href="SetStoreAtomicConfig(ISASI).md">SetStoreAtomicConfig</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p95151510125017"><a name="p95151510125017"></a><a name="p95151510125017"></a>设置原子操作使能位与原子操作类型。</p>
</td>
</tr>
<tr id="row466110104502"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p949614146509"><a name="p949614146509"></a><a name="p949614146509"></a><a href="GetStoreAtomicConfig(ISASI).md">GetStoreAtomicConfig</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p4576142854016"><a name="p4576142854016"></a><a name="p4576142854016"></a>获取原子操作使能位与原子操作类型的值。</p>
</td>
</tr>
<tr id="row161503191126"><td class="cellrowborder" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.1 "><p id="p1715151915126"><a name="p1715151915126"></a><a name="p1715151915126"></a>调试接口</p>
</td>
<td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.2 "><p id="p9552163084912"><a name="p9552163084912"></a><a name="p9552163084912"></a><a href="CheckLocalMemoryIA(ISASI).md">CheckLocalMemoryIA</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.99%" headers="mcps1.2.4.1.3 "><p id="p5863853185711"><a name="p5863853185711"></a><a name="p5863853185711"></a>监视设定范围内的UB读写行为，如果监视到有设定范围的读写行为则会出现EXCEPTION报错，未监视到设定范围的读写行为则不会报错。</p>
</td>
</tr>
<tr id="row18446920142913"><td class="cellrowborder" rowspan="3" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.1 "><p id="p164464202295"><a name="p164464202295"></a><a name="p164464202295"></a>Cube分组管理</p>
</td>
<td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.2 "><p id="p7499133216292"><a name="p7499133216292"></a><a name="p7499133216292"></a><a href="CubeResGroupHandle.md">CubeResGroupHandle</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.99%" headers="mcps1.2.4.1.3 "><p id="p24466202298"><a name="p24466202298"></a><a name="p24466202298"></a>CubeResGroupHandle用于在分离模式下通过软同步控制AIC和AIV之间进行通讯，实现<span id="zh-cn_topic_0000001588832845_ph168139148536"><a name="zh-cn_topic_0000001588832845_ph168139148536"></a><a name="zh-cn_topic_0000001588832845_ph168139148536"></a>AI Core</span>计算资源分组。</p>
</td>
</tr>
<tr id="row1493333852918"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p8398498294"><a name="p8398498294"></a><a name="p8398498294"></a><a href="GroupBarrier.md">GroupBarrier</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1493314380295"><a name="p1493314380295"></a><a name="p1493314380295"></a>当同一个<a href="CubeResGroupHandle.md">CubeResGroupHandle</a>中的两个AIV任务之间存在依赖关系时，可以使用GroupBarrier控制同步。</p>
</td>
</tr>
<tr id="row1786094112299"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p186215598290"><a name="p186215598290"></a><a name="p186215598290"></a><a href="KfcWorkspace.md">KfcWorkspace</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1776114192019"><a name="p1776114192019"></a><a name="p1776114192019"></a>KfcWorkspace为通信空间描述符，管理不同<a href="CubeResGroupHandle.md">CubeResGroupHandle</a>的消息通信区划分，与CubeResGroupHandle配合使用。KfcWorkspace的构造函数用于创建KfcWorkspace对象。</p>
</td>
</tr>
</tbody>
</table>

## 高阶API<a name="section3317105813235"></a>

**表 14**  数学计算API列表

<a name="table6328746161212"></a>
<table><thead align="left"><tr id="row18328114610121"><th class="cellrowborder" valign="top" width="37.71%" id="mcps1.2.3.1.1"><p id="p173281846121219"><a name="p173281846121219"></a><a name="p173281846121219"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.29%" id="mcps1.2.3.1.2"><p id="p232844620126"><a name="p232844620126"></a><a name="p232844620126"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row7328204651217"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p2032844612121"><a name="p2032844612121"></a><a name="p2032844612121"></a><a href="Acos.md">Acos</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p17328164651218"><a name="p17328164651218"></a><a name="p17328164651218"></a>按元素做反余弦函数计算。</p>
</td>
</tr>
<tr id="row19328124671211"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p17328204661218"><a name="p17328204661218"></a><a name="p17328204661218"></a><a href="Acosh.md">Acosh</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p83298462120"><a name="p83298462120"></a><a name="p83298462120"></a>按元素做双曲反余弦函数计算。</p>
</td>
</tr>
<tr id="row11329946171211"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p63297464128"><a name="p63297464128"></a><a name="p63297464128"></a><a href="Asin.md">Asin</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p20329114641219"><a name="p20329114641219"></a><a name="p20329114641219"></a>按元素做反正弦函数计算。</p>
</td>
</tr>
<tr id="row1632954617129"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p3329546101216"><a name="p3329546101216"></a><a name="p3329546101216"></a><a href="Asinh.md">Asinh</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p932994611216"><a name="p932994611216"></a><a name="p932994611216"></a>按元素做反双曲正弦函数计算。</p>
</td>
</tr>
<tr id="row1329646131216"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p10329184619129"><a name="p10329184619129"></a><a name="p10329184619129"></a><a href="Atan.md">Atan</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p232904617122"><a name="p232904617122"></a><a name="p232904617122"></a>按元素做三角函数反正切运算。</p>
</td>
</tr>
<tr id="row1932944621219"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1032944641214"><a name="p1032944641214"></a><a name="p1032944641214"></a><a href="Atanh.md">Atanh</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1532914614126"><a name="p1532914614126"></a><a name="p1532914614126"></a>按元素做反双曲正切余弦函数计算。</p>
</td>
</tr>
<tr id="row18329194631210"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1032984631216"><a name="p1032984631216"></a><a name="p1032984631216"></a><a href="Axpy-26.md">Axpy</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p8329114612121"><a name="p8329114612121"></a><a name="p8329114612121"></a>源操作数中每个元素与标量求积后和目的操作数中的对应元素相加。</p>
</td>
</tr>
<tr id="row732916466122"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1732974619121"><a name="p1732974619121"></a><a name="p1732974619121"></a><a href="Ceil.md">Ceil</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1432924610129"><a name="p1432924610129"></a><a name="p1432924610129"></a>获取大于或等于x的最小的整数值，即向正无穷取整操作。</p>
</td>
</tr>
<tr id="row232994671216"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p19329124613120"><a name="p19329124613120"></a><a name="p19329124613120"></a><a href="ClampMax.md">ClampMax</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p14329104681213"><a name="p14329104681213"></a><a name="p14329104681213"></a>将srcTensor中大于scalar的数替换为scalar，小于等于scalar的数保持不变，作为dstTensor输出。</p>
</td>
</tr>
<tr id="row53290468122"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p193291646111212"><a name="p193291646111212"></a><a name="p193291646111212"></a><a href="ClampMin.md">ClampMin</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1132984620124"><a name="p1132984620124"></a><a name="p1132984620124"></a>将srcTensor中小于scalar的数替换为scalar，大于等于scalar的数保持不变，作为dstTensor输出。</p>
</td>
</tr>
<tr id="row12329154631216"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1329174651218"><a name="p1329174651218"></a><a name="p1329174651218"></a><a href="Cos.md">Cos</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p13291946101211"><a name="p13291946101211"></a><a name="p13291946101211"></a>按元素做三角函数余弦运算。</p>
</td>
</tr>
<tr id="row1232914611214"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p53307467120"><a name="p53307467120"></a><a name="p53307467120"></a><a href="Cosh.md">Cosh</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p433011461121"><a name="p433011461121"></a><a name="p433011461121"></a>按元素做双曲余弦函数计算。</p>
</td>
</tr>
<tr id="row4330174681219"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p5330194616123"><a name="p5330194616123"></a><a name="p5330194616123"></a><a href="CumSum.md">CumSum</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p133303469127"><a name="p133303469127"></a><a name="p133303469127"></a>对数据按行依次累加或按列依次累加。</p>
</td>
</tr>
<tr id="row193305461120"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1433044616123"><a name="p1433044616123"></a><a name="p1433044616123"></a><a href="Digamma.md">Digamma</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p333019462124"><a name="p333019462124"></a><a name="p333019462124"></a>按元素计算x的gamma函数的对数导数。</p>
</td>
</tr>
<tr id="row16330546131210"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1233020469127"><a name="p1233020469127"></a><a name="p1233020469127"></a><a href="Erf.md">Erf</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1933044631210"><a name="p1933044631210"></a><a name="p1933044631210"></a>按元素做误差函数计算，也称为高斯误差函数。</p>
</td>
</tr>
<tr id="row1033010465120"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p11330194610124"><a name="p11330194610124"></a><a name="p11330194610124"></a><a href="Erfc.md">Erfc</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p13330204601215"><a name="p13330204601215"></a><a name="p13330204601215"></a>返回输入x的互补误差函数结果，积分区间为x到无穷大。</p>
</td>
</tr>
<tr id="row13330144611210"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p43301046151212"><a name="p43301046151212"></a><a name="p43301046151212"></a><a href="Exp-27.md">Exp</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p6330246201220"><a name="p6330246201220"></a><a name="p6330246201220"></a>按元素取自然指数。</p>
</td>
</tr>
<tr id="row16330174651218"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p6330154610126"><a name="p6330154610126"></a><a name="p6330154610126"></a><a href="Floor.md">Floor</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1033024691219"><a name="p1033024691219"></a><a name="p1033024691219"></a>获取小于或等于x的最小的整数值，即向负无穷取整操作。</p>
</td>
</tr>
<tr id="row17640143823614"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p10640163883615"><a name="p10640163883615"></a><a name="p10640163883615"></a><a href="Fmod.md">Fmod</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p136406388366"><a name="p136406388366"></a><a name="p136406388366"></a>按元素计算两个浮点数相除后的余数。</p>
</td>
</tr>
<tr id="row173301946181212"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p333014681217"><a name="p333014681217"></a><a name="p333014681217"></a><a href="Frac.md">Frac</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1033044671220"><a name="p1033044671220"></a><a name="p1033044671220"></a>按元素做取小数计算。</p>
</td>
</tr>
<tr id="row83303464127"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p8330144611210"><a name="p8330144611210"></a><a name="p8330144611210"></a><a href="Lgamma.md">Lgamma</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1133044651217"><a name="p1133044651217"></a><a name="p1133044651217"></a>按元素计算x的gamma函数的绝对值并求自然对数。</p>
</td>
</tr>
<tr id="row033124619126"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p4331114613128"><a name="p4331114613128"></a><a name="p4331114613128"></a><a href="Log.md">Log</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p5331204620122"><a name="p5331204620122"></a><a name="p5331204620122"></a>按元素以e、2、10为底做对数运算。</p>
</td>
</tr>
<tr id="row7331194661216"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p7331114617127"><a name="p7331114617127"></a><a name="p7331114617127"></a><a href="Power.md">Power</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p183314461122"><a name="p183314461122"></a><a name="p183314461122"></a>实现按元素做幂运算功能。</p>
</td>
</tr>
<tr id="row8331144631211"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p20331146121218"><a name="p20331146121218"></a><a name="p20331146121218"></a><a href="Round.md">Round</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p11331946151214"><a name="p11331946151214"></a><a name="p11331946151214"></a>将输入的元素四舍五入到最接近的整数。</p>
</td>
</tr>
<tr id="row0331174621213"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p233194613123"><a name="p233194613123"></a><a name="p233194613123"></a><a href="Sign.md">Sign</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p433110466122"><a name="p433110466122"></a><a name="p433110466122"></a>按元素执行Sign操作，Sign是指返回输入数据的符号。</p>
</td>
</tr>
<tr id="row13310463124"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p103311346121214"><a name="p103311346121214"></a><a name="p103311346121214"></a><a href="Sin.md">Sin</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1133110462120"><a name="p1133110462120"></a><a name="p1133110462120"></a>按元素做正弦函数计算。</p>
</td>
</tr>
<tr id="row10331124610124"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p5331194615122"><a name="p5331194615122"></a><a name="p5331194615122"></a><a href="Sinh.md">Sinh</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p2331154618126"><a name="p2331154618126"></a><a name="p2331154618126"></a>按元素做双曲正弦函数计算。</p>
</td>
</tr>
<tr id="row9331546131219"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p733118468123"><a name="p733118468123"></a><a name="p733118468123"></a><a href="Tan.md">Tan</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p103311246151214"><a name="p103311246151214"></a><a name="p103311246151214"></a>按元素做正切函数计算。</p>
</td>
</tr>
<tr id="row1533124619123"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p833114651211"><a name="p833114651211"></a><a name="p833114651211"></a><a href="Tanh.md">Tanh</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p033111464127"><a name="p033111464127"></a><a name="p033111464127"></a>按元素做逻辑回归Tanh。</p>
</td>
</tr>
<tr id="row7331164681211"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p203311346191210"><a name="p203311346191210"></a><a name="p203311346191210"></a><a href="Trunc.md">Trunc</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p20331144651218"><a name="p20331144651218"></a><a name="p20331144651218"></a>按元素做浮点数截断操作，即向零取整操作。</p>
</td>
</tr>
<tr id="row123329469123"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p103321046181214"><a name="p103321046181214"></a><a name="p103321046181214"></a><a href="Xor.md">Xor</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p533224611126"><a name="p533224611126"></a><a name="p533224611126"></a>按元素执行Xor（异或）运算。</p>
</td>
</tr>
</tbody>
</table>

**表 15**  量化操作API列表

<a name="table11421635141314"></a>
<table><thead align="left"><tr id="row10422735121314"><th class="cellrowborder" valign="top" width="37.71%" id="mcps1.2.3.1.1"><p id="p3422183561312"><a name="p3422183561312"></a><a name="p3422183561312"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.29%" id="mcps1.2.3.1.2"><p id="p542213354138"><a name="p542213354138"></a><a name="p542213354138"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row164273359136"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p742783581318"><a name="p742783581318"></a><a name="p742783581318"></a><a href="AscendAntiQuant.md">AscendAntiQuant</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p5427835121317"><a name="p5427835121317"></a><a name="p5427835121317"></a>按元素做伪量化计算，比如将int8_t数据类型伪量化为half数据类型。</p>
</td>
</tr>
<tr id="row12427435111311"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1642733591318"><a name="p1642733591318"></a><a name="p1642733591318"></a><a href="AscendDequant.md">AscendDequant</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p11427635151312"><a name="p11427635151312"></a><a name="p11427635151312"></a>按元素做反量化计算，比如将int32_t数据类型反量化为half/float等数据类型。</p>
</td>
</tr>
<tr id="row164271635201317"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p2427435201319"><a name="p2427435201319"></a><a name="p2427435201319"></a><a href="AscendQuant.md">AscendQuant</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p342718351139"><a name="p342718351139"></a><a name="p342718351139"></a>按元素做量化计算，比如将half/float数据类型量化为int8_t数据类型。</p>
</td>
</tr>
</tbody>
</table>

**表 16**  归一化操作API列表

<a name="table3781201031415"></a>
<table><thead align="left"><tr id="row12781510111418"><th class="cellrowborder" valign="top" width="37.71%" id="mcps1.2.3.1.1"><p id="p12782101020142"><a name="p12782101020142"></a><a name="p12782101020142"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.29%" id="mcps1.2.3.1.2"><p id="p6782101016142"><a name="p6782101016142"></a><a name="p6782101016142"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row8786410121418"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p4786131012145"><a name="p4786131012145"></a><a name="p4786131012145"></a><a href="BatchNorm.md">BatchNorm</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1778631013141"><a name="p1778631013141"></a><a name="p1778631013141"></a>对于每个batch中的样本，对其输入的每个特征在batch的维度上进行归一化。</p>
</td>
</tr>
<tr id="row87861710151413"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p11786410151410"><a name="p11786410151410"></a><a name="p11786410151410"></a><a href="DeepNorm.md">DeepNorm</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p17861410111419"><a name="p17861410111419"></a><a name="p17861410111419"></a>在深层神经网络训练过程中，可以替代LayerNorm的一种归一化方法。</p>
</td>
</tr>
<tr id="row18506114553319"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p571415485334"><a name="p571415485334"></a><a name="p571415485334"></a><a href="GroupNorm.md">GroupNorm</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p11714174853314"><a name="p11714174853314"></a><a name="p11714174853314"></a>将输入的C维度分为groupNum组，对每一组数据进行标准化。</p>
</td>
</tr>
<tr id="row5786191091412"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p11786131071415"><a name="p11786131071415"></a><a name="p11786131071415"></a><a href="LayerNorm.md">LayerNorm</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p157861910161416"><a name="p157861910161416"></a><a name="p157861910161416"></a>将输入数据收敛到[0, 1]之间，可以规范网络层输入输出数据分布的一种归一化方法。</p>
</td>
</tr>
<tr id="row378616105148"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p47863104149"><a name="p47863104149"></a><a name="p47863104149"></a><a href="LayerNorm.md">LayerNormGrad</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p11786141091411"><a name="p11786141091411"></a><a name="p11786141091411"></a>用于计算LayerNorm的反向传播梯度。</p>
</td>
</tr>
<tr id="row7786111071411"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1978721091412"><a name="p1978721091412"></a><a name="p1978721091412"></a><a href="LayerNormGradBeta.md">LayerNormGradBeta</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p978771018141"><a name="p978771018141"></a><a name="p978771018141"></a>用于获取反向beta/gmma的数值，和LayerNormGrad共同输出pdx, gmma和beta。</p>
</td>
</tr>
<tr id="row1833102211208"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p233172292010"><a name="p233172292010"></a><a name="p233172292010"></a><a href="Normalize.md">Normalize</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p33482202020"><a name="p33482202020"></a><a name="p33482202020"></a><a href="LayerNorm.md">LayerNorm</a>中，已知均值和方差，计算shape为[A，R]的输入数据的标准差的倒数rstd和归一化输出y。</p>
</td>
</tr>
<tr id="row478741061416"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p7787121071417"><a name="p7787121071417"></a><a name="p7787121071417"></a><a href="RmsNorm.md">RmsNorm</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p19787171017144"><a name="p19787171017144"></a><a name="p19787171017144"></a>实现对shape大小为[B，S，H]的输入数据的RmsNorm归一化。</p>
</td>
</tr>
<tr id="row48116253209"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p58111925192011"><a name="p58111925192011"></a><a name="p58111925192011"></a><a href="WelfordUpdate.md">WelfordUpdate</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p48111625102019"><a name="p48111625102019"></a><a name="p48111625102019"></a>实现Welford算法的前处理。</p>
</td>
</tr>
<tr id="row189721927192015"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p29731627182019"><a name="p29731627182019"></a><a name="p29731627182019"></a><a href="WelfordFinalize.md">WelfordFinalize</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p69731627112018"><a name="p69731627112018"></a><a name="p69731627112018"></a>实现Welford算法的后处理。</p>
</td>
</tr>
</tbody>
</table>

**表 17**  激活函数API列表

<a name="table952317081517"></a>
<table><thead align="left"><tr id="row052317019157"><th class="cellrowborder" valign="top" width="37.71%" id="mcps1.2.3.1.1"><p id="p13523605151"><a name="p13523605151"></a><a name="p13523605151"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.29%" id="mcps1.2.3.1.2"><p id="p20523100141510"><a name="p20523100141510"></a><a name="p20523100141510"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1523406156"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p135235061516"><a name="p135235061516"></a><a name="p135235061516"></a><a href="AdjustSoftMaxRes.md">AdjustSoftMaxRes</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1952380121514"><a name="p1952380121514"></a><a name="p1952380121514"></a>用于对SoftMax相关计算结果做后处理，调整SoftMax的计算结果为指定的值。</p>
</td>
</tr>
<tr id="row10523408156"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1452340201510"><a name="p1452340201510"></a><a name="p1452340201510"></a><a href="FasterGelu.md">FasterGelu</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p15232011159"><a name="p15232011159"></a><a name="p15232011159"></a>FastGelu化简版本的一种激活函数。</p>
</td>
</tr>
<tr id="row452380181515"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p752416091513"><a name="p752416091513"></a><a name="p752416091513"></a><a href="FasterGeluV2.md">FasterGeluV2</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p552413017153"><a name="p552413017153"></a><a name="p552413017153"></a><span>实现FastGeluV2</span>版本的一种激活函数。</p>
</td>
</tr>
<tr id="row16524110151518"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1952430161515"><a name="p1952430161515"></a><a name="p1952430161515"></a><a href="GeGLU.md">GeGLU</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p165241407155"><a name="p165241407155"></a><a name="p165241407155"></a>采用GeLU作为激活函数的GLU变体。</p>
</td>
</tr>
<tr id="row65248014151"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p252412091510"><a name="p252412091510"></a><a name="p252412091510"></a><a href="Gelu.md">Gelu</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p175241406158"><a name="p175241406158"></a><a name="p175241406158"></a>GELU是一个重要的激活函数，其灵感来源于relu和dropout，在激活中引入了随机正则的思想。</p>
</td>
</tr>
<tr id="row165241301159"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p155243071519"><a name="p155243071519"></a><a name="p155243071519"></a><a href="LogSoftMax.md">LogSoftMax</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p35248041516"><a name="p35248041516"></a><a name="p35248041516"></a>对输入tensor做LogSoftmax计算。</p>
</td>
</tr>
<tr id="row1352419071515"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1152410051518"><a name="p1152410051518"></a><a name="p1152410051518"></a><a href="ReGlu.md">ReGlu</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p75248013150"><a name="p75248013150"></a><a name="p75248013150"></a>一种GLU变体，使用Relu作为激活函数。</p>
</td>
</tr>
<tr id="row852413013150"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p852415071513"><a name="p852415071513"></a><a name="p852415071513"></a><a href="Sigmoid.md">Sigmoid</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1052419031510"><a name="p1052419031510"></a><a name="p1052419031510"></a>按元素做逻辑回归Sigmoid。</p>
</td>
</tr>
<tr id="row1252416071515"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p25241601158"><a name="p25241601158"></a><a name="p25241601158"></a><a href="Silu.md">Silu</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p852416041517"><a name="p852416041517"></a><a name="p852416041517"></a>按元素做Silu运算。</p>
</td>
</tr>
<tr id="row5524190141518"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p852419019156"><a name="p852419019156"></a><a name="p852419019156"></a><a href="SimpleSoftMax.md">SimpleSoftMax</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p35247012154"><a name="p35247012154"></a><a name="p35247012154"></a>使用计算好的sum和max数据对输入tensor做softmax计算。</p>
</td>
</tr>
<tr id="row1752419071511"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p195249051510"><a name="p195249051510"></a><a name="p195249051510"></a><a href="SoftMax.md">SoftMax</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p16525100121512"><a name="p16525100121512"></a><a name="p16525100121512"></a>对输入tensor按行做Softmax计算。</p>
</td>
</tr>
<tr id="row452512011153"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p205251707159"><a name="p205251707159"></a><a name="p205251707159"></a><a href="SoftmaxFlash.md">SoftmaxFlash</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1152511081511"><a name="p1152511081511"></a><a name="p1152511081511"></a>SoftMax增强版本，除了可以对输入tensor做softmaxflash计算，还可以根据上一次softmax计算的sum和max来更新本次的softmax计算结果。</p>
</td>
</tr>
<tr id="row352510091518"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p65251081518"><a name="p65251081518"></a><a name="p65251081518"></a><a href="SoftmaxFlashV2.md">SoftmaxFlashV2</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p175251107154"><a name="p175251107154"></a><a name="p175251107154"></a>SoftmaxFlash增强版本，对应FlashAttention-2算法。</p>
</td>
</tr>
<tr id="row46249508349"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p10625550163411"><a name="p10625550163411"></a><a name="p10625550163411"></a><a href="SoftmaxFlashV3.md">SoftmaxFlashV3</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p662545013412"><a name="p662545013412"></a><a name="p662545013412"></a>SoftmaxFlash增强版本，对应Softmax PASA算法。</p>
</td>
</tr>
<tr id="row65251015155"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p155254018158"><a name="p155254018158"></a><a name="p155254018158"></a><a href="SoftmaxGrad.md">SoftmaxGrad</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p8525507154"><a name="p8525507154"></a><a name="p8525507154"></a>对输入tensor做grad反向计算的一种方法。</p>
</td>
</tr>
<tr id="row152513016156"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p352512001517"><a name="p352512001517"></a><a name="p352512001517"></a><a href="SoftmaxGradFront.md">SoftmaxGradFront</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p752570131513"><a name="p752570131513"></a><a name="p752570131513"></a>对输入tensor做grad反向计算的一种方法。</p>
</td>
</tr>
<tr id="row135257081512"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p9525200111511"><a name="p9525200111511"></a><a name="p9525200111511"></a><a href="SwiGLU.md">SwiGLU</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1952550131511"><a name="p1952550131511"></a><a name="p1952550131511"></a>采用Swish作为激活函数的GLU变体。</p>
</td>
</tr>
<tr id="row1252515041520"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p752515071514"><a name="p752515071514"></a><a name="p752515071514"></a><a href="Swish.md">Swish</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p95251306151"><a name="p95251306151"></a><a name="p95251306151"></a>神经网络中的Swish激活函数。</p>
</td>
</tr>
</tbody>
</table>

**表 18**  归约操作API列表

<a name="table56871381153"></a>
<table><thead align="left"><tr id="row368753820157"><th class="cellrowborder" valign="top" width="37.71%" id="mcps1.2.3.1.1"><p id="p968711387154"><a name="p968711387154"></a><a name="p968711387154"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.29%" id="mcps1.2.3.1.2"><p id="p1368773841515"><a name="p1368773841515"></a><a name="p1368773841515"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row16251161812813"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1068811382157"><a name="p1068811382157"></a><a name="p1068811382157"></a><a href="Sum.md">Sum</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p13688203811515"><a name="p13688203811515"></a><a name="p13688203811515"></a>获取最后一个维度的元素总和。</p>
</td>
</tr>
<tr id="row106871338111517"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p16871238131513"><a name="p16871238131513"></a><a name="p16871238131513"></a><a href="Mean.md">Mean</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1668715385158"><a name="p1668715385158"></a><a name="p1668715385158"></a>根据最后一轴的方向对各元素求平均值。</p>
</td>
</tr>
<tr id="row186871038171510"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1268793811159"><a name="p1268793811159"></a><a name="p1268793811159"></a><a href="ReduceXorSum.md">ReduceXorSum</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p12688163811519"><a name="p12688163811519"></a><a name="p12688163811519"></a>按照元素执行Xor（按位异或）运算，并将计算结果ReduceSum求和。</p>
</td>
</tr>
<tr id="row15688103871519"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p95115441718"><a name="p95115441718"></a><a name="p95115441718"></a><a href="ReduceSum-35.md">ReduceSum</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p11517441717"><a name="p11517441717"></a><a name="p11517441717"></a>对一个多维向量按照指定的维度进行数据累加。</p>
</td>
</tr>
<tr id="row199410371158"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p209412372517"><a name="p209412372517"></a><a name="p209412372517"></a><a href="ReduceMean.md">ReduceMean</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p09416377512"><a name="p09416377512"></a><a name="p09416377512"></a>对一个多维向量按照指定的维度求平均值。</p>
</td>
</tr>
<tr id="row6979134616516"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p16979246851"><a name="p16979246851"></a><a name="p16979246851"></a><a href="ReduceMax-36.md">ReduceMax</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p197934616510"><a name="p197934616510"></a><a name="p197934616510"></a>对一个多维向量在指定的维度求最大值。</p>
</td>
</tr>
<tr id="row2429249255"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1542914496515"><a name="p1542914496515"></a><a name="p1542914496515"></a><a href="ReduceMin-37.md">ReduceMin</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p164291497518"><a name="p164291497518"></a><a name="p164291497518"></a>对一个多维向量在指定的维度求最小值。</p>
</td>
</tr>
<tr id="row1211653851"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p112113531558"><a name="p112113531558"></a><a name="p112113531558"></a><a href="ReduceAny.md">ReduceAny</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p15211153654"><a name="p15211153654"></a><a name="p15211153654"></a>对一个多维向量在指定的维度求逻辑或。</p>
</td>
</tr>
<tr id="row18411155515512"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p19411195519518"><a name="p19411195519518"></a><a name="p19411195519518"></a><a href="ReduceAll.md">ReduceAll</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p17411195512511"><a name="p17411195512511"></a><a name="p17411195512511"></a>对一个多维向量在指定的维度求逻辑与。</p>
</td>
</tr>
<tr id="row20399155816518"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p4399958855"><a name="p4399958855"></a><a name="p4399958855"></a><a href="ReduceProd.md">ReduceProd</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p183995581754"><a name="p183995581754"></a><a name="p183995581754"></a>对一个多维向量在指定的维度求积。</p>
</td>
</tr>
</tbody>
</table>

**表 19**  排序操作API列表

<a name="table1075717581619"></a>
<table><thead align="left"><tr id="row87570510167"><th class="cellrowborder" valign="top" width="37.71%" id="mcps1.2.3.1.1"><p id="p87573520167"><a name="p87573520167"></a><a name="p87573520167"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.29%" id="mcps1.2.3.1.2"><p id="p1475714531618"><a name="p1475714531618"></a><a name="p1475714531618"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row475745191616"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p875865181616"><a name="p875865181616"></a><a name="p875865181616"></a><a href="TopK.md">TopK</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p97581255164"><a name="p97581255164"></a><a name="p97581255164"></a>获取最后一个维度的前k个最大值或最小值及其对应的索引。</p>
</td>
</tr>
<tr id="row97581516168"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1475812517162"><a name="p1475812517162"></a><a name="p1475812517162"></a><a href="Concat.md">Concat</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p375875131613"><a name="p375875131613"></a><a name="p375875131613"></a>对数据进行预处理，将要排序的源操作数srcLocal一一对应的合入目标数据concatLocal中，数据预处理完后，可以进行Sort。</p>
</td>
</tr>
<tr id="row14758650165"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p47583591616"><a name="p47583591616"></a><a name="p47583591616"></a><a href="Extract.md">Extract</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p87581950161"><a name="p87581950161"></a><a name="p87581950161"></a>处理Sort的结果数据，输出排序后的value和index。</p>
</td>
</tr>
<tr id="row57583513169"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1375813514166"><a name="p1375813514166"></a><a name="p1375813514166"></a><a href="Sort.md">Sort</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p14758554166"><a name="p14758554166"></a><a name="p14758554166"></a>排序函数，按照数值大小进行降序排序。</p>
</td>
</tr>
<tr id="row475815512168"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p17581591616"><a name="p17581591616"></a><a name="p17581591616"></a><a href="MrgSort-38.md">MrgSort</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1375885121617"><a name="p1375885121617"></a><a name="p1375885121617"></a>将已经排好序的最多4条队列，合并排列成1条队列，结果按照score域由大到小排序。</p>
</td>
</tr>
</tbody>
</table>

**表 20**  数据过滤API列表

<a name="table7398513176"></a>
<table><thead align="left"><tr id="row133985161711"><th class="cellrowborder" valign="top" width="37.71%" id="mcps1.2.3.1.1"><p id="p23983191712"><a name="p23983191712"></a><a name="p23983191712"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.29%" id="mcps1.2.3.1.2"><p id="p10398111111716"><a name="p10398111111716"></a><a name="p10398111111716"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1943972012455"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p138751045194610"><a name="p138751045194610"></a><a name="p138751045194610"></a><a href="Select-39.md">Select</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p0875445174619"><a name="p0875445174619"></a><a name="p0875445174619"></a>给定两个源操作数src0和src1，根据maskTensor相应位置的值（非bit位）选取元素，得到目的操作数dst。</p>
</td>
</tr>
<tr id="row193983120178"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p103981111179"><a name="p103981111179"></a><a name="p103981111179"></a><a href="DropOut.md">DropOut</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p20398121101710"><a name="p20398121101710"></a><a name="p20398121101710"></a>提供根据MaskTensor对源操作数进行过滤的功能，得到目的操作数。</p>
</td>
</tr>
</tbody>
</table>

**表 21**  张量变换API列表

<a name="table86595781819"></a>
<table><thead align="left"><tr id="row16660147101819"><th class="cellrowborder" valign="top" width="37.669999999999995%" id="mcps1.2.3.1.1"><p id="p866010718184"><a name="p866010718184"></a><a name="p866010718184"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.33%" id="mcps1.2.3.1.2"><p id="p66603741815"><a name="p66603741815"></a><a name="p66603741815"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row02335914543"><td class="cellrowborder" valign="top" width="37.669999999999995%" headers="mcps1.2.3.1.1 "><p id="p192315965413"><a name="p192315965413"></a><a name="p192315965413"></a><a href="Transpose-40.md">Transpose</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.33%" headers="mcps1.2.3.1.2 "><p id="p1248169212"><a name="p1248169212"></a><a name="p1248169212"></a>对输入数据进行数据排布及Reshape操作。</p>
</td>
</tr>
<tr id="row113798510352"><td class="cellrowborder" valign="top" width="37.669999999999995%" headers="mcps1.2.3.1.1 "><p id="p73791051163519"><a name="p73791051163519"></a><a name="p73791051163519"></a><a href="TransData.md">TransData</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.33%" headers="mcps1.2.3.1.2 "><p id="p1837919517358"><a name="p1837919517358"></a><a name="p1837919517358"></a>将输入数据的排布格式转换为目标排布格式。</p>
</td>
</tr>
<tr id="row12653659194315"><td class="cellrowborder" valign="top" width="37.669999999999995%" headers="mcps1.2.3.1.1 "><p id="p68351016134419"><a name="p68351016134419"></a><a name="p68351016134419"></a><a href="Broadcast.md">Broadcast</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.33%" headers="mcps1.2.3.1.2 "><p id="p28356164441"><a name="p28356164441"></a><a name="p28356164441"></a>将输入按照输出shape进行广播。</p>
</td>
</tr>
<tr id="row061716212442"><td class="cellrowborder" valign="top" width="37.669999999999995%" headers="mcps1.2.3.1.1 "><p id="p118359168448"><a name="p118359168448"></a><a name="p118359168448"></a><a href="Pad.md">Pad</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.33%" headers="mcps1.2.3.1.2 "><p id="p483521619442"><a name="p483521619442"></a><a name="p483521619442"></a>对height * width的二维Tensor在width方向上pad到32B对齐。</p>
</td>
</tr>
<tr id="row1650326144416"><td class="cellrowborder" valign="top" width="37.669999999999995%" headers="mcps1.2.3.1.1 "><p id="p11835161634418"><a name="p11835161634418"></a><a name="p11835161634418"></a><a href="UnPad.md">UnPad</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.33%" headers="mcps1.2.3.1.2 "><p id="p6835161674420"><a name="p6835161674420"></a><a name="p6835161674420"></a>对height * width的二维Tensor在width方向上进行unpad。</p>
</td>
</tr>
<tr id="row1922374193911"><td class="cellrowborder" valign="top" width="37.669999999999995%" headers="mcps1.2.3.1.1 "><p id="p022434193913"><a name="p022434193913"></a><a name="p022434193913"></a><a href="Fill.md">Fill</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.33%" headers="mcps1.2.3.1.2 "><p id="p122418416392"><a name="p122418416392"></a><a name="p122418416392"></a>将Global Memory上的数据初始化为指定值。</p>
</td>
</tr>
</tbody>
</table>

**表 22**  索引计算API列表

<a name="table67319289189"></a>
<table><thead align="left"><tr id="row1873528161818"><th class="cellrowborder" valign="top" width="37.63%" id="mcps1.2.3.1.1"><p id="p473728141810"><a name="p473728141810"></a><a name="p473728141810"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.370000000000005%" id="mcps1.2.3.1.2"><p id="p1973328111812"><a name="p1973328111812"></a><a name="p1973328111812"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row05463289557"><td class="cellrowborder" valign="top" width="37.63%" headers="mcps1.2.3.1.1 "><p id="p1045773617556"><a name="p1045773617556"></a><a name="p1045773617556"></a><a href="Arange.md">Arange</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.370000000000005%" headers="mcps1.2.3.1.2 "><p id="p189228338558"><a name="p189228338558"></a><a name="p189228338558"></a>给定起始值，等差值和长度，返回一个等差数列。</p>
</td>
</tr>
</tbody>
</table>

**表 23**  矩阵计算API列表

<a name="table16634248182011"></a>
<table><thead align="left"><tr id="row763454810205"><th class="cellrowborder" valign="top" width="37.669999999999995%" id="mcps1.2.3.1.1"><p id="p16634164832017"><a name="p16634164832017"></a><a name="p16634164832017"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.33%" id="mcps1.2.3.1.2"><p id="p1363464815203"><a name="p1363464815203"></a><a name="p1363464815203"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row2635748192012"><td class="cellrowborder" valign="top" width="37.669999999999995%" headers="mcps1.2.3.1.1 "><p id="p10635248122013"><a name="p10635248122013"></a><a name="p10635248122013"></a><a href="Matmul-Kernel侧接口.md">Matmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.33%" headers="mcps1.2.3.1.2 "><p id="p1563517485201"><a name="p1563517485201"></a><a name="p1563517485201"></a>Matmul矩阵乘法的运算。</p>
</td>
</tr>
</tbody>
</table>

**表 24**  HCCL通信类API列表

<a name="table483522817566"></a>
<table><thead align="left"><tr id="row1183572813564"><th class="cellrowborder" valign="top" width="37.669999999999995%" id="mcps1.2.3.1.1"><p id="p4835152811563"><a name="p4835152811563"></a><a name="p4835152811563"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.33%" id="mcps1.2.3.1.2"><p id="p283502813566"><a name="p283502813566"></a><a name="p283502813566"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row7534141110415"><td class="cellrowborder" valign="top" width="37.669999999999995%" headers="mcps1.2.3.1.1 "><p id="p13534411643"><a name="p13534411643"></a><a name="p13534411643"></a><a href="HCCL通信类.md">HCCL通信类</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.33%" headers="mcps1.2.3.1.2 "><p id="p553418111947"><a name="p553418111947"></a><a name="p553418111947"></a>在AI Core侧编排集合通信任务。</p>
</td>
</tr>
</tbody>
</table>

**表 25**  卷积计算API列表

<a name="table12502184212139"></a>
<table><thead align="left"><tr id="row1150274261313"><th class="cellrowborder" valign="top" width="37.6%" id="mcps1.2.3.1.1"><p id="p45022423132"><a name="p45022423132"></a><a name="p45022423132"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.4%" id="mcps1.2.3.1.2"><p id="p75021942101318"><a name="p75021942101318"></a><a name="p75021942101318"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row185021242151312"><td class="cellrowborder" valign="top" width="37.6%" headers="mcps1.2.3.1.1 "><p id="p250294215131"><a name="p250294215131"></a><a name="p250294215131"></a><a href="Conv3D.md">Conv3D</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.4%" headers="mcps1.2.3.1.2 "><p id="p1650394291312"><a name="p1650394291312"></a><a name="p1650394291312"></a>3维卷积正向矩阵运算。</p>
</td>
</tr>
<tr id="row1217212361282"><td class="cellrowborder" valign="top" width="37.6%" headers="mcps1.2.3.1.1 "><p id="p5172113652817"><a name="p5172113652817"></a><a name="p5172113652817"></a><a href="Conv3DBackpropInput.md">Conv3DBackpropInput</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.4%" headers="mcps1.2.3.1.2 "><p id="p317293622815"><a name="p317293622815"></a><a name="p317293622815"></a>卷积的反向运算，求解特征矩阵的反向传播误差。</p>
</td>
</tr>
<tr id="row15623133842814"><td class="cellrowborder" valign="top" width="37.6%" headers="mcps1.2.3.1.1 "><p id="p1162303802817"><a name="p1162303802817"></a><a name="p1162303802817"></a><a href="Conv3DBackpropFilter.md">Conv3DBackpropFilter</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.4%" headers="mcps1.2.3.1.2 "><p id="p86233387286"><a name="p86233387286"></a><a name="p86233387286"></a>卷积的反向运算，求解权重的反向传播误差。</p>
</td>
</tr>
</tbody>
</table>

## Utils API<a name="section15221943104512"></a>

**表 26**  C++标准库API列表

<a name="table99801554584"></a>
<table><thead align="left"><tr id="row179811554088"><th class="cellrowborder" valign="top" width="37.71%" id="mcps1.2.3.1.1"><p id="p298155413815"><a name="p298155413815"></a><a name="p298155413815"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.29%" id="mcps1.2.3.1.2"><p id="p129815541982"><a name="p129815541982"></a><a name="p129815541982"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row99811354881"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p29811545811"><a name="p29811545811"></a><a name="p29811545811"></a><a href="max.md">max</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p138701595101"><a name="p138701595101"></a><a name="p138701595101"></a>比较相同数据类型的两个数中的最大值。</p>
</td>
</tr>
<tr id="row99818543818"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p179811546812"><a name="p179811546812"></a><a name="p179811546812"></a><a href="min.md">min</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p986935911109"><a name="p986935911109"></a><a name="p986935911109"></a>比较相同数据类型的两个数中的最小值。</p>
</td>
</tr>
<tr id="row17982175418816"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p298213542810"><a name="p298213542810"></a><a name="p298213542810"></a><a href="integer_sequence.md">integer_sequence</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p8869259101018"><a name="p8869259101018"></a><a name="p8869259101018"></a>用于生成一个整数序列。</p>
</td>
</tr>
<tr id="row139821454384"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1598219543820"><a name="p1598219543820"></a><a name="p1598219543820"></a><a href="tuple.md">tuple</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p88681259101020"><a name="p88681259101020"></a><a name="p88681259101020"></a><span>允许存储多个不同类型元素</span>的容器。</p>
</td>
</tr>
<tr id="row19821854887"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1829135316100"><a name="p1829135316100"></a><a name="p1829135316100"></a><a href="get.md">get</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p42902537103"><a name="p42902537103"></a><a name="p42902537103"></a><span>从</span>tuple<span>容器中提取指定位置的元素</span>。</p>
</td>
</tr>
<tr id="row19821254681"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p9290175311107"><a name="p9290175311107"></a><a name="p9290175311107"></a><a href="make_tuple.md">make_tuple</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p028955381015"><a name="p028955381015"></a><a name="p028955381015"></a><span>用于便捷地创建</span>tuple<span>对象。</span></p>
</td>
</tr>
<tr id="row169828546818"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p6289453151012"><a name="p6289453151012"></a><a name="p6289453151012"></a><a href="is_convertible.md">is_convertible</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1028845351011"><a name="p1028845351011"></a><a name="p1028845351011"></a>在程序编译时<span>判断两个类型之间是否可以进行隐式转换</span>。</p>
</td>
</tr>
<tr id="row498311545815"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p42881453151010"><a name="p42881453151010"></a><a name="p42881453151010"></a><a href="is_base_of.md">is_base_of</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p11287105315108"><a name="p11287105315108"></a><a name="p11287105315108"></a>在程序编译时<span>判断</span><span>一个类型是否为另一个类型的基类</span>。</p>
</td>
</tr>
<tr id="row1598312541817"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p172878539107"><a name="p172878539107"></a><a name="p172878539107"></a><a href="is_same.md">is_same</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1128711539108"><a name="p1128711539108"></a><a name="p1128711539108"></a><span>在程序</span><span>编译时判断两个类型是否完全相同。</span></p>
</td>
</tr>
<tr id="row498311541816"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p4286105318107"><a name="p4286105318107"></a><a name="p4286105318107"></a><a href="enable_if.md">enable_if</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1328665317104"><a name="p1328665317104"></a><a name="p1328665317104"></a><span>在程序编译时根据某个条件启用或禁用特定的函数模板、类模板或模板特化</span>。</p>
</td>
</tr>
<tr id="row49836541682"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p18285105321019"><a name="p18285105321019"></a><a name="p18285105321019"></a><a href="conditional.md">conditional</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1928525319102"><a name="p1928525319102"></a><a name="p1928525319102"></a>在程序编译时根据一个布尔条件从两个类型中选择一个类型。</p>
</td>
</tr>
<tr id="row15509255143618"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p968185918363"><a name="p968185918363"></a><a name="p968185918363"></a><a href="integral_constant.md">integral_constant</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p15098558367"><a name="p15098558367"></a><a name="p15098558367"></a>用于封装一个编译时常量整数值，是标准库中许多类型特性和编译时计算的基础组件。</p>
</td>
</tr>
</tbody>
</table>

**表 27**  平台信息获取API列表

<a name="table32991747162610"></a>
<table><thead align="left"><tr id="row13299184712616"><th class="cellrowborder" valign="top" width="37.6%" id="mcps1.2.3.1.1"><p id="p15299104762613"><a name="p15299104762613"></a><a name="p15299104762613"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.4%" id="mcps1.2.3.1.2"><p id="p123001847142611"><a name="p123001847142611"></a><a name="p123001847142611"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row63002047192611"><td class="cellrowborder" valign="top" width="37.6%" headers="mcps1.2.3.1.1 "><p id="p530034762613"><a name="p530034762613"></a><a name="p530034762613"></a><a href="PlatformAscendC.md">PlatformAscendC</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.4%" headers="mcps1.2.3.1.2 "><p id="p43001947152612"><a name="p43001947152612"></a><a name="p43001947152612"></a>在实现Host侧的Tiling函数时，可能需要获取一些硬件平台的信息，来支撑Tiling的计算，比如获取硬件平台的核数等信息。PlatformAscendC类提供获取这些平台信息的功能。</p>
</td>
</tr>
<tr id="row6300134719269"><td class="cellrowborder" valign="top" width="37.6%" headers="mcps1.2.3.1.1 "><p id="p14300184720265"><a name="p14300184720265"></a><a name="p14300184720265"></a><a href="PlatformAscendCManager.md">PlatformAscendCManager</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.4%" headers="mcps1.2.3.1.2 "><p id="p1330019473269"><a name="p1330019473269"></a><a name="p1330019473269"></a>基于Kernel Launch算子工程，通过基础调用（Kernel Launch）方式调用算子的场景下，可能需要获取硬件平台相关信息，比如获取硬件平台的核数。PlatformAscendCManager类提供获取平台信息的功能。</p>
</td>
</tr>
</tbody>
</table>

**表 28**  ContextBuilder API列表

<a name="table2675125415261"></a>
<table><thead align="left"><tr id="row1967612545262"><th class="cellrowborder" valign="top" width="37.6%" id="mcps1.2.3.1.1"><p id="p367695412260"><a name="p367695412260"></a><a name="p367695412260"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.4%" id="mcps1.2.3.1.2"><p id="p0676145422615"><a name="p0676145422615"></a><a name="p0676145422615"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1867785432614"><td class="cellrowborder" valign="top" width="37.6%" headers="mcps1.2.3.1.1 "><p id="p967795492618"><a name="p967795492618"></a><a name="p967795492618"></a><a href="ContextBuilder.md">ContextBuilder</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.4%" headers="mcps1.2.3.1.2 "><p id="p56781254122619"><a name="p56781254122619"></a><a name="p56781254122619"></a>ContextBuilder类提供一系列的API接口，支持手动构造类来验证Tiling函数以及KernelContext类用于TilingParse函数的验证。</p>
</td>
</tr>
</tbody>
</table>

**表 29**  RTC API列表

<a name="table59039568269"></a>
<table><thead align="left"><tr id="row99046568261"><th class="cellrowborder" valign="top" width="37.419999999999995%" id="mcps1.2.3.1.1"><p id="p1490425632618"><a name="p1490425632618"></a><a name="p1490425632618"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.580000000000005%" id="mcps1.2.3.1.2"><p id="p15904195682611"><a name="p15904195682611"></a><a name="p15904195682611"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1390513568263"><td class="cellrowborder" valign="top" width="37.419999999999995%" headers="mcps1.2.3.1.1 "><p id="p5905156122619"><a name="p5905156122619"></a><a name="p5905156122619"></a><a href="aclrtcCompileProg.md">aclrtcCompileProg</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p149052056172612"><a name="p149052056172612"></a><a name="p149052056172612"></a><span id="ph17905125672617"><a name="ph17905125672617"></a><a name="ph17905125672617"></a>编译接口，编译指定的程序。</span></p>
</td>
</tr>
<tr id="row3905185632610"><td class="cellrowborder" valign="top" width="37.419999999999995%" headers="mcps1.2.3.1.1 "><p id="p99051356182611"><a name="p99051356182611"></a><a name="p99051356182611"></a><a href="aclrtcCreateProg.md">aclrtcCreateProg</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p6905205652615"><a name="p6905205652615"></a><a name="p6905205652615"></a><span id="ph8905456152619"><a name="ph8905456152619"></a><a name="ph8905456152619"></a>通过给定的参数，创建编译程序的实例。</span></p>
</td>
</tr>
<tr id="row15905105615268"><td class="cellrowborder" valign="top" width="37.419999999999995%" headers="mcps1.2.3.1.1 "><p id="p190625642617"><a name="p190625642617"></a><a name="p190625642617"></a><a href="aclrtcDestroyProg.md">aclrtcDestroyProg</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p1490685613261"><a name="p1490685613261"></a><a name="p1490685613261"></a><span id="ph169061356152611"><a name="ph169061356152611"></a><a name="ph169061356152611"></a>销毁编译程序的实例。</span></p>
</td>
</tr>
<tr id="row8906956122613"><td class="cellrowborder" valign="top" width="37.419999999999995%" headers="mcps1.2.3.1.1 "><p id="p139061156142610"><a name="p139061156142610"></a><a name="p139061156142610"></a><a href="aclrtcGetBinData.md">aclrtcGetBinData</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p590617568261"><a name="p590617568261"></a><a name="p590617568261"></a><span id="ph5906145618261"><a name="ph5906145618261"></a><a name="ph5906145618261"></a>获取编译后的二进制数据。</span></p>
</td>
</tr>
<tr id="row790685613266"><td class="cellrowborder" valign="top" width="37.419999999999995%" headers="mcps1.2.3.1.1 "><p id="p18906185642615"><a name="p18906185642615"></a><a name="p18906185642615"></a><a href="aclrtcGetBinDataSize.md">aclrtcGetBinDataSize</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p2906135614269"><a name="p2906135614269"></a><a name="p2906135614269"></a><span id="ph190617563264"><a name="ph190617563264"></a><a name="ph190617563264"></a>获取编译的二进制数据大小。用于在<a href="aclrtcGetBinData.md">aclrtcGetBinData</a>获取二进制数据时分配对应大小的内存空间。</span></p>
</td>
</tr>
<tr id="row29061056112617"><td class="cellrowborder" valign="top" width="37.419999999999995%" headers="mcps1.2.3.1.1 "><p id="p390635610265"><a name="p390635610265"></a><a name="p390635610265"></a><a href="aclrtcGetCompileLogSize.md">aclrtcGetCompileLogSize</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p19906165612617"><a name="p19906165612617"></a><a name="p19906165612617"></a><span id="ph1290635618262"><a name="ph1290635618262"></a><a name="ph1290635618262"></a>获取编译日志的大小。用于在<a href="aclrtcGetCompileLog.md">aclrtcGetCompileLog</a>获取日志内容时分配对应大小的内存空间。</span></p>
</td>
</tr>
<tr id="row9907185610267"><td class="cellrowborder" valign="top" width="37.419999999999995%" headers="mcps1.2.3.1.1 "><p id="p16907105652615"><a name="p16907105652615"></a><a name="p16907105652615"></a><a href="aclrtcGetCompileLog.md">aclrtcGetCompileLog</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p149072056172614"><a name="p149072056172614"></a><a name="p149072056172614"></a><span id="ph19907156142610"><a name="ph19907156142610"></a><a name="ph19907156142610"></a>获取编译日志的内容，以字符串形式保存。</span></p>
</td>
</tr>
</tbody>
</table>

**表 30**  log API列表

<a name="table1514223372716"></a>
<table><thead align="left"><tr id="row3142033112715"><th class="cellrowborder" valign="top" width="37.79%" id="mcps1.2.3.1.1"><p id="p181421633182714"><a name="p181421633182714"></a><a name="p181421633182714"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.21%" id="mcps1.2.3.1.2"><p id="p1114213332712"><a name="p1114213332712"></a><a name="p1114213332712"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row914623311278"><td class="cellrowborder" valign="top" width="37.79%" headers="mcps1.2.3.1.1 "><p id="p014643382720"><a name="p014643382720"></a><a name="p014643382720"></a><a href="ASC_CPU_LOG.md">ASC_CPU_LOG</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.21%" headers="mcps1.2.3.1.2 "><p id="p101461833162715"><a name="p101461833162715"></a><a name="p101461833162715"></a><span id="ph1014643313273"><a name="ph1014643313273"></a><a name="ph1014643313273"></a>提供Host侧打印Log的功能。开发者可以在算子的TilingFunc代码中使用ASC_CPU_LOG_XXX接口来输出相关内容。</span></p>
</td>
</tr>
</tbody>
</table>

## AI CPU API<a name="section06362251213"></a>

**表 31**  AI CPU API列表

<a name="table340354212211"></a>
<table><thead align="left"><tr id="row1440344222113"><th class="cellrowborder" valign="top" width="37.419999999999995%" id="mcps1.2.3.1.1"><p id="p140384218217"><a name="p140384218217"></a><a name="p140384218217"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.580000000000005%" id="mcps1.2.3.1.2"><p id="p1140384218214"><a name="p1140384218214"></a><a name="p1140384218214"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row19403174252116"><td class="cellrowborder" valign="top" width="37.419999999999995%" headers="mcps1.2.3.1.1 "><p id="p2040314420218"><a name="p2040314420218"></a><a name="p2040314420218"></a><a href="printf-82.md">printf</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p340494292115"><a name="p340494292115"></a><a name="p340494292115"></a><span id="ph15404164211219"><a name="ph15404164211219"></a><a name="ph15404164211219"></a>该接口提供AI CPU算子Kernel调试场景下的格式化输出功能，默认将输出内容解析并打印在屏幕上。</span></p>
</td>
</tr>
<tr id="row10404124212210"><td class="cellrowborder" valign="top" width="37.419999999999995%" headers="mcps1.2.3.1.1 "><p id="p129831814226"><a name="p129831814226"></a><a name="p129831814226"></a><a href="assert-83.md">assert</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p1404204252114"><a name="p1404204252114"></a><a name="p1404204252114"></a><span id="ph040464220211"><a name="ph040464220211"></a><a name="ph040464220211"></a>该接口实现AI CPU算子Kernel调试场景下的assert断言功能。</span></p>
</td>
</tr>
</tbody>
</table>

