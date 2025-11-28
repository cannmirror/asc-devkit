# Matmul使用说明<a name="ZH-CN_TOPIC_0000001570911006"></a>

Ascend C提供一组Matmul高阶API，方便用户快速实现Matmul矩阵乘法的运算操作。

Matmul的计算公式为：C = A \* B + Bias，其示意图如下。

-   A、B为源操作数，A为左矩阵，形状为\[M, K\]；B为右矩阵，形状为\[K, N\]。
-   C为目的操作数，存放矩阵乘结果的矩阵，形状为\[M, N\]。
-   Bias为矩阵乘偏置，形状为\[1, N\]。对A\*B结果矩阵的每一行都采用该Bias进行偏置。

**图 1**  Matmul矩阵乘示意图<a name="fig3161943163113"></a>  
![](figures/Matmul矩阵乘示意图.png "Matmul矩阵乘示意图")

>![](public_sys-resources/icon-note.gif) **说明：** 
>下文中提及的M轴方向，即为A矩阵纵向；K轴方向，即为A矩阵横向或B矩阵纵向；N轴方向，即为B矩阵横向；尾轴，即为矩阵最后一个维度。

Kernel侧实现Matmul矩阵乘运算的步骤概括为：

1.  创建Matmul对象。
2.  初始化操作。
3.  设置左矩阵A、右矩阵B、Bias。
4.  完成矩阵乘操作。
5.  结束矩阵乘操作。

使用Matmul API实现矩阵乘运算的具体步骤如下：

1.  创建Matmul对象。

    创建Matmul对象的示例如下：

    -   默认为MIX模式（包含矩阵计算和矢量计算），该场景下，不能定义ASCENDC\_CUBE\_ONLY宏。
    -   纯Cube模式（只有矩阵计算）场景下，需要在代码中定义ASCENDC\_CUBE\_ONLY宏。

    ```
    // 纯cube模式（只有矩阵计算）场景下，需要设置该代码宏，并且必须在#include "lib/matmul_intf.h"之前设置
    // #define ASCENDC_CUBE_ONLY 
    #include "lib/matmul_intf.h"
    
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> aType; 
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> bType; 
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> cType; 
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType; 
    AscendC::Matmul<aType, bType, cType, biasType> mm; 
    ```

    创建对象时需要传入A、B、C、Bias的参数类型信息， 类型信息通过[MatmulType](#table1188045714378)来定义，包括：内存逻辑位置、数据格式、数据类型。

    ```
    template <AscendC::TPosition POSITION, CubeFormat FORMAT, typename TYPE, bool ISTRANS = false, LayoutMode LAYOUT = LayoutMode::NONE, bool IBSHARE = false> struct MatmulType {
        constexpr static AscendC::TPosition pos = POSITION;
        constexpr static CubeFormat format = FORMAT;
        using T = TYPE;
        constexpr static bool isTrans = ISTRANS;
        constexpr static LayoutMode layout = LAYOUT;
        constexpr static bool ibShare = IBSHARE;
        
    };
    ```

    **表 1**  MatmulType参数说明

    <a name="table1188045714378"></a>
    <table><thead align="left"><tr id="row1588055783717"><th class="cellrowborder" valign="top" width="17.09%" id="mcps1.2.3.1.1"><p id="p108895312110"><a name="p108895312110"></a><a name="p108895312110"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="82.91%" id="mcps1.2.3.1.2"><p id="p1588075712372"><a name="p1588075712372"></a><a name="p1588075712372"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row088010575370"><td class="cellrowborder" valign="top" width="17.09%" headers="mcps1.2.3.1.1 "><p id="p18868503127"><a name="p18868503127"></a><a name="p18868503127"></a>POSITION</p>
    </td>
    <td class="cellrowborder" valign="top" width="82.91%" headers="mcps1.2.3.1.2 "><p id="p11880157123717"><a name="p11880157123717"></a><a name="p11880157123717"></a>内存逻辑位置。</p>
    <p id="p688113573378"><a name="p688113573378"></a><a name="p688113573378"></a>针对<span id="ph178811574370"><a name="ph178811574370"></a><a name="ph178811574370"></a><term id="zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312391781_term1253731311225"></a>Ascend 910C</term></span>：</p>
    <a name="ul13881557133718"></a><a name="ul13881557133718"></a><ul id="ul13881557133718"><li>A矩阵可设置为TPosition::GM，TPosition::VECOUT，TPosition::TSCM</li><li>B矩阵可设置为TPosition::GM，TPosition::VECOUT，TPosition::TSCM</li><li>Bias可设置为TPosition::GM，TPosition::VECOUT</li><li>C矩阵可设置为TPosition::GM，TPosition::VECIN, TPosition::CO1</li></ul>
    <p id="p58814571371"><a name="p58814571371"></a><a name="p58814571371"></a>针对<span id="ph7881135712378"><a name="ph7881135712378"></a><a name="ph7881135712378"></a><term id="zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312391781_term11962195213215"></a>Ascend 910B</term></span>：</p>
    <a name="ul0881205793717"></a><a name="ul0881205793717"></a><ul id="ul0881205793717"><li>A矩阵可设置为TPosition::GM，TPosition::VECOUT，TPosition::TSCM</li><li>B矩阵可设置为TPosition::GM，TPosition::VECOUT，TPosition::TSCM</li><li>Bias可设置为TPosition::GM，TPosition::VECOUT</li><li>C矩阵可设置为TPosition::GM，TPosition::VECIN, TPosition::CO1</li></ul>
    <p id="p24261319154111"><a name="p24261319154111"></a><a name="p24261319154111"></a>注意，C矩阵设置为TPosition::CO1时，C矩阵的数据排布格式仅支持CubeFormat::NZ，C矩阵的数据类型仅支持float、int32_t。</p>
    </td>
    </tr>
    <tr id="row1488214577379"><td class="cellrowborder" valign="top" width="17.09%" headers="mcps1.2.3.1.1 "><p id="p15886145020124"><a name="p15886145020124"></a><a name="p15886145020124"></a>FORMAT</p>
    </td>
    <td class="cellrowborder" valign="top" width="82.91%" headers="mcps1.2.3.1.2 "><p id="p99724910252"><a name="p99724910252"></a><a name="p99724910252"></a>数据的物理排布格式。</p>
    <p id="p19882165713712"><a name="p19882165713712"></a><a name="p19882165713712"></a>针对<span id="ph2883057143719"><a name="ph2883057143719"></a><a name="ph2883057143719"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_1"></a>Ascend 910C</term></span>：</p>
    <a name="ul38831357143718"></a><a name="ul38831357143718"></a><ul id="ul38831357143718"><li>A矩阵可设置为CubeFormat::ND，CubeFormat::NZ, CubeFormat::VECTOR</li><li>B矩阵可设置为CubeFormat::ND，CubeFormat::NZ</li><li>Bias可设置为CubeFormat::ND</li><li>C矩阵可设置为CubeFormat::ND，CubeFormat::NZ，CubeFormat::ND_ALIGN</li></ul>
    <p id="p1288365783712"><a name="p1288365783712"></a><a name="p1288365783712"></a>针对<span id="ph19883115713710"><a name="ph19883115713710"></a><a name="ph19883115713710"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_1"><a name="zh-cn_topic_0000001312391781_term11962195213215_1"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_1"></a>Ascend 910B</term></span>：</p>
    <a name="ul3883185773710"></a><a name="ul3883185773710"></a><ul id="ul3883185773710"><li>A矩阵可设置为CubeFormat::ND，CubeFormat::NZ, CubeFormat::VECTOR</li><li>B矩阵可设置为CubeFormat::ND，CubeFormat::NZ</li><li>Bias可设置为CubeFormat::ND</li><li>C矩阵可设置为CubeFormat::ND，CubeFormat::NZ，CubeFormat::ND_ALIGN</li></ul>
    <p id="p671644313140"><a name="p671644313140"></a><a name="p671644313140"></a>关于CubeFormat::NZ格式的A矩阵、B矩阵、C矩阵的对齐约束，请参考<a href="#table98851538118">表3</a>。</p>
    </td>
    </tr>
    <tr id="row16884145713371"><td class="cellrowborder" valign="top" width="17.09%" headers="mcps1.2.3.1.1 "><p id="p128866505123"><a name="p128866505123"></a><a name="p128866505123"></a>TYPE</p>
    </td>
    <td class="cellrowborder" valign="top" width="82.91%" headers="mcps1.2.3.1.2 "><p id="p374213782918"><a name="p374213782918"></a><a name="p374213782918"></a>数据类型。</p>
    <div class="p" id="p88856572370"><a name="p88856572370"></a><a name="p88856572370"></a>针对<span id="ph4885185793710"><a name="ph4885185793710"></a><a name="ph4885185793710"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_2"><a name="zh-cn_topic_0000001312391781_term1253731311225_2"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_2"></a>Ascend 910C</term></span>：<a name="ul1188518573370"></a><a name="ul1188518573370"></a><ul id="ul1188518573370"><li>A矩阵可设置为half、float、bfloat16_t 、int8_t、int4b_t</li><li>B矩阵可设置为half、float、bfloat16_t 、int8_t、int4b_t</li><li>Bias可设置为half、float、int32_t</li><li>C矩阵可设置为half、float、bfloat16_t、int32_t、int8_t</li></ul>
    </div>
    <div class="p" id="p138851257153713"><a name="p138851257153713"></a><a name="p138851257153713"></a>针对<span id="ph7885257133719"><a name="ph7885257133719"></a><a name="ph7885257133719"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_2"><a name="zh-cn_topic_0000001312391781_term11962195213215_2"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_2"></a>Ascend 910B</term></span>：<a name="ul198858571377"></a><a name="ul198858571377"></a><ul id="ul198858571377"><li>A矩阵可设置为half、float、bfloat16_t 、int8_t、int4b_t</li><li>B矩阵可设置为half、float、bfloat16_t 、int8_t、int4b_t</li><li>Bias可设置为half、float、int32_t</li><li>C矩阵可设置为half、float、bfloat16_t、int32_t、int8_t</li></ul>
    </div>
    <a name="ul4885757173710"></a><a name="ul4885757173710"></a>
    <p id="p6886457163710"><a name="p6886457163710"></a><a name="p6886457163710"></a><strong id="b7886175793718"><a name="b7886175793718"></a><a name="b7886175793718"></a>注意：除B矩阵为int8_t数据类型外，A矩阵和B矩阵数据类型需要一致，具体数据类型组合关系请参考</strong><a href="#table1996113269499">表2</a>。A矩阵和B矩阵为int4b_t数据类型时，矩阵内轴的数据个数必须为偶数。例如，A矩阵为int4b_t数据类型且不转置时，<a href="TCubeTiling结构体.md#p11899125875617">singleCoreK</a>必须是偶数。</p>
    </td>
    </tr>
    <tr id="row198861357173716"><td class="cellrowborder" valign="top" width="17.09%" headers="mcps1.2.3.1.1 "><p id="p84551411817"><a name="p84551411817"></a><a name="p84551411817"></a>ISTRANS</p>
    </td>
    <td class="cellrowborder" valign="top" width="82.91%" headers="mcps1.2.3.1.2 "><p id="p4886185713714"><a name="p4886185713714"></a><a name="p4886185713714"></a>是否开启支持矩阵转置的功能。</p>
    <a name="ul1388645718375"></a><a name="ul1388645718375"></a><ul id="ul1388645718375"><li>true：开启支持矩阵转置的功能，运行时可以分别通过<a href="SetTensorA.md">SetTensorA</a>和<a href="SetTensorB.md">SetTensorB</a>中的isTransposeA、isTransposeB参数设置A、B矩阵是否转置。若设置A、B矩阵转置，Matmul会认为A矩阵形状为[K, M]，B矩阵形状为[N, K]。</li><li>false：默认值，不开启支持矩阵转置的功能，通过<a href="SetTensorA.md">SetTensorA</a>和<a href="SetTensorB.md">SetTensorB</a>不能设置A、B矩阵的转置情况。Matmul会认为A矩阵形状为[M, K]，B矩阵形状为[K, N]。</li></ul>
    <p id="p25351314131118"><a name="p25351314131118"></a><a name="p25351314131118"></a>注意，由于<span id="ph173181828090"><a name="ph173181828090"></a><a name="ph173181828090"></a><span id="ph14318162813918"><a name="ph14318162813918"></a><a name="ph14318162813918"></a>L1 Buffer</span></span>上的矩阵数据有分形对齐的约束，A、B矩阵转置和不转置时所需的L1空间可能不相同，在开启支持矩阵转置功能时，必须保证按照<a href="TCubeTiling结构体.md">Matmul Tiling参数</a>申请的L1空间不超过<span id="ph20110352101813"><a name="ph20110352101813"></a><a name="ph20110352101813"></a><span id="ph411015210181"><a name="ph411015210181"></a><a name="ph411015210181"></a>L1 Buffer</span></span>的规格，判断方式为(depthA1*Ceil(baseM/c0Size)*baseK + depthB1*Ceil(baseN/c0Size)*baseK) * db * sizeoof(dtype) &lt; L1Size，db表示L1是否开启double buffer，取值1（不开启double buffer）或2（开启double buffer），其余参数的含义请参考<a href="TCubeTiling结构体.md#table1563162142915">表1</a>。</p>
    </td>
    </tr>
    <tr id="row1488675713371"><td class="cellrowborder" valign="top" width="17.09%" headers="mcps1.2.3.1.1 "><p id="p127582139543"><a name="p127582139543"></a><a name="p127582139543"></a>LAYOUT</p>
    </td>
    <td class="cellrowborder" valign="top" width="82.91%" headers="mcps1.2.3.1.2 "><p id="p14887957103713"><a name="p14887957103713"></a><a name="p14887957103713"></a>表征数据的排布。</p>
    <p id="p1887165763713"><a name="p1887165763713"></a><a name="p1887165763713"></a>NONE：默认值，表示不使用BatchMatmul；其他选项表示使用BatchMatmul。</p>
    <p id="p178871657133710"><a name="p178871657133710"></a><a name="p178871657133710"></a>NORMAL：BMNK的数据排布格式，具体可参考<a href="IterateBatch.md#li536045110115">IterateBatch</a>中对该数据排布的介绍。</p>
    <p id="p1388775703719"><a name="p1388775703719"></a><a name="p1388775703719"></a>BSNGD：原始BSH shape做reshape后的数据排布，具体可参考<a href="IterateBatch.md#li298041002213">IterateBatch</a>中对该数据排布的介绍。</p>
    <p id="p148874573377"><a name="p148874573377"></a><a name="p148874573377"></a>SBNGD：原始SBH shape做reshape后的数据排布，具体可参考<a href="IterateBatch.md#li6785191319227">IterateBatch</a>中对该数据排布的介绍。</p>
    <p id="p888713577371"><a name="p888713577371"></a><a name="p888713577371"></a>BNGS1S2：一般为前两种数据排布进行矩阵乘的输出，S1S2数据连续存放，一个S1S2为一个batch的计算数据，具体可参考<a href="IterateBatch.md#li1922441712222">IterateBatch</a>中对该数据排布的介绍。</p>
    </td>
    </tr>
    <tr id="row88871857133714"><td class="cellrowborder" valign="top" width="17.09%" headers="mcps1.2.3.1.1 "><p id="p1613334125414"><a name="p1613334125414"></a><a name="p1613334125414"></a>IBSHARE</p>
    </td>
    <td class="cellrowborder" valign="top" width="82.91%" headers="mcps1.2.3.1.2 "><p id="p1288710573379"><a name="p1288710573379"></a><a name="p1288710573379"></a>是否使能IBShare（IntraBlock Share）。IBShare的功能是能够复用<span id="ph988725713374"><a name="ph988725713374"></a><a name="ph988725713374"></a><span id="ph488711573374"><a name="ph488711573374"></a><a name="ph488711573374"></a>L1 Buffer</span></span>上相同的A矩阵或B矩阵数据，复用的矩阵必须在<span id="ph6401203815911"><a name="ph6401203815911"></a><a name="ph6401203815911"></a><span id="ph440113388919"><a name="ph440113388919"></a><a name="ph440113388919"></a>L1 Buffer</span></span>上全载。A矩阵和B矩阵仅有一个使能IBShare的场景，与<a href="MatmulConfig.md#table6981133810309">IBShare模板</a>配合使用，具体参数设置详见<a href="MatmulConfig.md#table1761013213153">表2</a>。</p>
    <p id="p58873575373"><a name="p58873575373"></a><a name="p58873575373"></a>注意，A矩阵和B矩阵同时使能IBShare的场景，表示<span id="ph12906134517152"><a name="ph12906134517152"></a><a name="ph12906134517152"></a><span id="ph16906945101512"><a name="ph16906945101512"></a><a name="ph16906945101512"></a>L1 Buffer</span></span>上的A矩阵和B矩阵同时复用，需要满足：</p>
    <a name="ul38871557143711"></a><a name="ul38871557143711"></a><ul id="ul38871557143711"><li>同一算子中其它Matmul对象的A矩阵和B矩阵也必须同时使能IBShare；</li><li><span id="ph68351959122717"><a name="ph68351959122717"></a><a name="ph68351959122717"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_3"><a name="zh-cn_topic_0000001312391781_term11962195213215_3"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_3"></a>Ascend 910B</term></span>，获取矩阵计算结果时，只支持调用<a href="IterateAll.md">IterateAll</a>接口，且只支持输出到GlobalTensor，即计算结果放置于Global Memory的地址。</li><li><span id="ph10607560287"><a name="ph10607560287"></a><a name="ph10607560287"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_3"><a name="zh-cn_topic_0000001312391781_term1253731311225_3"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_3"></a>Ascend 910C</term></span>，获取矩阵计算结果时，只支持调用<a href="IterateAll.md">IterateAll</a>接口，且只支持输出到GlobalTensor，即计算结果放置于Global Memory的地址。</li></ul>
    <p id="p10924173410583"><a name="p10924173410583"></a><a name="p10924173410583"></a><span id="ph10924103465815"><a name="ph10924103465815"></a><a name="ph10924103465815"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_4"><a name="zh-cn_topic_0000001312391781_term1253731311225_4"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_4"></a>Ascend 910C</term></span>支持该参数。</p>
    <p id="p19924634105816"><a name="p19924634105816"></a><a name="p19924634105816"></a><span id="ph19248344581"><a name="ph19248344581"></a><a name="ph19248344581"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_4"><a name="zh-cn_topic_0000001312391781_term11962195213215_4"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_4"></a>Ascend 910B</term></span>支持该参数。</p>
    </td>
    </tr>
    </tbody>
    </table>

    **表 2**  Matmul输入输出数据类型的支持列表

    <a name="table1996113269499"></a>
    <table><thead align="left"><tr id="row14961182654919"><th class="cellrowborder" valign="top" width="17.88%" id="mcps1.2.6.1.1"><p id="p1696192654916"><a name="p1696192654916"></a><a name="p1696192654916"></a>A矩阵</p>
    </th>
    <th class="cellrowborder" valign="top" width="16.37%" id="mcps1.2.6.1.2"><p id="p1796116269498"><a name="p1796116269498"></a><a name="p1796116269498"></a>B矩阵</p>
    </th>
    <th class="cellrowborder" valign="top" width="13.639999999999999%" id="mcps1.2.6.1.3"><p id="p196172610496"><a name="p196172610496"></a><a name="p196172610496"></a>Bias</p>
    </th>
    <th class="cellrowborder" valign="top" width="14.430000000000001%" id="mcps1.2.6.1.4"><p id="p12961122616491"><a name="p12961122616491"></a><a name="p12961122616491"></a>C矩阵</p>
    </th>
    <th class="cellrowborder" valign="top" width="37.68%" id="mcps1.2.6.1.5"><p id="p484471411911"><a name="p484471411911"></a><a name="p484471411911"></a>支持平台</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1196162615492"><td class="cellrowborder" valign="top" width="17.88%" headers="mcps1.2.6.1.1 "><p id="p996152614492"><a name="p996152614492"></a><a name="p996152614492"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="16.37%" headers="mcps1.2.6.1.2 "><p id="p1997111258524"><a name="p1997111258524"></a><a name="p1997111258524"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.639999999999999%" headers="mcps1.2.6.1.3 "><p id="p9961142614915"><a name="p9961142614915"></a><a name="p9961142614915"></a>float/half</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.430000000000001%" headers="mcps1.2.6.1.4 "><p id="p99621226104910"><a name="p99621226104910"></a><a name="p99621226104910"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="37.68%" headers="mcps1.2.6.1.5 "><a name="ul1522112064712"></a><a name="ul1522112064712"></a><ul id="ul1522112064712"><li><span id="ph9394123345614"><a name="ph9394123345614"></a><a name="ph9394123345614"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_5"><a name="zh-cn_topic_0000001312391781_term1253731311225_5"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_5"></a>Ascend 910C</term></span></li><li><span id="ph045515502448"><a name="ph045515502448"></a><a name="ph045515502448"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_5"><a name="zh-cn_topic_0000001312391781_term11962195213215_5"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_5"></a>Ascend 910B</term></span></li></ul>
    </td>
    </tr>
    <tr id="row199621026164912"><td class="cellrowborder" valign="top" width="17.88%" headers="mcps1.2.6.1.1 "><p id="p1296202624918"><a name="p1296202624918"></a><a name="p1296202624918"></a>half</p>
    </td>
    <td class="cellrowborder" valign="top" width="16.37%" headers="mcps1.2.6.1.2 "><p id="p159621926184911"><a name="p159621926184911"></a><a name="p159621926184911"></a>half</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.639999999999999%" headers="mcps1.2.6.1.3 "><p id="p196212613495"><a name="p196212613495"></a><a name="p196212613495"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.430000000000001%" headers="mcps1.2.6.1.4 "><p id="p296222664917"><a name="p296222664917"></a><a name="p296222664917"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="37.68%" headers="mcps1.2.6.1.5 "><a name="ul1427715527471"></a><a name="ul1427715527471"></a><ul id="ul1427715527471"><li><span id="ph18238371566"><a name="ph18238371566"></a><a name="ph18238371566"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_6"><a name="zh-cn_topic_0000001312391781_term1253731311225_6"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_6"></a>Ascend 910C</term></span></li><li><span id="ph8457252154413"><a name="ph8457252154413"></a><a name="ph8457252154413"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_6"><a name="zh-cn_topic_0000001312391781_term11962195213215_6"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_6"></a>Ascend 910B</term></span></li></ul>
    </td>
    </tr>
    <tr id="row244475111124"><td class="cellrowborder" valign="top" width="17.88%" headers="mcps1.2.6.1.1 "><p id="p86982059161214"><a name="p86982059161214"></a><a name="p86982059161214"></a>half</p>
    </td>
    <td class="cellrowborder" valign="top" width="16.37%" headers="mcps1.2.6.1.2 "><p id="p669885914123"><a name="p669885914123"></a><a name="p669885914123"></a>half</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.639999999999999%" headers="mcps1.2.6.1.3 "><p id="p169835911123"><a name="p169835911123"></a><a name="p169835911123"></a>half</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.430000000000001%" headers="mcps1.2.6.1.4 "><p id="p86983595127"><a name="p86983595127"></a><a name="p86983595127"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="37.68%" headers="mcps1.2.6.1.5 "><a name="ul176985599127"></a><a name="ul176985599127"></a><ul id="ul176985599127"><li><span id="ph18730108105716"><a name="ph18730108105716"></a><a name="ph18730108105716"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_7"><a name="zh-cn_topic_0000001312391781_term1253731311225_7"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_7"></a>Ascend 910C</term></span></li><li><span id="ph569845911214"><a name="ph569845911214"></a><a name="ph569845911214"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_7"><a name="zh-cn_topic_0000001312391781_term11962195213215_7"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_7"></a>Ascend 910B</term></span></li></ul>
    </td>
    </tr>
    <tr id="row81081424532"><td class="cellrowborder" valign="top" width="17.88%" headers="mcps1.2.6.1.1 "><p id="p810824219538"><a name="p810824219538"></a><a name="p810824219538"></a>int8_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="16.37%" headers="mcps1.2.6.1.2 "><p id="p6109194213530"><a name="p6109194213530"></a><a name="p6109194213530"></a>int8_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.639999999999999%" headers="mcps1.2.6.1.3 "><p id="p71091442185313"><a name="p71091442185313"></a><a name="p71091442185313"></a>int32_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.430000000000001%" headers="mcps1.2.6.1.4 "><p id="p010934245319"><a name="p010934245319"></a><a name="p010934245319"></a>int32_t/half</p>
    </td>
    <td class="cellrowborder" valign="top" width="37.68%" headers="mcps1.2.6.1.5 "><a name="ul18917166154810"></a><a name="ul18917166154810"></a><ul id="ul18917166154810"><li><span id="ph66293110575"><a name="ph66293110575"></a><a name="ph66293110575"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_8"><a name="zh-cn_topic_0000001312391781_term1253731311225_8"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_8"></a>Ascend 910C</term></span></li><li><span id="ph3487175413443"><a name="ph3487175413443"></a><a name="ph3487175413443"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_8"><a name="zh-cn_topic_0000001312391781_term11962195213215_8"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_8"></a>Ascend 910B</term></span></li></ul>
    </td>
    </tr>
    <tr id="row561656124819"><td class="cellrowborder" valign="top" width="17.88%" headers="mcps1.2.6.1.1 "><p id="p1471141017486"><a name="p1471141017486"></a><a name="p1471141017486"></a>int4b_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="16.37%" headers="mcps1.2.6.1.2 "><p id="p135313377483"><a name="p135313377483"></a><a name="p135313377483"></a>int4b_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.639999999999999%" headers="mcps1.2.6.1.3 "><p id="p12717102489"><a name="p12717102489"></a><a name="p12717102489"></a>int32_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.430000000000001%" headers="mcps1.2.6.1.4 "><p id="p727113223242"><a name="p727113223242"></a><a name="p727113223242"></a>int32_t/half</p>
    </td>
    <td class="cellrowborder" valign="top" width="37.68%" headers="mcps1.2.6.1.5 "><a name="ul7824826184816"></a><a name="ul7824826184816"></a><ul id="ul7824826184816"><li><span id="ph76269145575"><a name="ph76269145575"></a><a name="ph76269145575"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_9"><a name="zh-cn_topic_0000001312391781_term1253731311225_9"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_9"></a>Ascend 910C</term></span></li><li><span id="ph189325714442"><a name="ph189325714442"></a><a name="ph189325714442"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_9"><a name="zh-cn_topic_0000001312391781_term11962195213215_9"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_9"></a>Ascend 910B</term></span></li></ul>
    </td>
    </tr>
    <tr id="row68030432129"><td class="cellrowborder" valign="top" width="17.88%" headers="mcps1.2.6.1.1 "><p id="p10184164871215"><a name="p10184164871215"></a><a name="p10184164871215"></a>bfloat16_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="16.37%" headers="mcps1.2.6.1.2 "><p id="p1018444861219"><a name="p1018444861219"></a><a name="p1018444861219"></a>bfloat16_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.639999999999999%" headers="mcps1.2.6.1.3 "><p id="p10184348131214"><a name="p10184348131214"></a><a name="p10184348131214"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.430000000000001%" headers="mcps1.2.6.1.4 "><p id="p14184194801213"><a name="p14184194801213"></a><a name="p14184194801213"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="37.68%" headers="mcps1.2.6.1.5 "><a name="ul41845482127"></a><a name="ul41845482127"></a><ul id="ul41845482127"><li><span id="ph08421725717"><a name="ph08421725717"></a><a name="ph08421725717"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_10"><a name="zh-cn_topic_0000001312391781_term1253731311225_10"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_10"></a>Ascend 910C</term></span></li><li><span id="ph17184848161211"><a name="ph17184848161211"></a><a name="ph17184848161211"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_10"><a name="zh-cn_topic_0000001312391781_term11962195213215_10"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_10"></a>Ascend 910B</term></span></li></ul>
    </td>
    </tr>
    <tr id="row3804111619153"><td class="cellrowborder" valign="top" width="17.88%" headers="mcps1.2.6.1.1 "><p id="p171231421101515"><a name="p171231421101515"></a><a name="p171231421101515"></a>bfloat16_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="16.37%" headers="mcps1.2.6.1.2 "><p id="p2012311212155"><a name="p2012311212155"></a><a name="p2012311212155"></a>bfloat16_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.639999999999999%" headers="mcps1.2.6.1.3 "><p id="p1812362115151"><a name="p1812362115151"></a><a name="p1812362115151"></a>half</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.430000000000001%" headers="mcps1.2.6.1.4 "><p id="p112322191520"><a name="p112322191520"></a><a name="p112322191520"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="37.68%" headers="mcps1.2.6.1.5 "><a name="ul14123421161512"></a><a name="ul14123421161512"></a><ul id="ul14123421161512"><li><span id="ph14186972017"><a name="ph14186972017"></a><a name="ph14186972017"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_11"><a name="zh-cn_topic_0000001312391781_term1253731311225_11"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_11"></a>Ascend 910C</term></span></li><li><span id="ph1112322121511"><a name="ph1112322121511"></a><a name="ph1112322121511"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_11"><a name="zh-cn_topic_0000001312391781_term11962195213215_11"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_11"></a>Ascend 910B</term></span></li></ul>
    </td>
    </tr>
    <tr id="row13751942806"><td class="cellrowborder" valign="top" width="17.88%" headers="mcps1.2.6.1.1 "><p id="p163762427019"><a name="p163762427019"></a><a name="p163762427019"></a>half</p>
    </td>
    <td class="cellrowborder" valign="top" width="16.37%" headers="mcps1.2.6.1.2 "><p id="p1137634216010"><a name="p1137634216010"></a><a name="p1137634216010"></a>half</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.639999999999999%" headers="mcps1.2.6.1.3 "><p id="p937616423018"><a name="p937616423018"></a><a name="p937616423018"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.430000000000001%" headers="mcps1.2.6.1.4 "><p id="p1337613425012"><a name="p1337613425012"></a><a name="p1337613425012"></a>int8_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="37.68%" headers="mcps1.2.6.1.5 "><a name="ul11144133314486"></a><a name="ul11144133314486"></a><ul id="ul11144133314486"><li><span id="ph110110105014"><a name="ph110110105014"></a><a name="ph110110105014"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_12"><a name="zh-cn_topic_0000001312391781_term1253731311225_12"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_12"></a>Ascend 910C</term></span></li><li><span id="ph1019411144516"><a name="ph1019411144516"></a><a name="ph1019411144516"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_12"><a name="zh-cn_topic_0000001312391781_term11962195213215_12"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_12"></a>Ascend 910B</term></span></li></ul>
    </td>
    </tr>
    <tr id="row46043713615"><td class="cellrowborder" valign="top" width="17.88%" headers="mcps1.2.6.1.1 "><p id="p71282013183619"><a name="p71282013183619"></a><a name="p71282013183619"></a>bfloat16_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="16.37%" headers="mcps1.2.6.1.2 "><p id="p7128913113616"><a name="p7128913113616"></a><a name="p7128913113616"></a>bfloat16_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.639999999999999%" headers="mcps1.2.6.1.3 "><p id="p512841363618"><a name="p512841363618"></a><a name="p512841363618"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.430000000000001%" headers="mcps1.2.6.1.4 "><p id="p212821353618"><a name="p212821353618"></a><a name="p212821353618"></a>int8_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="37.68%" headers="mcps1.2.6.1.5 "><a name="ul1712841314364"></a><a name="ul1712841314364"></a><ul id="ul1712841314364"><li><span id="ph18690181217011"><a name="ph18690181217011"></a><a name="ph18690181217011"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_13"><a name="zh-cn_topic_0000001312391781_term1253731311225_13"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_13"></a>Ascend 910C</term></span></li><li><span id="ph18128101303615"><a name="ph18128101303615"></a><a name="ph18128101303615"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_13"><a name="zh-cn_topic_0000001312391781_term11962195213215_13"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_13"></a>Ascend 910B</term></span></li></ul>
    </td>
    </tr>
    <tr id="row2050214177187"><td class="cellrowborder" valign="top" width="17.88%" headers="mcps1.2.6.1.1 "><p id="p194813307182"><a name="p194813307182"></a><a name="p194813307182"></a>int8_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="16.37%" headers="mcps1.2.6.1.2 "><p id="p1481530191812"><a name="p1481530191812"></a><a name="p1481530191812"></a>int8_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.639999999999999%" headers="mcps1.2.6.1.3 "><p id="p13481730201817"><a name="p13481730201817"></a><a name="p13481730201817"></a>int32_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.430000000000001%" headers="mcps1.2.6.1.4 "><p id="p18481123012185"><a name="p18481123012185"></a><a name="p18481123012185"></a>int8_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="37.68%" headers="mcps1.2.6.1.5 "><a name="ul194812030191811"></a><a name="ul194812030191811"></a><ul id="ul194812030191811"><li><span id="ph119018351105"><a name="ph119018351105"></a><a name="ph119018351105"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_14"><a name="zh-cn_topic_0000001312391781_term1253731311225_14"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_14"></a>Ascend 910C</term></span></li><li><span id="ph18481203081820"><a name="ph18481203081820"></a><a name="ph18481203081820"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_14"><a name="zh-cn_topic_0000001312391781_term11962195213215_14"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_14"></a>Ascend 910B</term></span></li></ul>
    </td>
    </tr>
    <tr id="row10732163714419"><td class="cellrowborder" valign="top" width="17.88%" headers="mcps1.2.6.1.1 "><p id="p121620444411"><a name="p121620444411"></a><a name="p121620444411"></a>half</p>
    </td>
    <td class="cellrowborder" valign="top" width="16.37%" headers="mcps1.2.6.1.2 "><p id="p21694411410"><a name="p21694411410"></a><a name="p21694411410"></a>half</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.639999999999999%" headers="mcps1.2.6.1.3 "><p id="p116174484120"><a name="p116174484120"></a><a name="p116174484120"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.430000000000001%" headers="mcps1.2.6.1.4 "><p id="p51684419413"><a name="p51684419413"></a><a name="p51684419413"></a>half</p>
    </td>
    <td class="cellrowborder" valign="top" width="37.68%" headers="mcps1.2.6.1.5 "><a name="ul71611449414"></a><a name="ul71611449414"></a><ul id="ul71611449414"><li><span id="ph08601838605"><a name="ph08601838605"></a><a name="ph08601838605"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_15"><a name="zh-cn_topic_0000001312391781_term1253731311225_15"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_15"></a>Ascend 910C</term></span></li><li><span id="ph191644411410"><a name="ph191644411410"></a><a name="ph191644411410"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_15"><a name="zh-cn_topic_0000001312391781_term11962195213215_15"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_15"></a>Ascend 910B</term></span></li></ul>
    </td>
    </tr>
    <tr id="row279202617185"><td class="cellrowborder" valign="top" width="17.88%" headers="mcps1.2.6.1.1 "><p id="p279312262184"><a name="p279312262184"></a><a name="p279312262184"></a>half</p>
    </td>
    <td class="cellrowborder" valign="top" width="16.37%" headers="mcps1.2.6.1.2 "><p id="p3793192621810"><a name="p3793192621810"></a><a name="p3793192621810"></a>half</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.639999999999999%" headers="mcps1.2.6.1.3 "><p id="p13793226131815"><a name="p13793226131815"></a><a name="p13793226131815"></a>half</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.430000000000001%" headers="mcps1.2.6.1.4 "><p id="p15793122681812"><a name="p15793122681812"></a><a name="p15793122681812"></a>half</p>
    </td>
    <td class="cellrowborder" valign="top" width="37.68%" headers="mcps1.2.6.1.5 "><a name="ul61575741916"></a><a name="ul61575741916"></a><ul id="ul61575741916"><li><span id="ph184551144704"><a name="ph184551144704"></a><a name="ph184551144704"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_16"><a name="zh-cn_topic_0000001312391781_term1253731311225_16"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_16"></a>Ascend 910C</term></span></li><li><span id="ph1415714731914"><a name="ph1415714731914"></a><a name="ph1415714731914"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_16"><a name="zh-cn_topic_0000001312391781_term11962195213215_16"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_16"></a>Ascend 910B</term></span></li></ul>
    </td>
    </tr>
    <tr id="row842765934119"><td class="cellrowborder" valign="top" width="17.88%" headers="mcps1.2.6.1.1 "><p id="p1767833425"><a name="p1767833425"></a><a name="p1767833425"></a>bfloat16_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="16.37%" headers="mcps1.2.6.1.2 "><p id="p187671338429"><a name="p187671338429"></a><a name="p187671338429"></a>bfloat16_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.639999999999999%" headers="mcps1.2.6.1.3 "><p id="p1776714314422"><a name="p1776714314422"></a><a name="p1776714314422"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.430000000000001%" headers="mcps1.2.6.1.4 "><p id="p1276763164213"><a name="p1276763164213"></a><a name="p1276763164213"></a>bfloat16_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="37.68%" headers="mcps1.2.6.1.5 "><a name="ul147679315421"></a><a name="ul147679315421"></a><ul id="ul147679315421"><li><span id="ph27638471019"><a name="ph27638471019"></a><a name="ph27638471019"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_17"><a name="zh-cn_topic_0000001312391781_term1253731311225_17"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_17"></a>Ascend 910C</term></span></li><li><span id="ph77671833424"><a name="ph77671833424"></a><a name="ph77671833424"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_17"><a name="zh-cn_topic_0000001312391781_term11962195213215_17"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_17"></a>Ascend 910B</term></span></li></ul>
    </td>
    </tr>
    </tbody>
    </table>

2.  初始化操作。

    ```
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling); // 初始化matmul对象，参数含义请参考REGIST_MATMUL_OBJ章节
    ```

3.  设置左矩阵A、右矩阵B、Bias。

    ```
    mm.SetTensorA(gm_a);    // 设置左矩阵A
    mm.SetTensorB(gm_b);    // 设置右矩阵B
    mm.SetBias(gm_bias);    // 设置Bias
    
    ```

4.  完成矩阵乘操作。

    用户可以选择以下三种调用方式之一。

    -   调用[Iterate](Iterate.md#li135771283591)完成单次迭代计算，叠加while循环完成单核全量数据的计算。Iterate方式，可以自行控制迭代次数，完成所需数据量的计算，方式比较灵活。

        ```
        // API接口内部会进行循环结束条件判断处理
        while (mm.Iterate()) {   
            mm.GetTensorC(gm_c); 
        }
        ```

    -   调用[IterateAll](IterateAll.md)完成单核上所有数据的计算。IterateAll方式，无需循环迭代，使用比较简单。

        ```
        mm.IterateAll(gm_c);
        ```

    -   用户申请用于存放矩阵乘结果的逻辑位置CO1内存，调用一次或多次[Iterate](Iterate.md#li4843165185812)完成单次或多次迭代计算，在需要搬出计算结果时，调用[Fixpipe](Fixpipe.md)接口完成CO1上计算结果的搬运，然后释放申请的CO1内存。该方式下，用户可以灵活控制计算和搬运的节奏，根据实际需要，一次计算对应一次结果的搬出，或者将多次计算结果缓存在CO1内存中，再一次性搬出计算结果。

        在此种调用方式下，创建Matmul对象时，必须定义C矩阵的内存逻辑位置为TPosition::CO1、数据排布格式为CubeFormat::NZ、数据类型为float或int32\_t。

        ```
        // 定义C矩阵的类型信息
        typedef AscendC::MatmulType<AscendC::TPosition::CO1, CubeFormat::NZ, float> cType;
        // 创建Matmul对象
        AscendC::Matmul<aType, bType, cType, biasType> mm; 
        
        // 用户提前申请CO1的内存l0cTensor
        TQue<TPosition::CO1, 1> CO1_;
        // 128 * 1024为申请的CO1内存大小
        GetTPipePtr()->InitBuffer(CO1_, 1, 128 * 1024);
        // L0cT为C矩阵的数据类型。
        // A矩阵数据类型是int8_t或int4b_t时，C矩阵的数据类型是int32_t。
        // A矩阵数据类型是half、float或bfloat16_t时，C矩阵的数据类型是float。
        LocalTensor<L0cT> l0cTensor = CO1_.template AllocTensor<L0cT>();
        
        // 将l0cTensor作为入参传入Iterate，矩阵乘结果输出到用户申请的l0cTensor上
        mm.Iterate(false, l0cTensor);
        
        // 调用Fixpipe接口将CO1上的计算结果搬运到GM
        FixpipeParamsV220 params;
        params.nSize = nSize;
        params.mSize = mSize;
        params.srcStride = srcStride;
        params.dstStride = dstStride;
        CO1_.EnQue(l0cTensor);
        CO1_.template DeQue<L0cT>();
        Fixpipe<cType, L0cT, CFG_ROW_MAJOR>(gm[dstOffset], l0cTensor, params);
        
        //释放CO1内存
        CO1_.FreeTensor(l0cTensor);
        ```

5.  结束矩阵乘操作。

    ```
    mm.End();
    ```

**表 3**  CubeFormat::NZ格式的矩阵对齐要求

<a name="table98851538118"></a>
<table><thead align="left"><tr id="row7885231715"><th class="cellrowborder" valign="top" width="33.33333333333333%" id="mcps1.2.4.1.1"><p id="p188858315118"><a name="p188858315118"></a><a name="p188858315118"></a>源/目的操作数</p>
</th>
<th class="cellrowborder" valign="top" width="33.33333333333333%" id="mcps1.2.4.1.2"><p id="p168851531917"><a name="p168851531917"></a><a name="p168851531917"></a>外轴</p>
</th>
<th class="cellrowborder" valign="top" width="33.33333333333333%" id="mcps1.2.4.1.3"><p id="p988573014"><a name="p988573014"></a><a name="p988573014"></a>内轴</p>
</th>
</tr>
</thead>
<tbody><tr id="row3885731510"><td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.1 "><p id="p1888510312118"><a name="p1888510312118"></a><a name="p1888510312118"></a>A矩阵/B矩阵</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.2 "><p id="p788523816"><a name="p788523816"></a><a name="p788523816"></a>16的倍数</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.3 "><p id="p88851135117"><a name="p88851135117"></a><a name="p88851135117"></a>C0_size的倍数</p>
</td>
</tr>
<tr id="row748664714916"><td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.1 "><p id="p848718471396"><a name="p848718471396"></a><a name="p848718471396"></a>C矩阵（使能channel_split功能）</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.2 "><p id="p144876471395"><a name="p144876471395"></a><a name="p144876471395"></a>16的倍数</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.3 "><p id="p1948711471198"><a name="p1948711471198"></a><a name="p1948711471198"></a>C0_size的倍数</p>
</td>
</tr>
<tr id="row6135165918117"><td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.1 "><p id="p15817171514712"><a name="p15817171514712"></a><a name="p15817171514712"></a>C矩阵（不使能channel_split功能）</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.2 "><p id="p51367599110"><a name="p51367599110"></a><a name="p51367599110"></a>16的倍数</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.3 "><p id="p5136165918111"><a name="p5136165918111"></a><a name="p5136165918111"></a>float/int32_t：16的倍数</p>
<p id="p1917716523120"><a name="p1917716523120"></a><a name="p1917716523120"></a>half/bfloat16_t/int8_t：C0_size的倍数</p>
</td>
</tr>
<tr id="row154673191173"><td class="cellrowborder" colspan="3" valign="top" headers="mcps1.2.4.1.1 mcps1.2.4.1.2 mcps1.2.4.1.3 "><p id="p4767112213712"><a name="p4767112213712"></a><a name="p4767112213712"></a>注1：float/int32_t数据类型的C0_size为8，half/bfloat16_t数据类型的C0_size为16，int8_t数据类型的C0_size为32，int4b_t数据类型的C0_size为64。</p>
<p id="p1070874461018"><a name="p1070874461018"></a><a name="p1070874461018"></a>注2：channel_split功能通过<a href="MatmulConfig.md#table1761013213153">MatmulConfig</a>中的isEnableChannelSplit参数配置，具体内容请参考<a href="MatmulConfig.md#table1761013213153">MatmulConfig</a>。</p>
</td>
</tr>
</tbody>
</table>

## 需要包含的头文件<a name="section1682364117469"></a>

```
#include "lib/matmul/matmul_intf.h"
```

## 实现原理<a name="section13229175017585"></a>

以输入矩阵A \(GM, ND, half\)、矩阵B\(GM, ND, half\)，输出矩阵C \(GM, ND, float\)，无Bias场景为例，其中\(GM, ND, half\)表示数据存放在GM上，数据格式为ND，数据类型为half，描述Matmul高阶API典型场景的内部算法框图，如下图所示。

**图 2**  Matmul算法框图<a name="fig072411991916"></a>  
![](figures/Matmul算法框图.png "Matmul算法框图")

计算过程分为如下几步：

1.  数据从GM搬到A1：DataCopy每次从矩阵A，搬出一个stepM\*baseM\*stepKa\*baseK的矩阵块a1，循环多次完成矩阵A的搬运；数据从GM搬到B1：DataCopy每次从矩阵B，搬出一个stepKb\*baseK\*stepN\*baseN的矩阵块b1，循环多次完成矩阵B的搬运；
2.  数据从A1搬到A2：LoadData每次从矩阵块a1，搬出一个baseM \* baseK的矩阵块a0；数据从B1搬到B2，并完成转置：LoadData每次从矩阵块b1，搬出一个baseK \* baseN的矩阵块，并将其转置为baseN \* baseK的矩阵块b0；
3.  矩阵乘：每次完成一个矩阵块a0 \* b0的计算，得到baseM \* baseN的矩阵块co1；
4.  数据从矩阵块co1搬到矩阵块co2： DataCopy每次搬运一块baseM \* baseN的矩阵块co1到singleCoreM \* singleCoreN的矩阵块co2中；
5.  重复2-4步骤，完成矩阵块a1 \* b1的计算；
6.  数据从矩阵块co2搬到矩阵块C：DataCopy每次搬运一块singleCoreM \* singleCoreN的矩阵块co2到矩阵块C中；
7.  重复1-6步骤，完成矩阵A \* B = C的计算。

