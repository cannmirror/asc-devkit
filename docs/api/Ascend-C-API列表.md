# Ascend C API列表<a name="ZH-CN_TOPIC_0000001917094304"></a>

Ascend C 提供了一套层次化的 API 体系，涵盖了从底层 C 扩展到高阶 C++ 类库的完整能力。它支持开发者以标准 C/C++ 语法为基础，在 AI Core\(SIMD/SIMT\) 及 AI CPU 等多种编程模型下，灵活实现精细化的内存管理与高效的矢量/矩阵运算。

## API分类总览<a name="section12203103771513"></a>

下表展示了 Ascend C API 的总体分类，帮助开发者根据编程模型和功能需求快速定位所需API。

<a name="table4683185919158"></a>
<table><thead align="left"><tr id="row1314616019164"><th class="cellrowborder" valign="top" width="33.333333333333336%" id="mcps1.1.4.1.1"><p id="p1614615010163"><a name="p1614615010163"></a><a name="p1614615010163"></a>API一级分类</p>
</th>
<th class="cellrowborder" valign="top" width="33.333333333333336%" id="mcps1.1.4.1.2"><p id="p21467015167"><a name="p21467015167"></a><a name="p21467015167"></a>API二级分类</p>
</th>
<th class="cellrowborder" valign="top" width="33.333333333333336%" id="mcps1.1.4.1.3"><p id="p314616041618"><a name="p314616041618"></a><a name="p314616041618"></a>分类说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1214610131613"><td class="cellrowborder" rowspan="3" valign="top" width="33.333333333333336%" headers="mcps1.1.4.1.1 "><p id="p121462071615"><a name="p121462071615"></a><a name="p121462071615"></a><a href="SIMD-API/SIMD-API列表.md">SIMD API</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.333333333333336%" headers="mcps1.1.4.1.2 "><p id="p111469010160"><a name="p111469010160"></a><a name="p111469010160"></a><a href="SIMD-API/基础API/基础API.md">基础API</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.333333333333336%" headers="mcps1.1.4.1.3 "><p id="p1014610015168"><a name="p1014610015168"></a><a name="p1014610015168"></a>实现对硬件能力的抽象，开放芯片的能力，保证完备性和兼容性。标注为ISASI（Instruction Set Architecture Special Interface，硬件体系结构相关的接口）类别的API，不能保证跨硬件版本兼容。</p>
</td>
</tr>
<tr id="row414620011618"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p11464015163"><a name="p11464015163"></a><a name="p11464015163"></a><a href="SIMD-API/C-API.md">C API</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p814610181615"><a name="p814610181615"></a><a name="p814610181615"></a>纯C接口，开放芯片完备编程能力，支持数组分配内存，一般基于指针编程，提供与业界一致的C语言编程体验。</p>
</td>
</tr>
<tr id="row17146160131611"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p1114616012162"><a name="p1114616012162"></a><a name="p1114616012162"></a><a href="SIMD-API/高阶API/高阶API.md">高阶 API</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p11467019166"><a name="p11467019166"></a><a name="p11467019166"></a>实现一些常用的计算算法，用于提高编程开发效率，通常会调用多种基础API实现。高阶API包括数学库、Matmul、Softmax等API。高阶API可以保证兼容性。</p>
</td>
</tr>
<tr id="row15146130141612"><td class="cellrowborder" valign="top" width="33.333333333333336%" headers="mcps1.1.4.1.1 "><p id="p2014613020165"><a name="p2014613020165"></a><a name="p2014613020165"></a><a href="SIMT-API/概述.md">SIMT API</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.333333333333336%" headers="mcps1.1.4.1.2 "><p id="p10146200151616"><a name="p10146200151616"></a><a name="p10146200151616"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="33.333333333333336%" headers="mcps1.1.4.1.3 "><p id="p61468071616"><a name="p61468071616"></a><a name="p61468071616"></a>对标业界，提供单指令多线程API。以单条指令多个线程的形式来实现并行计算。SIMT编程主要用于向量计算，特别适合处理离散访问、复杂控制逻辑等场景。SIMT API支持两种编程模型：SIMT编程、SIMD与SIMT混合编程，具体支持的API请分别参见<a href="SIMT-API/SIMT编程简介/API列表.md">SIMT编程API列表</a>、<a href="SIMT-API/SIMD与SIMT混合编程简介/API列表-148.md">SIMD与SIMT混合编程API列表</a>。</p>
</td>
</tr>
<tr id="row191468061612"><td class="cellrowborder" valign="top" width="33.333333333333336%" headers="mcps1.1.4.1.1 "><p id="p11465091612"><a name="p11465091612"></a><a name="p11465091612"></a><a href="AI-CPU-API/AI-CPU-API列表.md">AI CPU API</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.333333333333336%" headers="mcps1.1.4.1.2 "><p id="p61466011169"><a name="p61466011169"></a><a name="p61466011169"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="33.333333333333336%" headers="mcps1.1.4.1.3 "><p id="p1514617018163"><a name="p1514617018163"></a><a name="p1514617018163"></a>通常作为上述API的补充，主要承担非矩阵类、逻辑比较复杂的分支密集型计算。</p>
</td>
</tr>
<tr id="row7146007164"><td class="cellrowborder" valign="top" width="33.333333333333336%" headers="mcps1.1.4.1.1 "><p id="p1314617019165"><a name="p1314617019165"></a><a name="p1314617019165"></a><a href="Utils-API/Utils-API列表.md">Utils API</a></p>
</td>
<td class="cellrowborder" valign="top" width="33.333333333333336%" headers="mcps1.1.4.1.2 "><p id="p114710121617"><a name="p114710121617"></a><a name="p114710121617"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="33.333333333333336%" headers="mcps1.1.4.1.3 "><p id="p1714716014160"><a name="p1714716014160"></a><a name="p1714716014160"></a>丰富的通用工具类，涵盖标准库（目前仅支持SIMD）、平台信息获取、运行时编译及日志输出等功能，支持开发者高效实现算子开发与性能优化。</p>
</td>
</tr>
</tbody>
</table>

