# 910B迁移910D不兼容样例
## 概述
针对910D不兼容910B的部分特性，本小节提供了若干样例，用户可以根据样例进行迁移。
本小节样例使用<<<>>>内核调用符来完成算子核函数在NPU侧运行验证的基础流程，给出了对应的端到端实现。
## 支持的AI处理器
- Ascend 910D
## Ascend兼容性样例
<table>
<tr>
<td rowspan="1" align="center">样例名称</td>
<td colspan="2" align="center">描述</td>
</tr>
<tr>
<td rowspan="2" align="center"><a href="./data_copy_l1togm/">L1到GM搬运</a></td>
<td align="center"><a href="./data_copy_l1togm/910B/">910B</a></td>
<td align="center">直接使用DataCopy接口进行搬运</td>
</tr>
<tr>
<td align="center"><a href="./data_copy_l1togm/910D/">910D</a></td>
<td align="center">DataCopy接口不支持L1 Buffer到GM的通路。用户在cube only场景下，可以在GM多分配一个单位矩阵，通过Mmad矩阵乘法计算输出到L0C Buffer，再从L0C Buffer通过Fixpipe搬运到GM。</td>
</tr>
<tr>
<td rowspan="2" align="center"><a href="./inits_const_value/">L0A Buffer/L0B Buffer初始化</a></td>
<td align="center"><a href="./inits_const_value/910B/">910B</a></td>
<td align="center">直接使用基础API InitConstValue对L0A Buffer和L0B Buffer进行初始化</td>
</tr>
<tr>
<td align="center"><a href="./inits_const_value/910D/">910D</a></td>
<td align="center">351x架构版本删除L0A Buffer/L0B Buffer初始化的相关硬件指令。用户可以通过先初始化L1 Buffer，再通过LoadData接口将L1 Buffer上的数据搬运到L0A Buffer/L0B Buffer。</td>
</tr>
<td rowspan="2" align="center"><a href="./matmul_s4/">int4b_t数据类型的矩阵计算</a></td>
<td align="center"><a href="./matmul_s4/910B/">910B</a></td>
<td align="center">直接使用Matmul高阶APi进行矩阵计算</td>
</tr>
<tr>
<td align="center"><a href="./matmul_s4/910D/">910D</a></td>
<td align="center">Cube计算单元删除int4b_t数据类型。用户可以在算子侧通过MIX模式再Vector Core进行int4b_t到int8_t的Cast转换，再通过UB搬运到L1后进行Mmad计算。</td>
</tr>
<td rowspan="2" align="center"><a href="./set_loaddata_boundary/">L1 Buffer带边界值场景</a></td>
<td align="center"><a href="./set_loaddata_boundary/910B/">910B</a></td>
<td align="center">使用SetLoadDataBoundary</td>
</tr>
<tr>
<td align="center"><a href="./set_loaddata_boundary/910D/">910D</a></td>
<td align="center">351x架构硬件删除了L1 Buffer的边界值设定相关寄存器，不再支持SetLoadDataBoundary接口。该接口用于设置Load3D时L1 Buffer的边界值。如果指令在处理源操作数时，源操作数在L1 Buffer上的地址超出设置的边界，则会从L1 Buffer的起始地址开始夺取。设置为0表示无边界，可使用整个L1 Buffer。</td>
</tr>
</table>