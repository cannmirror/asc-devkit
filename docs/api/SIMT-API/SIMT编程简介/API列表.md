# API列表<a name="ZH-CN_TOPIC_0000002541425560"></a>

## 同步与内存栅栏<a name="section1440016363389"></a>

**表 1**  同步接口

<a name="table355621172410"></a>
<table><thead align="left"><tr id="row105561111192410"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p11556191111244"><a name="p11556191111244"></a><a name="p11556191111244"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p1655618115241"><a name="p1655618115241"></a><a name="p1655618115241"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row10556171192410"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12556111118244"><a name="p12556111118244"></a><a name="p12556111118244"></a><a href="../同步与内存栅栏/同步接口/asc_syncthreads.md">asc_syncthreads</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1179785312220"><a name="p1179785312220"></a><a name="p1179785312220"></a>等待当前thread block内所有thread代码都执行到该函数位置。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  内存栅栏接口

<a name="table16916526133816"></a>
<table><thead align="left"><tr id="row1591692616387"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p4916182653815"><a name="p4916182653815"></a><a name="p4916182653815"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p5916162611387"><a name="p5916162611387"></a><a name="p5916162611387"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row391616266389"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p7916112613819"><a name="p7916112613819"></a><a name="p7916112613819"></a><a href="../同步与内存栅栏/内存栅栏接口/asc_threadfence.md">asc_threadfence</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p49163265384"><a name="p49163265384"></a><a name="p49163265384"></a>用于保证不同核对同一份全局、共享内存的访问过程中，写入操作的时序性。</p>
</td>
</tr>
<tr id="row7101146144212"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p20106461425"><a name="p20106461425"></a><a name="p20106461425"></a><a href="../同步与内存栅栏/内存栅栏接口/asc_threadfence_block.md">asc_threadfence_block</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p19101046194216"><a name="p19101046194216"></a><a name="p19101046194216"></a>用于协调同一线程块（Thread Block）内线程之间的内存操作顺序，<span>确保某一线程在调用asc_</span>threadfence_block()<span>之前</span><span>的所有内存读写操作对</span><span>同一线程块内的其他线程</span><span>可见</span>。</p>
</td>
</tr>
</tbody>
</table>

## 原子操作<a name="section632385711384"></a>

**表 3**  原子操作

<a name="table17209165495117"></a>
<table><thead align="left"><tr id="row720915541514"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p16210954205119"><a name="p16210954205119"></a><a name="p16210954205119"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p122101254105114"><a name="p122101254105114"></a><a name="p122101254105114"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row221025405119"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p18210165415117"><a name="p18210165415117"></a><a name="p18210165415117"></a><a href="../原子操作/asc_atomic_add.md">asc_atomic_add</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p14210105415119"><a name="p14210105415119"></a><a name="p14210105415119"></a>对<span id="ph10536132505718"><a name="ph10536132505718"></a><a name="ph10536132505718"></a>Unified Buffer</span>或<span id="ph1753616252577"><a name="ph1753616252577"></a><a name="ph1753616252577"></a>Global Memory</span>上的数据与指定数据执行原子加操作，即将指定数据累加到<span id="ph15143152082811"><a name="ph15143152082811"></a><a name="ph15143152082811"></a>Unified Buffer</span>或<span id="ph214322082812"><a name="ph214322082812"></a><a name="ph214322082812"></a>Global Memory</span>的数据中。</p>
</td>
</tr>
<tr id="row102101054135111"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p10210654105113"><a name="p10210654105113"></a><a name="p10210654105113"></a><a href="../原子操作/asc_atomic_sub.md">asc_atomic_sub</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1021019548518"><a name="p1021019548518"></a><a name="p1021019548518"></a>对<span id="ph5434440195718"><a name="ph5434440195718"></a><a name="ph5434440195718"></a>Unified Buffer</span>或<span id="ph1343414075710"><a name="ph1343414075710"></a><a name="ph1343414075710"></a>Global Memory</span>上的数据与指定数据执行原子减操作，即在<span id="ph93051920194612"><a name="ph93051920194612"></a><a name="ph93051920194612"></a>Unified Buffer</span>或<span id="ph1330532015466"><a name="ph1330532015466"></a><a name="ph1330532015466"></a>Global Memory</span>的数据上减去指定数据。</p>
</td>
</tr>
<tr id="row14210155416511"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12101954125120"><a name="p12101954125120"></a><a name="p12101954125120"></a><a href="../原子操作/asc_atomic_exch.md">asc_atomic_exch</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p921017541514"><a name="p921017541514"></a><a name="p921017541514"></a>对<span id="ph19824155135715"><a name="ph19824155135715"></a><a name="ph19824155135715"></a>Unified Buffer</span>或<span id="ph2824251175717"><a name="ph2824251175717"></a><a name="ph2824251175717"></a>Global Memory</span>地址做原子赋值操作，即将指定数据赋值到<span id="ph38242515577"><a name="ph38242515577"></a><a name="ph38242515577"></a>Unified Buffer</span>或<span id="ph1782418514572"><a name="ph1782418514572"></a><a name="ph1782418514572"></a>Global Memory</span>地址中。</p>
</td>
</tr>
<tr id="row9210125445116"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p16210654145112"><a name="p16210654145112"></a><a name="p16210654145112"></a><a href="../原子操作/asc_atomic_max.md">asc_atomic_max</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p182101754115113"><a name="p182101754115113"></a><a name="p182101754115113"></a>对<span id="ph82259014584"><a name="ph82259014584"></a><a name="ph82259014584"></a>Unified Buffer</span>或<span id="ph722514018587"><a name="ph722514018587"></a><a name="ph722514018587"></a>Global Memory</span>数据做原子求最大值操作，即将<span id="ph72255035818"><a name="ph72255035818"></a><a name="ph72255035818"></a>Unified Buffer</span>或<span id="ph3225180185819"><a name="ph3225180185819"></a><a name="ph3225180185819"></a>Global Memory</span>的数据与指定数据中的最大值赋值到<span id="ph16444335125614"><a name="ph16444335125614"></a><a name="ph16444335125614"></a>Unified Buffer</span>或<span id="ph2444133513562"><a name="ph2444133513562"></a><a name="ph2444133513562"></a>Global Memory</span>地址中。</p>
</td>
</tr>
<tr id="row5210165425115"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p321035415115"><a name="p321035415115"></a><a name="p321035415115"></a><a href="../原子操作/asc_atomic_min.md">asc_atomic_min</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p16210155445111"><a name="p16210155445111"></a><a name="p16210155445111"></a>对<span id="ph236151245817"><a name="ph236151245817"></a><a name="ph236151245817"></a>Unified Buffer</span>或<span id="ph17361012165811"><a name="ph17361012165811"></a><a name="ph17361012165811"></a>Global Memory</span>数据做原子求最小值操作，即将<span id="ph103671216583"><a name="ph103671216583"></a><a name="ph103671216583"></a>Unified Buffer</span>或<span id="ph43616124587"><a name="ph43616124587"></a><a name="ph43616124587"></a>Global Memory</span>的数据与指定数据中的最小值赋值到<span id="ph1336191245816"><a name="ph1336191245816"></a><a name="ph1336191245816"></a>Unified Buffer</span>或<span id="ph1236121265812"><a name="ph1236121265812"></a><a name="ph1236121265812"></a>Global Memory</span>地址中。</p>
</td>
</tr>
<tr id="row162101254175110"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p421011547518"><a name="p421011547518"></a><a name="p421011547518"></a><a href="../原子操作/asc_atomic_inc.md">asc_atomic_inc</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p0210115417511"><a name="p0210115417511"></a><a name="p0210115417511"></a>对<span id="ph4991141914580"><a name="ph4991141914580"></a><a name="ph4991141914580"></a>Unified Buffer</span>或<span id="ph18991719175819"><a name="ph18991719175819"></a><a name="ph18991719175819"></a>Global Memory</span>上address的数值进行原子加1操作，如果address上的数值大于等于指定数值val，则对address赋值为0，否则将address上数值加1。</p>
</td>
</tr>
<tr id="row19210145435112"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p2210115455112"><a name="p2210115455112"></a><a name="p2210115455112"></a><a href="../原子操作/asc_atomic_dec.md">asc_atomic_dec</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1021025445117"><a name="p1021025445117"></a><a name="p1021025445117"></a>对<span id="ph1959983155818"><a name="ph1959983155818"></a><a name="ph1959983155818"></a>Unified Buffer</span>或<span id="ph05991831195817"><a name="ph05991831195817"></a><a name="ph05991831195817"></a>Global Memory</span>上address的数值进行原子减1操作，如果address上的数值等于0或大于指定数值val，则对address赋值为val，否则将address上数值减1。</p>
</td>
</tr>
<tr id="row13198193011527"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p48379327525"><a name="p48379327525"></a><a name="p48379327525"></a><a href="../原子操作/asc_atomic_cas.md">asc_atomic_cas</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p101941330125214"><a name="p101941330125214"></a><a name="p101941330125214"></a>对<span id="ph8959114015812"><a name="ph8959114015812"></a><a name="ph8959114015812"></a>Unified Buffer</span>或<span id="ph0959144013587"><a name="ph0959144013587"></a><a name="ph0959144013587"></a>Global Memory</span>上address的数值进行原子比较赋值操作，如果address上的数值等于指定数值compare，则对address赋值为指定数值val，否则address的数值不变。</p>
</td>
</tr>
<tr id="row1819812308522"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p18837203213524"><a name="p18837203213524"></a><a name="p18837203213524"></a><a href="../原子操作/asc_atomic_and.md">asc_atomic_and</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p919417304529"><a name="p919417304529"></a><a name="p919417304529"></a>对<span id="ph18404486584"><a name="ph18404486584"></a><a name="ph18404486584"></a>Unified Buffer</span>或<span id="ph138406481583"><a name="ph138406481583"></a><a name="ph138406481583"></a>Global Memory</span>上address的数值与指定数值val进行原子与（&）操作，即将address数值与（&）val的结果赋值到<span id="ph7927849219"><a name="ph7927849219"></a><a name="ph7927849219"></a>Unified Buffer</span>或<span id="ph4927742218"><a name="ph4927742218"></a><a name="ph4927742218"></a>Global Memory</span>上。</p>
</td>
</tr>
<tr id="row11982030105210"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1837163211522"><a name="p1837163211522"></a><a name="p1837163211522"></a><a href="../原子操作/asc_atomic_or.md">asc_atomic_or</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p15194130125218"><a name="p15194130125218"></a><a name="p15194130125218"></a>对<span id="ph652716542215"><a name="ph652716542215"></a><a name="ph652716542215"></a>Unified Buffer</span>或<span id="ph1652785412220"><a name="ph1652785412220"></a><a name="ph1652785412220"></a>Global Memory</span>上address的数值与指定数值val进行原子或（|）操作，即将address数值或（|）val的结果赋值到<span id="ph1333582155916"><a name="ph1333582155916"></a><a name="ph1333582155916"></a>Unified Buffer</span>或<span id="ph13335112125912"><a name="ph13335112125912"></a><a name="ph13335112125912"></a>Global Memory</span>上。</p>
</td>
</tr>
<tr id="row819816309523"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p108358323529"><a name="p108358323529"></a><a name="p108358323529"></a><a href="../原子操作/asc_atomic_xor.md">asc_atomic_xor</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p8194203065216"><a name="p8194203065216"></a><a name="p8194203065216"></a>对<span id="ph12461111115918"><a name="ph12461111115918"></a><a name="ph12461111115918"></a>Unified Buffer</span>或<span id="ph19461181110597"><a name="ph19461181110597"></a><a name="ph19461181110597"></a>Global Memory</span>上address的数值与指定数值val进行原子异或（^）操作，即将address数值异或（^）val的结果赋值到<span id="ph746121145910"><a name="ph746121145910"></a><a name="ph746121145910"></a>Unified Buffer</span>或<span id="ph1946111113591"><a name="ph1946111113591"></a><a name="ph1946111113591"></a>Global Memory</span>上。</p>
</td>
</tr>
</tbody>
</table>

## Warp函数<a name="section625115172398"></a>

**表 4**  Warp Vote类函数

<a name="table13746514532"></a>
<table><thead align="left"><tr id="row53744575316"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p133744575314"><a name="p133744575314"></a><a name="p133744575314"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p1137419545315"><a name="p1137419545315"></a><a name="p1137419545315"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row63742535318"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p137411525310"><a name="p137411525310"></a><a name="p137411525310"></a><a href="../Warp函数/Warp-Vote类函数/asc_all.md">asc_all</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p118785325243"><a name="p118785325243"></a><a name="p118785325243"></a>判断是否所有活跃线程的输入均不为0。</p>
</td>
</tr>
<tr id="row1374145125316"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p3374165145317"><a name="p3374165145317"></a><a name="p3374165145317"></a><a href="../Warp函数/Warp-Vote类函数/asc_any.md">asc_any</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p361320505593"><a name="p361320505593"></a><a name="p361320505593"></a>判断是否有活跃线程的输入不为0。</p>
</td>
</tr>
<tr id="row937495135320"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1237412595310"><a name="p1237412595310"></a><a name="p1237412595310"></a><a href="../Warp函数/Warp-Vote类函数/asc_ballot.md">asc_ballot</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p178511254143017"><a name="p178511254143017"></a><a name="p178511254143017"></a>判断Warp内每个活跃线程的输入是否不为0。</p>
</td>
</tr>
<tr id="row937475185312"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p18374152538"><a name="p18374152538"></a><a name="p18374152538"></a><a href="../Warp函数/Warp-Vote类函数/asc_activemask.md">asc_activemask</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p13965195011409"><a name="p13965195011409"></a><a name="p13965195011409"></a>查看Warp内所有线程是否为活跃状态。</p>
</td>
</tr>
</tbody>
</table>

**表 5**  Warp Shfl类函数

<a name="table1890835202518"></a>
<table><thead align="left"><tr id="row99082515250"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p13908105172513"><a name="p13908105172513"></a><a name="p13908105172513"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p5908195102516"><a name="p5908195102516"></a><a name="p5908195102516"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row189081057250"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p17908105102519"><a name="p17908105102519"></a><a name="p17908105102519"></a><a href="../Warp函数/Warp-Shfl类函数/asc_shfl.md">asc_shfl</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p890810552515"><a name="p890810552515"></a><a name="p890810552515"></a>获取Warp内指定线程srcLane输入的用于交换的var值。</p>
</td>
</tr>
<tr id="row1990810562514"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1890811542513"><a name="p1890811542513"></a><a name="p1890811542513"></a><a href="../Warp函数/Warp-Shfl类函数/asc_shfl_up.md">asc_shfl_up</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1290815592512"><a name="p1290815592512"></a><a name="p1290815592512"></a>获取Warp内当前线程向前偏移delta（当前线程LaneId-delta）的线程输入的用于交换的var值。</p>
</td>
</tr>
<tr id="row1790825172519"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p990812518259"><a name="p990812518259"></a><a name="p990812518259"></a><a href="../Warp函数/Warp-Shfl类函数/asc_shfl_down.md">asc_shfl_down</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p159086518255"><a name="p159086518255"></a><a name="p159086518255"></a>获取Warp内当前线程向后偏移delta（当前线程LaneId+delta）的线程输入的用于交换的var值。</p>
</td>
</tr>
<tr id="row1190814518252"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p69086519254"><a name="p69086519254"></a><a name="p69086519254"></a><a href="../Warp函数/Warp-Shfl类函数/asc_shfl_xor.md">asc_shfl_xor</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p209084592519"><a name="p209084592519"></a><a name="p209084592519"></a>获取Warp内当前线程LaneId与输入laneMask做异或操作（LaneId^laneMask）得到的dstLaneId对应线程输入的用于交换的var值。</p>
</td>
</tr>
</tbody>
</table>

**表 6**  Warp Reduce类函数

<a name="table1458810589259"></a>
<table><thead align="left"><tr id="row14588185802515"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p8588125814256"><a name="p8588125814256"></a><a name="p8588125814256"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p158835872514"><a name="p158835872514"></a><a name="p158835872514"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row95881458192510"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p11588165811258"><a name="p11588165811258"></a><a name="p11588165811258"></a><a href="../Warp函数/Warp-Reduce类函数/asc_reduce_add.md">asc_reduce_add</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p105881058172520"><a name="p105881058172520"></a><a name="p105881058172520"></a>对Warp内所有活跃线程输入的val求和。</p>
</td>
</tr>
<tr id="row145884588251"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p17588185820251"><a name="p17588185820251"></a><a name="p17588185820251"></a><a href="../Warp函数/Warp-Reduce类函数/asc_reduce_max.md">asc_reduce_max</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p858945852519"><a name="p858945852519"></a><a name="p858945852519"></a>对Warp内所有活跃线程输入的val求最大值。</p>
</td>
</tr>
<tr id="row1658945816254"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p9589658192520"><a name="p9589658192520"></a><a name="p9589658192520"></a><a href="../Warp函数/Warp-Reduce类函数/asc_reduce_min.md">asc_reduce_min</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p258955842519"><a name="p258955842519"></a><a name="p258955842519"></a>对Warp内所有活跃线程输入val求最小值。</p>
</td>
</tr>
</tbody>
</table>

## 数学函数<a name="section39223495399"></a>

**表 7**  half类型算术函数

<a name="table1890694612146"></a>
<table><thead align="left"><tr id="row79062468142"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p3906124671416"><a name="p3906124671416"></a><a name="p3906124671416"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p89060468140"><a name="p89060468140"></a><a name="p89060468140"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row105902581299"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1454617720"><a name="p1454617720"></a><a name="p1454617720"></a><a href="../数学函数/half类型/half类型算术函数/__habs.md">__habs</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p134549171823"><a name="p134549171823"></a><a name="p134549171823"></a>获取输入数据的绝对值。</p>
</td>
</tr>
<tr id="row189275813914"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p99017231749"><a name="p99017231749"></a><a name="p99017231749"></a><a href="../数学函数/half类型/half类型算术函数/__hfma.md">__hfma</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p390152312411"><a name="p390152312411"></a><a name="p390152312411"></a>对输入数据x、y、z，计算x与y相乘加上z的结果。</p>
</td>
</tr>
<tr id="row9906946171412"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1287949171917"><a name="p1287949171917"></a><a name="p1287949171917"></a><a href="../数学函数/half类型/half类型算术函数/__hadd.md">__hadd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p228604981916"><a name="p228604981916"></a><a name="p228604981916"></a>计算两个half类型数据的相加结果，并遵循CAST_RINT模式舍入。</p>
</td>
</tr>
<tr id="row1490674612146"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p7286124918195"><a name="p7286124918195"></a><a name="p7286124918195"></a><a href="../数学函数/half类型/half类型算术函数/__hsub.md">__hsub</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p19284114920196"><a name="p19284114920196"></a><a name="p19284114920196"></a>计算两个half类型数据的相减结果，并遵循CAST_RINT模式舍入。</p>
</td>
</tr>
<tr id="row1390694618149"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p129069461149"><a name="p129069461149"></a><a name="p129069461149"></a><a href="../数学函数/half类型/half类型算术函数/__hmul.md">__hmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p0906134641415"><a name="p0906134641415"></a><a name="p0906134641415"></a>计算两个half类型数据的相乘结果，并遵循CAST_RINT模式舍入。</p>
</td>
</tr>
<tr id="row109061346141420"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p179061846141412"><a name="p179061846141412"></a><a name="p179061846141412"></a><a href="../数学函数/half类型/half类型算术函数/__hdiv.md">__hdiv</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p14906146191418"><a name="p14906146191418"></a><a name="p14906146191418"></a>计算两个half类型数据的相除结果，并遵循CAST_RINT模式舍入。</p>
</td>
</tr>
<tr id="row109061546141412"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p090604671412"><a name="p090604671412"></a><a name="p090604671412"></a><a href="../数学函数/half类型/half类型算术函数/__hneg.md">__hneg</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1190619461145"><a name="p1190619461145"></a><a name="p1190619461145"></a>获取输入half类型数据的负值。</p>
</td>
</tr>
<tr id="row139061246151415"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p4906124611413"><a name="p4906124611413"></a><a name="p4906124611413"></a><a href="../数学函数/half类型/half类型算术函数/__hfma_relu.md">__hfma_relu</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1890619465142"><a name="p1890619465142"></a><a name="p1890619465142"></a>对输入half类型数据x、y、z，计算x与y相乘加上z的结果，并遵循CAST_RINT模式舍入。负数结果置为0。</p>
</td>
</tr>
</tbody>
</table>

**表 8**  half类型比较函数

<a name="table1675172621519"></a>
<table><thead align="left"><tr id="row3751261155"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p2075172616155"><a name="p2075172616155"></a><a name="p2075172616155"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p27510260152"><a name="p27510260152"></a><a name="p27510260152"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row19607411201010"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p81110351658"><a name="p81110351658"></a><a name="p81110351658"></a><a href="../数学函数/half类型/half类型比较函数/__hmax.md">__hmax</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p16734242758"><a name="p16734242758"></a><a name="p16734242758"></a>获取两个输入数据中的最大值。</p>
</td>
</tr>
<tr id="row1725841115105"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1658211381353"><a name="p1658211381353"></a><a name="p1658211381353"></a><a href="../数学函数/half类型/half类型比较函数/__hmin.md">__hmin</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p7582438957"><a name="p7582438957"></a><a name="p7582438957"></a>获取两个输入数据中的最小值。</p>
</td>
</tr>
<tr id="row1975122620151"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p874412316911"><a name="p874412316911"></a><a name="p874412316911"></a><a href="../数学函数/half类型/half类型比较函数/__hisnan.md">__hisnan</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p15744113114918"><a name="p15744113114918"></a><a name="p15744113114918"></a>判断浮点数是否为nan。</p>
</td>
</tr>
<tr id="row15751426151512"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p199724331919"><a name="p199724331919"></a><a name="p199724331919"></a><a href="../数学函数/half类型/half类型比较函数/__hisinf.md">__hisinf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p19728332911"><a name="p19728332911"></a><a name="p19728332911"></a>判断浮点数是否为无穷。</p>
</td>
</tr>
<tr id="row0752265159"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p58191040233"><a name="p58191040233"></a><a name="p58191040233"></a><a href="../数学函数/half类型/half类型比较函数/__heq.md">__heq</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p311601811135"><a name="p311601811135"></a><a name="p311601811135"></a>比较两个half类型数据是否相等，相等时返回true。</p>
</td>
</tr>
<tr id="row1751426151519"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p781916401530"><a name="p781916401530"></a><a name="p781916401530"></a><a href="../数学函数/half类型/half类型比较函数/__hne.md">__hne</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p161162183135"><a name="p161162183135"></a><a name="p161162183135"></a>比较两个half类型数据是否不相等，不相等时返回true。</p>
</td>
</tr>
<tr id="row1675626191520"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1881915401934"><a name="p1881915401934"></a><a name="p1881915401934"></a><a href="../数学函数/half类型/half类型比较函数/__hle.md">__hle</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p511601817136"><a name="p511601817136"></a><a name="p511601817136"></a>比较两个half类型数据，仅当第一个数小于或等于第二个数时返回true。</p>
</td>
</tr>
<tr id="row10755262153"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1819340234"><a name="p1819340234"></a><a name="p1819340234"></a><a href="../数学函数/half类型/half类型比较函数/__hge.md">__hge</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p9116161814136"><a name="p9116161814136"></a><a name="p9116161814136"></a>比较两个half类型数据，仅当第一个数大于或等于第二个数时返回true。</p>
</td>
</tr>
<tr id="row8752026121512"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p98197401835"><a name="p98197401835"></a><a name="p98197401835"></a><a href="../数学函数/half类型/half类型比较函数/__hlt.md">__hlt</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p13116718111320"><a name="p13116718111320"></a><a name="p13116718111320"></a>比较两个half类型数据，仅当第一个数小于第二个数时返回true。</p>
</td>
</tr>
<tr id="row175726181512"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p158196409317"><a name="p158196409317"></a><a name="p158196409317"></a><a href="../数学函数/half类型/half类型比较函数/__hgt.md">__hgt</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1011616182135"><a name="p1011616182135"></a><a name="p1011616182135"></a>比较两个half类型数据，仅当第一个数大于第二个数时返回true。</p>
</td>
</tr>
<tr id="row97692613155"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1881912401335"><a name="p1881912401335"></a><a name="p1881912401335"></a><a href="../数学函数/half类型/half类型比较函数/__hequ.md">__hequ</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p191162182139"><a name="p191162182139"></a><a name="p191162182139"></a>比较两个half类型数据是否相等，相等时返回true。若任一输入为nan，返回true。</p>
</td>
</tr>
<tr id="row18761926171512"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1481913401331"><a name="p1481913401331"></a><a name="p1481913401331"></a><a href="../数学函数/half类型/half类型比较函数/__hneu.md">__hneu</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p17116161819138"><a name="p17116161819138"></a><a name="p17116161819138"></a>比较两个half类型数据是否不相等，不相等时返回true。若任一输入为nan，返回true。</p>
</td>
</tr>
<tr id="row1676226161516"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p158191340935"><a name="p158191340935"></a><a name="p158191340935"></a><a href="../数学函数/half类型/half类型比较函数/__hleu.md">__hleu</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1611615183137"><a name="p1611615183137"></a><a name="p1611615183137"></a>比较两个half类型数据，当第一个数小于或等于第二个数时返回true。若任一输入为nan，返回true。</p>
</td>
</tr>
<tr id="row187662651511"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12819134011312"><a name="p12819134011312"></a><a name="p12819134011312"></a><a href="../数学函数/half类型/half类型比较函数/__hgeu.md">__hgeu</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p61167184135"><a name="p61167184135"></a><a name="p61167184135"></a>比较两个half类型数据，当第一个数大于或等于第二个数时返回true。若任一输入为nan，返回true。</p>
</td>
</tr>
<tr id="row0766266150"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1981994015313"><a name="p1981994015313"></a><a name="p1981994015313"></a><a href="../数学函数/half类型/half类型比较函数/__hltu.md">__hltu</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p11116418191315"><a name="p11116418191315"></a><a name="p11116418191315"></a>比较两个half类型数据，当第一个数小于第二个数时返回true。若任一输入为nan，返回true。</p>
</td>
</tr>
<tr id="row5302970212"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p208199403319"><a name="p208199403319"></a><a name="p208199403319"></a><a href="../数学函数/half类型/half类型比较函数/__hgtu.md">__hgtu</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p5116218131312"><a name="p5116218131312"></a><a name="p5116218131312"></a>比较两个half类型数据，当第一个数大于第二个数时返回true。若任一输入为nan，返回true。</p>
</td>
</tr>
<tr id="row9762932172110"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p137621532162119"><a name="p137621532162119"></a><a name="p137621532162119"></a><a href="../数学函数/half类型/half类型比较函数/__hmax_nan.md">__hmax_nan</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p5762193222114"><a name="p5762193222114"></a><a name="p5762193222114"></a>获取两个输入数据中的最大值。任一输入为nan时返回nan。</p>
</td>
</tr>
<tr id="row181872338217"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p11187193314218"><a name="p11187193314218"></a><a name="p11187193314218"></a><a href="../数学函数/half类型/half类型比较函数/__hmin_nan.md">__hmin_nan</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1218713313211"><a name="p1218713313211"></a><a name="p1218713313211"></a>获取两个输入数据中的最小值。任一输入为nan时返回nan。</p>
</td>
</tr>
</tbody>
</table>

**表 9**  half类型数学库函数

<a name="table274310062011"></a>
<table><thead align="left"><tr id="row17743905203"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p14743170172018"><a name="p14743170172018"></a><a name="p14743170172018"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p5743180192016"><a name="p5743180192016"></a><a name="p5743180192016"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row10743604201"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1094025518518"><a name="p1094025518518"></a><a name="p1094025518518"></a><a href="../数学函数/half类型/half类型数学库函数/htanh.md">htanh</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p994116551751"><a name="p994116551751"></a><a name="p994116551751"></a>获取输入数据的三角函数双曲正切值。</p>
</td>
</tr>
<tr id="row157441604201"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p719612281674"><a name="p719612281674"></a><a name="p719612281674"></a><a href="../数学函数/half类型/half类型数学库函数/hexp.md">hexp</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p719652819716"><a name="p719652819716"></a><a name="p719652819716"></a>指定输入x，获取e的x次方。</p>
</td>
</tr>
<tr id="row9744809200"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1324524814720"><a name="p1324524814720"></a><a name="p1324524814720"></a><a href="../数学函数/half类型/half类型数学库函数/hexp2.md">hexp2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p192458482718"><a name="p192458482718"></a><a name="p192458482718"></a>指定输入x，获取2的x次方。</p>
</td>
</tr>
<tr id="row1074430132012"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p125110782314"><a name="p125110782314"></a><a name="p125110782314"></a><a href="../数学函数/half类型/half类型数学库函数/hexp10.md">hexp10</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p2050311271224"><a name="p2050311271224"></a><a name="p2050311271224"></a>指定输入x，获取10的x次方。</p>
</td>
</tr>
<tr id="row87449017208"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p205031127102215"><a name="p205031127102215"></a><a name="p205031127102215"></a><a href="../数学函数/half类型/half类型数学库函数/hlog.md">hlog</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p950212742212"><a name="p950212742212"></a><a name="p950212742212"></a>获取以e为底，输入数据的对数。</p>
</td>
</tr>
<tr id="row3744150142012"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p105021627142216"><a name="p105021627142216"></a><a name="p105021627142216"></a><a href="../数学函数/half类型/half类型数学库函数/hlog2.md">hlog2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p16501152710228"><a name="p16501152710228"></a><a name="p16501152710228"></a>获取以2为底，输入数据的对数。</p>
</td>
</tr>
<tr id="row474415052018"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p4849183302213"><a name="p4849183302213"></a><a name="p4849183302213"></a><a href="../数学函数/half类型/half类型数学库函数/hlog10.md">hlog10</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p18848103320225"><a name="p18848103320225"></a><a name="p18848103320225"></a>获取以10为底，输入数据的对数。</p>
</td>
</tr>
<tr id="row11744160152011"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p179215397228"><a name="p179215397228"></a><a name="p179215397228"></a><a href="../数学函数/half类型/half类型数学库函数/hcos.md">hcos</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p3792133982217"><a name="p3792133982217"></a><a name="p3792133982217"></a>获取输入数据的三角函数余弦值。</p>
</td>
</tr>
<tr id="row147442018209"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p20237325144615"><a name="p20237325144615"></a><a name="p20237325144615"></a><a href="../数学函数/half类型/half类型数学库函数/hsin.md">hsin</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1523711254465"><a name="p1523711254465"></a><a name="p1523711254465"></a>获取输入数据的三角函数正弦值。</p>
</td>
</tr>
<tr id="row1174440172011"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1813116501295"><a name="p1813116501295"></a><a name="p1813116501295"></a><a href="../数学函数/half类型/half类型数学库函数/hsqrt.md">hsqrt</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1913110508910"><a name="p1913110508910"></a><a name="p1913110508910"></a>获取输入数据x的平方根。</p>
</td>
</tr>
<tr id="row15744604201"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1548917177488"><a name="p1548917177488"></a><a name="p1548917177488"></a><a href="../数学函数/half类型/half类型数学库函数/hrsqrt.md">hrsqrt</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1578384488"><a name="p1578384488"></a><a name="p1578384488"></a>获取输入数据x的平方根的倒数。</p>
</td>
</tr>
<tr id="row187449019207"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p657728154811"><a name="p657728154811"></a><a name="p657728154811"></a><a href="../数学函数/half类型/half类型数学库函数/hrcp.md">hrcp</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p205771488484"><a name="p205771488484"></a><a name="p205771488484"></a>获取输入数据x的倒数。</p>
</td>
</tr>
<tr id="row15704818145017"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p5291690717"><a name="p5291690717"></a><a name="p5291690717"></a><a href="../数学函数/half类型/half类型数学库函数/hrint.md">hrint</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p182918911719"><a name="p182918911719"></a><a name="p182918911719"></a>获取与输入数据最接近的整数，若存在两个同样接近的整数，则获取其中的偶数。</p>
</td>
</tr>
<tr id="row6302164565119"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p17477934135219"><a name="p17477934135219"></a><a name="p17477934135219"></a><a href="../数学函数/half类型/half类型数学库函数/hfloor.md">hfloor</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p194761346522"><a name="p194761346522"></a><a name="p194761346522"></a>获取小于或等于输入数据的最大整数值。</p>
</td>
</tr>
<tr id="row1269954513513"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p2476123410527"><a name="p2476123410527"></a><a name="p2476123410527"></a><a href="../数学函数/half类型/half类型数学库函数/hceil.md">hceil</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p0476173419522"><a name="p0476173419522"></a><a name="p0476173419522"></a>获取大于或等于输入数据的最小整数值。</p>
</td>
</tr>
<tr id="row17849103735218"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p19850537135214"><a name="p19850537135214"></a><a name="p19850537135214"></a><a href="../数学函数/half类型/half类型数学库函数/htrunc.md">htrunc</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p5850137135214"><a name="p5850137135214"></a><a name="p5850137135214"></a>获取对输入数据的浮点数截断后的整数。</p>
</td>
</tr>
</tbody>
</table>

**表 10**  half类型精度转换函数

<a name="table1927111059"></a>
<table><thead align="left"><tr id="row1324119516"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p421111452"><a name="p421111452"></a><a name="p421111452"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p19210119518"><a name="p19210119518"></a><a name="p19210119518"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1213111558"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p7545194413475"><a name="p7545194413475"></a><a name="p7545194413475"></a><a href="../数学函数/half类型/half类型精度转换函数/__float2half.md">__float2half</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1980032615245"><a name="p1980032615245"></a><a name="p1980032615245"></a>获取输入遵循CAST_RINT模式转换成的半精度浮点数。</p>
</td>
</tr>
<tr id="row8211118510"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p10545844174711"><a name="p10545844174711"></a><a name="p10545844174711"></a><a href="../数学函数/half类型/half类型精度转换函数/__float2half_rn.md">__float2half_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p180014266243"><a name="p180014266243"></a><a name="p180014266243"></a>获取输入遵循CAST_RINT模式转换成的半精度浮点数。</p>
</td>
</tr>
<tr id="row16231116512"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p15327122083313"><a name="p15327122083313"></a><a name="p15327122083313"></a><a href="../数学函数/half类型/half类型精度转换函数/__float2half_rn_sat.md">__float2half_rn_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p180052672410"><a name="p180052672410"></a><a name="p180052672410"></a>饱和模式下获取输入遵循CAST_RINT模式转换成的半精度浮点数。</p>
</td>
</tr>
<tr id="row1513415114105"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1759210138116"><a name="p1759210138116"></a><a name="p1759210138116"></a><a href="../数学函数/half类型/half类型精度转换函数/__float22half2_rn_sat.md">__float22half2_rn_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1880312692414"><a name="p1880312692414"></a><a name="p1880312692414"></a>饱和模式下获取输入的两个分量遵循<span id="text16803172616246"><a name="text16803172616246"></a><a name="text16803172616246"></a>CAST_RINT</span>模式转换成的half2类型数据。</p>
</td>
</tr>
<tr id="row1721411850"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p17545154414473"><a name="p17545154414473"></a><a name="p17545154414473"></a><a href="../数学函数/half类型/half类型精度转换函数/__float2half_rz.md">__float2half_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p58001326192411"><a name="p58001326192411"></a><a name="p58001326192411"></a>获取输入遵循CAST_TRUNC模式转换成的半精度浮点数。</p>
</td>
</tr>
<tr id="row42511957"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p969141265211"><a name="p969141265211"></a><a name="p969141265211"></a><a href="../数学函数/half类型/half类型精度转换函数/__float2half_rz_sat.md">__float2half_rz_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p3800102620248"><a name="p3800102620248"></a><a name="p3800102620248"></a>饱和模式下获取输入遵循CAST_TRUNC模式转换成的半精度浮点数。</p>
</td>
</tr>
<tr id="row41081825141015"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p34841401311"><a name="p34841401311"></a><a name="p34841401311"></a><a href="../数学函数/half类型/half类型精度转换函数/__float22half2_rz.md">__float22half2_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p20804152611241"><a name="p20804152611241"></a><a name="p20804152611241"></a>获取输入的两个分量遵循CAST_TRUNC模式转换成的half2类型数据。</p>
</td>
</tr>
<tr id="row15558141414117"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p060214471734"><a name="p060214471734"></a><a name="p060214471734"></a><a href="../数学函数/half类型/half类型精度转换函数/__float22half2_rz_sat.md">__float22half2_rz_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p7804326152414"><a name="p7804326152414"></a><a name="p7804326152414"></a>饱和模式下获取输入的两个分量遵循CAST_TRUNC模式转换成的half2类型数据。</p>
</td>
</tr>
<tr id="row2216116518"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p15546164413471"><a name="p15546164413471"></a><a name="p15546164413471"></a><a href="../数学函数/half类型/half类型精度转换函数/__float2half_rd.md">__float2half_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p6800192611245"><a name="p6800192611245"></a><a name="p6800192611245"></a>获取输入遵循CAST_FLOOR模式转换成的半精度浮点数。</p>
</td>
</tr>
<tr id="row19271118511"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p142252015185219"><a name="p142252015185219"></a><a name="p142252015185219"></a><a href="../数学函数/half类型/half类型精度转换函数/__float2half_rd_sat.md">__float2half_rd_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p178000268248"><a name="p178000268248"></a><a name="p178000268248"></a>饱和模式下获取输入遵循CAST_FLOOR模式转换成的半精度浮点数。</p>
</td>
</tr>
<tr id="row6975194961010"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p21209511319"><a name="p21209511319"></a><a name="p21209511319"></a><a href="../数学函数/half类型/half类型精度转换函数/__float22half2_rd.md">__float22half2_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p48041526182419"><a name="p48041526182419"></a><a name="p48041526182419"></a>获取输入的两个分量遵循CAST_FLOOR模式转换成的half2类型数据。</p>
</td>
</tr>
<tr id="row959494681117"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1558823061114"><a name="p1558823061114"></a><a name="p1558823061114"></a><a href="../数学函数/half类型/half类型精度转换函数/__float22half2_rd_sat.md">__float22half2_rd_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p13804126102410"><a name="p13804126102410"></a><a name="p13804126102410"></a>饱和模式下获取输入的两个分量遵循CAST_FLOOR模式转换成的half2类型数据。</p>
</td>
</tr>
<tr id="row626111954"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p5546124415471"><a name="p5546124415471"></a><a name="p5546124415471"></a><a href="../数学函数/half类型/half类型精度转换函数/__float2half_ru.md">__float2half_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1480022682415"><a name="p1480022682415"></a><a name="p1480022682415"></a>获取输入遵循CAST_CEIL模式转换成的半精度浮点数。</p>
</td>
</tr>
<tr id="row12212113520"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p125581818525"><a name="p125581818525"></a><a name="p125581818525"></a><a href="../数学函数/half类型/half类型精度转换函数/__float2half_ru_sat.md">__float2half_ru_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p15800926112412"><a name="p15800926112412"></a><a name="p15800926112412"></a>饱和模式下获取输入遵循CAST_CEIL模式转换成的半精度浮点数。</p>
</td>
</tr>
<tr id="row2066215601110"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p11128123411113"><a name="p11128123411113"></a><a name="p11128123411113"></a><a href="../数学函数/half类型/half类型精度转换函数/__float22half2_ru.md">__float22half2_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p880411260246"><a name="p880411260246"></a><a name="p880411260246"></a>获取输入的两个分量遵循CAST_CEIL模式转换成的half2类型数据。</p>
</td>
</tr>
<tr id="row14910115610116"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p73650391114"><a name="p73650391114"></a><a name="p73650391114"></a><a href="../数学函数/half类型/half类型精度转换函数/__float22half2_ru_sat.md">__float22half2_ru_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1380472618243"><a name="p1380472618243"></a><a name="p1380472618243"></a>饱和模式下获取输入的两个分量遵循CAST_CEIL模式转换成的half2类型数据。</p>
</td>
</tr>
<tr id="row1021611853"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p174237032416"><a name="p174237032416"></a><a name="p174237032416"></a><a href="../数学函数/half类型/half类型精度转换函数/__float2half_rna.md">__float2half_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p12796175754013"><a name="p12796175754013"></a><a name="p12796175754013"></a>获取输入遵循CAST_ROUND模式转换成的半精度浮点数。</p>
</td>
</tr>
<tr id="row1429111852"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p3265721195210"><a name="p3265721195210"></a><a name="p3265721195210"></a><a href="../数学函数/half类型/half类型精度转换函数/__float2half_rna_sat.md">__float2half_rna_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p10800326132415"><a name="p10800326132415"></a><a name="p10800326132415"></a>饱和模式下获取输入遵循CAST_ROUND模式转换成的半精度浮点数。</p>
</td>
</tr>
<tr id="row42241398125"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p163871649835"><a name="p163871649835"></a><a name="p163871649835"></a><a href="../数学函数/half类型/half类型精度转换函数/__float22half2_rna.md">__float22half2_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p5804626142418"><a name="p5804626142418"></a><a name="p5804626142418"></a>获取输入的两个分量遵循CAST_ROUND模式转换成的half2类型数据。</p>
</td>
</tr>
<tr id="row44156393129"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p11222344171019"><a name="p11222344171019"></a><a name="p11222344171019"></a><a href="../数学函数/half类型/half类型精度转换函数/__float22half2_rna_sat.md">__float22half2_rna_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p13804326152415"><a name="p13804326152415"></a><a name="p13804326152415"></a>饱和模式下获取输入的两个分量遵循CAST_ROUND模式转换成的half2类型数据。</p>
</td>
</tr>
<tr id="row2219115515"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p2025143152418"><a name="p2025143152418"></a><a name="p2025143152418"></a><a href="../数学函数/half类型/half类型精度转换函数/__float2half_ro.md">__float2half_ro</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p62816133411"><a name="p62816133411"></a><a name="p62816133411"></a>获取输入遵循CAST_ODD模式转换成的半精度浮点数。</p>
</td>
</tr>
<tr id="row1023116512"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p148356589558"><a name="p148356589558"></a><a name="p148356589558"></a><a href="../数学函数/half类型/half类型精度转换函数/__float2half_ro_sat.md">__float2half_ro_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p38011226172412"><a name="p38011226172412"></a><a name="p38011226172412"></a>饱和模式下获取输入遵循CAST_ODD模式转换成的半精度浮点数。</p>
</td>
</tr>
<tr id="row6402948589"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p14705134518317"><a name="p14705134518317"></a><a name="p14705134518317"></a><a href="../数学函数/half类型/half类型精度转换函数/__float22half2_ro.md">__float22half2_ro</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p380410264244"><a name="p380410264244"></a><a name="p380410264244"></a>获取输入的两个分量遵循CAST_ODD模式转换成的half2类型数据。</p>
</td>
</tr>
<tr id="row11815144818816"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1956125221011"><a name="p1956125221011"></a><a name="p1956125221011"></a><a href="../数学函数/half类型/half类型精度转换函数/__float22half2_ro_sat.md">__float22half2_ro_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p7804132652418"><a name="p7804132652418"></a><a name="p7804132652418"></a>饱和模式下获取输入的两个分量遵循CAST_ODD模式转换成的half2类型数据。</p>
</td>
</tr>
<tr id="row1653035814813"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p84021236104720"><a name="p84021236104720"></a><a name="p84021236104720"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2float.md">__half2float</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p78065268240"><a name="p78065268240"></a><a name="p78065268240"></a>获取输入转换成的浮点数。</p>
</td>
</tr>
<tr id="row47254581782"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p26230334318"><a name="p26230334318"></a><a name="p26230334318"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2half_rn.md">__half2half_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p380612260248"><a name="p380612260248"></a><a name="p380612260248"></a>获取输入遵循CAST_RINT模式取整后的half类型数据。</p>
</td>
</tr>
<tr id="row1910165816818"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p172331663439"><a name="p172331663439"></a><a name="p172331663439"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2half_rz.md">__half2half_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p480672610242"><a name="p480672610242"></a><a name="p480672610242"></a>获取输入遵循CAST_TRUNC模式取整后的half类型数据。</p>
</td>
</tr>
<tr id="row114019491085"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1366710817435"><a name="p1366710817435"></a><a name="p1366710817435"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2half_rd.md">__half2half_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1806162652412"><a name="p1806162652412"></a><a name="p1806162652412"></a>获取输入遵循CAST_FLOOR模式取整后的half类型数据。</p>
</td>
</tr>
<tr id="row152571249887"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p8638812154313"><a name="p8638812154313"></a><a name="p8638812154313"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2half_ru.md">__half2half_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p980611264244"><a name="p980611264244"></a><a name="p980611264244"></a>获取输入遵循CAST_CEIL模式取整后的half类型数据。</p>
</td>
</tr>
<tr id="row16461349481"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p8657111094313"><a name="p8657111094313"></a><a name="p8657111094313"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2half_rna.md">__half2half_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p128061626172410"><a name="p128061626172410"></a><a name="p128061626172410"></a>获取输入遵循CAST_ROUND模式取整后的half类型数据。</p>
</td>
</tr>
<tr id="row13667194911812"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p940253618477"><a name="p940253618477"></a><a name="p940253618477"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2uint_rn.md">__half2uint_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p7806626112418"><a name="p7806626112418"></a><a name="p7806626112418"></a>获取输入遵循CAST_RINT模式转换成的无符号整数。</p>
</td>
</tr>
<tr id="row8849204915818"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p087218441464"><a name="p087218441464"></a><a name="p087218441464"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2uint_rz.md">__half2uint_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p680710268249"><a name="p680710268249"></a><a name="p680710268249"></a>获取输入遵循<span>CAST_TRUNC</span>模式转换成的无符号整数。</p>
</td>
</tr>
<tr id="row1939750586"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1040253613476"><a name="p1040253613476"></a><a name="p1040253613476"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2uint_rd.md">__half2uint_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1080762612417"><a name="p1080762612417"></a><a name="p1080762612417"></a>获取输入遵循CAST_FLOOR模式转换成的无符号整数。</p>
</td>
</tr>
<tr id="row1123725012813"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p164026367478"><a name="p164026367478"></a><a name="p164026367478"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2uint_ru.md">__half2uint_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p380782611242"><a name="p380782611242"></a><a name="p380782611242"></a>获取输入遵循CAST_CEIL模式转换成的无符号整数。</p>
</td>
</tr>
<tr id="row13444185016812"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p18671145510249"><a name="p18671145510249"></a><a name="p18671145510249"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2uint_rna.md">__half2uint_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p6807192642416"><a name="p6807192642416"></a><a name="p6807192642416"></a><span>获取输入遵循CAST_ROUND模式转换成的无符号整数。</span></p>
</td>
</tr>
<tr id="row1928782851514"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p54035364478"><a name="p54035364478"></a><a name="p54035364478"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2int_rn.md">__half2int_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p780722662419"><a name="p780722662419"></a><a name="p780722662419"></a>获取输入遵循CAST_RINT模式转换成的有符号整数。</p>
</td>
</tr>
<tr id="row9857728181519"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p15403153684711"><a name="p15403153684711"></a><a name="p15403153684711"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2int_rz.md">__half2int_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p208071726162419"><a name="p208071726162419"></a><a name="p208071726162419"></a>获取输入遵循<span>CAST_TRUNC</span>模式转换成的有符号整数。</p>
</td>
</tr>
<tr id="row1647132917159"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p13403163615474"><a name="p13403163615474"></a><a name="p13403163615474"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2int_rd.md">__half2int_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p15807112617244"><a name="p15807112617244"></a><a name="p15807112617244"></a>获取输入遵循<span>CAST_FLOOR</span>模式转换成的有符号整数。</p>
</td>
</tr>
<tr id="row151120306151"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p340393612478"><a name="p340393612478"></a><a name="p340393612478"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2int_ru.md">__half2int_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p10807226192415"><a name="p10807226192415"></a><a name="p10807226192415"></a>获取输入遵循<span>CAST_CEIL</span>模式转换成的有符号整数。</p>
</td>
</tr>
<tr id="row105671330161510"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p170715817249"><a name="p170715817249"></a><a name="p170715817249"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2int_rna.md">__half2int_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p188071226162415"><a name="p188071226162415"></a><a name="p188071226162415"></a>获取输入遵循<span>CAST_ROUND</span>模式转换成的有符号整数。</p>
</td>
</tr>
<tr id="row192765327154"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p14403193624713"><a name="p14403193624713"></a><a name="p14403193624713"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2ull_rn.md">__half2ull_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p10807162620241"><a name="p10807162620241"></a><a name="p10807162620241"></a>获取输入遵循CAST_RINT模式转换成的64位无符号整数。</p>
</td>
</tr>
<tr id="row47561532131515"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p18403183611479"><a name="p18403183611479"></a><a name="p18403183611479"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2ull_rz.md">__half2ull_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p78081026112412"><a name="p78081026112412"></a><a name="p78081026112412"></a>获取输入遵循<span>CAST_TRUNC</span>模式转换成的64位无符号整数。</p>
</td>
</tr>
<tr id="row928218338151"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p184033366476"><a name="p184033366476"></a><a name="p184033366476"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2ull_rd.md">__half2ull_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p7808182612246"><a name="p7808182612246"></a><a name="p7808182612246"></a>获取输入遵循<span>CAST_FLOOR</span>模式转换成的64位无符号整数。</p>
</td>
</tr>
<tr id="row1780673316156"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1740312366473"><a name="p1740312366473"></a><a name="p1740312366473"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2ull_ru.md">__half2ull_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p138088265243"><a name="p138088265243"></a><a name="p138088265243"></a>获取输入遵循<span>CAST_CEIL</span>模式转换成的64位无符号整数。</p>
</td>
</tr>
<tr id="row1748293491515"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p842616213251"><a name="p842616213251"></a><a name="p842616213251"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2ull_rna.md">__half2ull_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p8808826102413"><a name="p8808826102413"></a><a name="p8808826102413"></a>获取输入遵循<span>CAST_ROUND</span>模式转换成的64位无符号整数。</p>
</td>
</tr>
<tr id="row1672813351158"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p4403123617475"><a name="p4403123617475"></a><a name="p4403123617475"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2ll_rn.md">__half2ll_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p11808192611244"><a name="p11808192611244"></a><a name="p11808192611244"></a>获取输入遵循CAST_RINT模式转换成的64位有符号整数。</p>
</td>
</tr>
<tr id="row8251123619150"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p6403736174713"><a name="p6403736174713"></a><a name="p6403736174713"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2ll_rz.md">__half2ll_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p138081226142416"><a name="p138081226142416"></a><a name="p138081226142416"></a>获取输入遵循<span>CAST_TRUNC</span>模式转换成的64位有符号整数。</p>
</td>
</tr>
<tr id="row480943620158"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p15403153684713"><a name="p15403153684713"></a><a name="p15403153684713"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2ll_rd.md">__half2ll_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p98081326102417"><a name="p98081326102417"></a><a name="p98081326102417"></a>获取输入遵循CAST_FLOOR模式转换成的64位有符号整数。</p>
</td>
</tr>
<tr id="row83853374159"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p340433634711"><a name="p340433634711"></a><a name="p340433634711"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2ll_ru.md">__half2ll_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p480882672414"><a name="p480882672414"></a><a name="p480882672414"></a>获取输入遵循CAST_CEIL模式转换成的64位有符号整数。</p>
</td>
</tr>
<tr id="row21511381157"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p16350136102510"><a name="p16350136102510"></a><a name="p16350136102510"></a><a href="../数学函数/half类型/half类型精度转换函数/__half2ll_rna.md">__half2ll_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p380842642420"><a name="p380842642420"></a><a name="p380842642420"></a>获取输入遵循CAST_ROUND模式转换成的64位有符号整数。</p>
</td>
</tr>
<tr id="row970315911914"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1404036184712"><a name="p1404036184712"></a><a name="p1404036184712"></a><a href="../数学函数/half类型/half类型精度转换函数/__bfloat162half_rn.md">__bfloat162half_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p8809162614246"><a name="p8809162614246"></a><a name="p8809162614246"></a>获取输入遵循CAST_RINT模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row172471510161915"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1430222854418"><a name="p1430222854418"></a><a name="p1430222854418"></a><a href="../数学函数/half类型/half类型精度转换函数/__bfloat162half_rn_sat.md">__bfloat162half_rn_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p2809326112414"><a name="p2809326112414"></a><a name="p2809326112414"></a>饱和模式下获取输入遵循CAST_RINT模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row20671141061912"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p6404163619470"><a name="p6404163619470"></a><a name="p6404163619470"></a><a href="../数学函数/half类型/half类型精度转换函数/__bfloat162half_rz.md">__bfloat162half_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p4809182682416"><a name="p4809182682416"></a><a name="p4809182682416"></a>获取输入遵循CAST_TRUNC模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row1265151111194"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p13861831174415"><a name="p13861831174415"></a><a name="p13861831174415"></a><a href="../数学函数/half类型/half类型精度转换函数/__bfloat162half_rz_sat.md">__bfloat162half_rz_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p138093263249"><a name="p138093263249"></a><a name="p138093263249"></a>饱和模式下获取输入遵循CAST_TRUNC模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row2517181110199"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1940433611475"><a name="p1940433611475"></a><a name="p1940433611475"></a><a href="../数学函数/half类型/half类型精度转换函数/__bfloat162half_rd.md">__bfloat162half_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p5809112620249"><a name="p5809112620249"></a><a name="p5809112620249"></a>获取输入遵循CAST_FLOOR模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row589812111196"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1663263304412"><a name="p1663263304412"></a><a name="p1663263304412"></a><a href="../数学函数/half类型/half类型精度转换函数/__bfloat162half_rd_sat.md">__bfloat162half_rd_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p198099267241"><a name="p198099267241"></a><a name="p198099267241"></a>饱和模式下获取输入遵循CAST_FLOOR模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row1930511126192"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p10404153619471"><a name="p10404153619471"></a><a name="p10404153619471"></a><a href="../数学函数/half类型/half类型精度转换函数/__bfloat162half_ru.md">__bfloat162half_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p180912260249"><a name="p180912260249"></a><a name="p180912260249"></a>获取输入遵循CAST_CEIL模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row1667971217191"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p5961153519445"><a name="p5961153519445"></a><a name="p5961153519445"></a><a href="../数学函数/half类型/half类型精度转换函数/__bfloat162half_ru_sat.md">__bfloat162half_ru_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p280922612242"><a name="p280922612242"></a><a name="p280922612242"></a>饱和模式下获取输入遵循CAST_CEIL模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row1798121391915"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1274189122515"><a name="p1274189122515"></a><a name="p1274189122515"></a><a href="../数学函数/half类型/half类型精度转换函数/__bfloat162half_rna.md">__bfloat162half_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1080992614249"><a name="p1080992614249"></a><a name="p1080992614249"></a>获取输入遵循CAST_ROUND模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row7533181312198"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p20159103815440"><a name="p20159103815440"></a><a name="p20159103815440"></a><a href="../数学函数/half类型/half类型精度转换函数/__bfloat162half_rna_sat.md">__bfloat162half_rna_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p4809172618249"><a name="p4809172618249"></a><a name="p4809172618249"></a>饱和模式下获取输入遵循CAST_ROUND模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row1164614389206"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p5406123618479"><a name="p5406123618479"></a><a name="p5406123618479"></a><a href="../数学函数/half类型/half类型精度转换函数/__uint2half_rn.md">__uint2half_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p10813162616242"><a name="p10813162616242"></a><a name="p10813162616242"></a>获取输入遵循CAST_RINT模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row1342133982011"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p7722145711343"><a name="p7722145711343"></a><a name="p7722145711343"></a><a href="../数学函数/half类型/half类型精度转换函数/__uint2half_rn_sat.md">__uint2half_rn_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p10813826172419"><a name="p10813826172419"></a><a name="p10813826172419"></a><span>饱和模式下获取输入的uint32数据转换成的half数据，并遵循CAST_RINT模式</span>。</p>
</td>
</tr>
<tr id="row83661839202011"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12406153614471"><a name="p12406153614471"></a><a name="p12406153614471"></a><a href="../数学函数/half类型/half类型精度转换函数/__uint2half_rz.md">__uint2half_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p15784855318"><a name="p15784855318"></a><a name="p15784855318"></a>获取输入遵循CAST_TRUNC模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row11748143972015"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12500282357"><a name="p12500282357"></a><a name="p12500282357"></a><a href="../数学函数/half类型/half类型精度转换函数/__uint2half_rz_sat.md">__uint2half_rz_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p78131826112411"><a name="p78131826112411"></a><a name="p78131826112411"></a><span>饱和模式下获取输入的uint32数据转换成的half数据，并遵循CAST_TRUNC模式</span>。</p>
</td>
</tr>
<tr id="row12236164032014"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1640763614720"><a name="p1640763614720"></a><a name="p1640763614720"></a><a href="../数学函数/half类型/half类型精度转换函数/__uint2half_rd.md">__uint2half_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p4813326192419"><a name="p4813326192419"></a><a name="p4813326192419"></a>获取输入遵循CAST_FLOOR模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row66091340172011"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p17407021123518"><a name="p17407021123518"></a><a name="p17407021123518"></a><a href="../数学函数/half类型/half类型精度转换函数/__uint2half_rd_sat.md">__uint2half_rd_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p14813112614249"><a name="p14813112614249"></a><a name="p14813112614249"></a><span>饱和模式下获取输入的uint32数据转换成的half数据，并遵循CAST_FLOOR模式</span>。</p>
</td>
</tr>
<tr id="row912364114200"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p104077362478"><a name="p104077362478"></a><a name="p104077362478"></a><a href="../数学函数/half类型/half类型精度转换函数/__uint2half_ru.md">__uint2half_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p148131260243"><a name="p148131260243"></a><a name="p148131260243"></a>获取输入遵循CAST_CEIL模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row853444111208"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p584192743518"><a name="p584192743518"></a><a name="p584192743518"></a><a href="../数学函数/half类型/half类型精度转换函数/__uint2half_ru_sat.md">__uint2half_ru_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1581652602417"><a name="p1581652602417"></a><a name="p1581652602417"></a><span>饱和模式下获取输入的uint32数据转换成的half数据，并遵循CAST_CEIL模式</span>。</p>
</td>
</tr>
<tr id="row7523422208"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p7456630202512"><a name="p7456630202512"></a><a name="p7456630202512"></a><a href="../数学函数/half类型/half类型精度转换函数/__uint2half_rna.md">__uint2half_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p081682622411"><a name="p081682622411"></a><a name="p081682622411"></a>获取输入遵循CAST_ROUND模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row1943454214209"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1925333103518"><a name="p1925333103518"></a><a name="p1925333103518"></a><a href="../数学函数/half类型/half类型精度转换函数/__uint2half_rna_sat.md">__uint2half_rna_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p168167266243"><a name="p168167266243"></a><a name="p168167266243"></a><span>饱和模式下获取输入的uint32数据转换成的half数据，并遵循CAST_ROUND模式</span>。</p>
</td>
</tr>
<tr id="row1229525916258"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p17408143674715"><a name="p17408143674715"></a><a name="p17408143674715"></a><a href="../数学函数/half类型/half类型精度转换函数/__int2half_rn.md">__int2half_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1581817269248"><a name="p1581817269248"></a><a name="p1581817269248"></a>获取输入遵循CAST_RINT模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row13709559122516"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p153743475353"><a name="p153743475353"></a><a name="p153743475353"></a><a href="../数学函数/half类型/half类型精度转换函数/__int2half_rn_sat.md">__int2half_rn_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p881822610242"><a name="p881822610242"></a><a name="p881822610242"></a><span>饱和模式下获取输入的int32数据转换成的half数据，并遵循CAST_RINT模式</span>。</p>
</td>
</tr>
<tr id="row10841306266"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1340873674712"><a name="p1340873674712"></a><a name="p1340873674712"></a><a href="../数学函数/half类型/half类型精度转换函数/__int2half_rz.md">__int2half_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p28181926162412"><a name="p28181926162412"></a><a name="p28181926162412"></a>获取输入遵循CAST_TRUNC模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row244215042613"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p89356538355"><a name="p89356538355"></a><a name="p89356538355"></a><a href="../数学函数/half类型/half类型精度转换函数/__int2half_rz_sat.md">__int2half_rz_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p17818192619245"><a name="p17818192619245"></a><a name="p17818192619245"></a><span>饱和模式下获取输入的int32数据转换成的half数据，并遵循CAST_TRUNC模式</span>。</p>
</td>
</tr>
<tr id="row128787022616"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p194081236174713"><a name="p194081236174713"></a><a name="p194081236174713"></a><a href="../数学函数/half类型/half类型精度转换函数/__int2half_rd.md">__int2half_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1481819261243"><a name="p1481819261243"></a><a name="p1481819261243"></a>获取输入遵循CAST_FLOOR模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row1128441112611"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p353855673516"><a name="p353855673516"></a><a name="p353855673516"></a><a href="../数学函数/half类型/half类型精度转换函数/__int2half_rd_sat.md">__int2half_rd_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1381818269242"><a name="p1381818269242"></a><a name="p1381818269242"></a><span>饱和模式下获取输入的int32数据转换成的half数据，并遵循CAST_FLOOR模式</span>。</p>
</td>
</tr>
<tr id="row069671152610"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p164081636124717"><a name="p164081636124717"></a><a name="p164081636124717"></a><a href="../数学函数/half类型/half类型精度转换函数/__int2half_ru.md">__int2half_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p6819192615241"><a name="p6819192615241"></a><a name="p6819192615241"></a>获取输入遵循CAST_CEIL模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row161103213268"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p9218115913356"><a name="p9218115913356"></a><a name="p9218115913356"></a><a href="../数学函数/half类型/half类型精度转换函数/__int2half_ru_sat.md">__int2half_ru_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p881922618242"><a name="p881922618242"></a><a name="p881922618242"></a><span>饱和模式下获取输入的int32数据转换成的half数据，并遵循CAST_CEIL模式</span>。</p>
</td>
</tr>
<tr id="row145353292612"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1394013386252"><a name="p1394013386252"></a><a name="p1394013386252"></a><a href="../数学函数/half类型/half类型精度转换函数/__int2half_rna.md">__int2half_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1881962611242"><a name="p1881962611242"></a><a name="p1881962611242"></a>获取输入遵循CAST_ROUND模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row1598219212267"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p449514111365"><a name="p449514111365"></a><a name="p449514111365"></a><a href="../数学函数/half类型/half类型精度转换函数/__int2half_rna_sat.md">__int2half_rna_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1181952614242"><a name="p1181952614242"></a><a name="p1181952614242"></a><span>饱和模式下获取输入的int32数据转换成的half数据，并遵循CAST_ROUND模式</span>。</p>
</td>
</tr>
<tr id="row207331834324"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p13409936104712"><a name="p13409936104712"></a><a name="p13409936104712"></a><a href="../数学函数/half类型/half类型精度转换函数/__ull2half_rn.md">__ull2half_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p3820162632417"><a name="p3820162632417"></a><a name="p3820162632417"></a>获取输入遵循CAST_RINT模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row11760420323"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p34094362477"><a name="p34094362477"></a><a name="p34094362477"></a><a href="../数学函数/half类型/half类型精度转换函数/__ull2half_rz.md">__ull2half_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p382132615248"><a name="p382132615248"></a><a name="p382132615248"></a>获取输入遵循CAST_TRUNC模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row17503174173219"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p194101336164717"><a name="p194101336164717"></a><a name="p194101336164717"></a><a href="../数学函数/half类型/half类型精度转换函数/__ull2half_rd.md">__ull2half_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p4286101219815"><a name="p4286101219815"></a><a name="p4286101219815"></a>获取输入遵循CAST_FLOOR模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row138644473210"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p17410123617479"><a name="p17410123617479"></a><a name="p17410123617479"></a><a href="../数学函数/half类型/half类型精度转换函数/__ull2half_ru.md">__ull2half_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p782192611240"><a name="p782192611240"></a><a name="p782192611240"></a>获取输入遵循CAST_CEIL模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row1833695123219"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p85461548102519"><a name="p85461548102519"></a><a name="p85461548102519"></a><a href="../数学函数/half类型/half类型精度转换函数/__ull2half_rna.md">__ull2half_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p16821102612245"><a name="p16821102612245"></a><a name="p16821102612245"></a>获取输入遵循CAST_ROUND模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row181217260354"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p15411173619473"><a name="p15411173619473"></a><a name="p15411173619473"></a><a href="../数学函数/half类型/half类型精度转换函数/__ll2half_rn.md">__ll2half_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p18822162692415"><a name="p18822162692415"></a><a name="p18822162692415"></a>获取输入遵循CAST_RINT模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row52581427143515"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1941118362474"><a name="p1941118362474"></a><a name="p1941118362474"></a><a href="../数学函数/half类型/half类型精度转换函数/__ll2half_rz.md">__ll2half_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p20823826112418"><a name="p20823826112418"></a><a name="p20823826112418"></a>获取输入遵循CAST_TRUNC模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row12616627103510"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1411203610475"><a name="p1411203610475"></a><a name="p1411203610475"></a><a href="../数学函数/half类型/half类型精度转换函数/__ll2half_rd.md">__ll2half_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p298285041610"><a name="p298285041610"></a><a name="p298285041610"></a>获取输入遵循CAST_FLOOR模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row121244283358"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1411103614473"><a name="p1411103614473"></a><a name="p1411103614473"></a><a href="../数学函数/half类型/half类型精度转换函数/__ll2half_ru.md">__ll2half_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p982332662410"><a name="p982332662410"></a><a name="p982332662410"></a>获取输入遵循CAST_CEIL模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row185082816354"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p163965814259"><a name="p163965814259"></a><a name="p163965814259"></a><a href="../数学函数/half类型/half类型精度转换函数/__ll2half_rna.md">__ll2half_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p8823112662417"><a name="p8823112662417"></a><a name="p8823112662417"></a>获取输入遵循CAST_ROUND模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row7245194613712"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p675722216157"><a name="p675722216157"></a><a name="p675722216157"></a><a href="../数学函数/half类型/half类型精度转换函数/__floats2half2_rn.md">__floats2half2_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1182712612249"><a name="p1182712612249"></a><a name="p1182712612249"></a>将输入的数据x，y遵循CAST_RINT模式分别转换为bfloat16类型并填充到half2的前后两部分，返回转换后的half2类型数据。</p>
</td>
</tr>
<tr id="row1559004643713"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p125155917443"><a name="p125155917443"></a><a name="p125155917443"></a><a href="../数学函数/half类型/half类型精度转换函数/__float22half2_rn.md">__float22half2_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p18827182682419"><a name="p18827182682419"></a><a name="p18827182682419"></a>将float2类型数据遵循CAST_RINT模式转换为half2类型，返回转换后的half2类型数据。</p>
</td>
</tr>
<tr id="row1997164619377"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p14730318111610"><a name="p14730318111610"></a><a name="p14730318111610"></a><a href="../数学函数/half类型/half类型精度转换函数/__low2half.md">__low2half</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p13827112619241"><a name="p13827112619241"></a><a name="p13827112619241"></a>返回输入数据的低16位。</p>
</td>
</tr>
<tr id="row941418479372"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p14281132110162"><a name="p14281132110162"></a><a name="p14281132110162"></a><a href="../数学函数/half类型/half类型精度转换函数/__low2half2.md">__low2half2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1582715264243"><a name="p1582715264243"></a><a name="p1582715264243"></a>将输入数据的低16位填充到half2并返回。</p>
</td>
</tr>
<tr id="row16916447123720"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1256155216173"><a name="p1256155216173"></a><a name="p1256155216173"></a><a href="../数学函数/half类型/half类型精度转换函数/__high2half.md">__high2half</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p168271226112411"><a name="p168271226112411"></a><a name="p168271226112411"></a>提取输入half2的高16位，并返回</p>
</td>
</tr>
<tr id="row5618104843717"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p948711505171"><a name="p948711505171"></a><a name="p948711505171"></a><a href="../数学函数/half类型/half类型精度转换函数/__high2half2.md">__high2half2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1782762619247"><a name="p1782762619247"></a><a name="p1782762619247"></a>将输入数据的的高16位填充到half2并返回结果。</p>
</td>
</tr>
<tr id="row158154953718"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p13540173112612"><a name="p13540173112612"></a><a name="p13540173112612"></a><a href="../数学函数/half类型/half类型精度转换函数/__highs2half2.md">__highs2half2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p20827192613243"><a name="p20827192613243"></a><a name="p20827192613243"></a>分别提取两个half2输入的高16位，并填充到half2中。返回填充后的数据。</p>
</td>
</tr>
<tr id="row16471144993711"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1037813488177"><a name="p1037813488177"></a><a name="p1037813488177"></a><a href="../数学函数/half类型/half类型精度转换函数/__lows2half2.md">__lows2half2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p482820264245"><a name="p482820264245"></a><a name="p482820264245"></a>分别提取两个half2输入的低16位，并填充到half2中。返回填充后的数据。</p>
</td>
</tr>
<tr id="row286119493378"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p529822681818"><a name="p529822681818"></a><a name="p529822681818"></a><a href="../数学函数/half类型/half类型精度转换函数/__halves2half2.md">__halves2half2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1382818261244"><a name="p1382818261244"></a><a name="p1382818261244"></a>将输入的数据分别填充为half2前后两个分量，返回填充后数据。</p>
</td>
</tr>
<tr id="row329805013374"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p10927023151818"><a name="p10927023151818"></a><a name="p10927023151818"></a><a href="../数学函数/half类型/half类型精度转换函数/__half22float2.md">__half22float2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p48281265241"><a name="p48281265241"></a><a name="p48281265241"></a>将half2的两个分量分别转换为float，并填充到float2返回。</p>
</td>
</tr>
<tr id="row1288081519396"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p73293236199"><a name="p73293236199"></a><a name="p73293236199"></a><a href="../数学函数/half类型/half类型精度转换函数/__ushort_as_half.md">__ushort_as_half</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1482992652411"><a name="p1482992652411"></a><a name="p1482992652411"></a><span>将unsigned short int的按位重新解释为half，即将unsigned short int的数据存储的位按照half的格式进行读取</span>。</p>
</td>
</tr>
</tbody>
</table>

**表 11**  half2类型算术函数

<a name="table569680104113"></a>
<table><thead align="left"><tr id="row1469610154117"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p5696130164119"><a name="p5696130164119"></a><a name="p5696130164119"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p169712024119"><a name="p169712024119"></a><a name="p169712024119"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row146971004413"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p4697130124114"><a name="p4697130124114"></a><a name="p4697130124114"></a><a href="../数学函数/half类型/half2类型算术函数/__haddx2.md">__haddx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1069716004117"><a name="p1069716004117"></a><a name="p1069716004117"></a>计算两个half2类型数据各分量的相加结果，并遵循CAST_RINT模式舍入。</p>
</td>
</tr>
<tr id="row116976010415"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p9697170184114"><a name="p9697170184114"></a><a name="p9697170184114"></a><a href="../数学函数/half类型/half2类型算术函数/__hsubx2.md">__hsubx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1269717064110"><a name="p1269717064110"></a><a name="p1269717064110"></a>计算两个half2类型数据各分量的相减结果，并遵循CAST_RINT模式舍入。</p>
</td>
</tr>
<tr id="row169730184113"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p6697120174115"><a name="p6697120174115"></a><a name="p6697120174115"></a><a href="../数学函数/half类型/half2类型算术函数/__hmulx2.md">__hmulx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1369710016412"><a name="p1369710016412"></a><a name="p1369710016412"></a>计算两个half2类型数据各分量的相乘结果，并遵循CAST_RINT模式舍入。</p>
</td>
</tr>
<tr id="row469720015411"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p146975034119"><a name="p146975034119"></a><a name="p146975034119"></a><a href="../数学函数/half类型/half2类型算术函数/__hdivx2.md">__hdivx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p36975094114"><a name="p36975094114"></a><a name="p36975094114"></a>计算两个half2类型数据各分量的相除结果，并遵循CAST_RINT模式舍入。</p>
</td>
</tr>
<tr id="row146972011415"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p06978014115"><a name="p06978014115"></a><a name="p06978014115"></a><a href="../数学函数/half类型/half2类型算术函数/__habsx2.md">__habsx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p176973094115"><a name="p176973094115"></a><a name="p176973094115"></a>计算输入half2类型数据各分量的绝对值。</p>
</td>
</tr>
<tr id="row176975074113"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p169740194116"><a name="p169740194116"></a><a name="p169740194116"></a><a href="../数学函数/half类型/half2类型算术函数/__hfmax2.md">__hfmax2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p4647164116237"><a name="p4647164116237"></a><a name="p4647164116237"></a>计算两个half2类型数据各分量的乘加的结果（前两个输入相乘后与第三个输入相加），并遵循CAST_RINT模式舍入。</p>
</td>
</tr>
<tr id="row8697200164111"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12697905414"><a name="p12697905414"></a><a name="p12697905414"></a><a href="../数学函数/half类型/half2类型算术函数/__hnegx2.md">__hnegx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1369714011415"><a name="p1369714011415"></a><a name="p1369714011415"></a>获取输入half2类型数据各分量的负值。</p>
</td>
</tr>
<tr id="row66971900411"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p2069719013412"><a name="p2069719013412"></a><a name="p2069719013412"></a><a href="../数学函数/half类型/half2类型算术函数/__hfmax2_relu.md">__hfmax2_relu</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p14697209414"><a name="p14697209414"></a><a name="p14697209414"></a>计算两个half2类型数据各分量的乘加的结果（前两个输入相乘后与第三个输入相加），并遵循CAST_RINT模式舍入。负数结果置为0。</p>
</td>
</tr>
<tr id="row469717074117"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p8697140174112"><a name="p8697140174112"></a><a name="p8697140174112"></a><a href="../数学函数/half类型/half2类型算术函数/__hcmadd.md">__hcmadd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p18109185611236"><a name="p18109185611236"></a><a name="p18109185611236"></a>将三个half2输入视为复数（第一个分量为实部，第二个分量为虚部），执行复数乘加运算x*y+z。</p>
</td>
</tr>
</tbody>
</table>

**表 12**  half2类型比较函数

<a name="table1565715521823"></a>
<table><thead align="left"><tr id="row13657052729"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="p9657652627"><a name="p9657652627"></a><a name="p9657652627"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="p20657145210211"><a name="p20657145210211"></a><a name="p20657145210211"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row7657352927"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p174914815413"><a name="p174914815413"></a><a name="p174914815413"></a><a href="../数学函数/half类型/half2类型比较函数/__hbeqx2.md">__hbeqx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1365712521921"><a name="p1365712521921"></a><a name="p1365712521921"></a>比较两个half2类型数据的两个分量是否相等，仅当两个分量均相等时返回true。</p>
</td>
</tr>
<tr id="row14657115219212"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11749884413"><a name="p11749884413"></a><a name="p11749884413"></a><a href="../数学函数/half类型/half2类型比较函数/__hbnex2.md">__hbnex2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p9657205210219"><a name="p9657205210219"></a><a name="p9657205210219"></a>比较两个half2类型数据的两个分量是否不相等，仅当两个分量均不相等时返回true。</p>
</td>
</tr>
<tr id="row66574521823"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p4750181941"><a name="p4750181941"></a><a name="p4750181941"></a><a href="../数学函数/half类型/half2类型比较函数/__hblex2.md">__hblex2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1665755211210"><a name="p1665755211210"></a><a name="p1665755211210"></a>比较两个half2类型数据的两个分量，仅当两个分量均满足第一个数小于或等于第二个数时返回true。</p>
</td>
</tr>
<tr id="row20657652824"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p2750581410"><a name="p2750581410"></a><a name="p2750581410"></a><a href="../数学函数/half类型/half2类型比较函数/__hbgex2.md">__hbgex2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p0657952227"><a name="p0657952227"></a><a name="p0657952227"></a>比较两个half2类型数据的两个分量，仅当两个分量均满足第一个数大于或等于第二个数时返回true。</p>
</td>
</tr>
<tr id="row2065717521324"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p7750481644"><a name="p7750481644"></a><a name="p7750481644"></a><a href="../数学函数/half类型/half2类型比较函数/__hbltx2.md">__hbltx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p365795214219"><a name="p365795214219"></a><a name="p365795214219"></a>比较两个half2类型数据的两个分量，仅当两个分量均满足第一个数小于第二个数时返回true。</p>
</td>
</tr>
<tr id="row86572521729"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p3750181644"><a name="p3750181644"></a><a name="p3750181644"></a><a href="../数学函数/half类型/half2类型比较函数/__hbgtx2.md">__hbgtx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p46571452828"><a name="p46571452828"></a><a name="p46571452828"></a>比较两个half2类型数据的两个分量，仅当两个分量均满足第一个数大于第二个数时返回true。</p>
</td>
</tr>
<tr id="row149082046436"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p875014816417"><a name="p875014816417"></a><a name="p875014816417"></a><a href="../数学函数/half类型/half2类型比较函数/__hbequx2.md">__hbequx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p9908114613318"><a name="p9908114613318"></a><a name="p9908114613318"></a>比较两个half2类型数据的两个分量是否相等，当两个分量均相等时返回true。若任一输入的分量为nan，该分量的比较结果为true。</p>
</td>
</tr>
<tr id="row62228471838"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p2750198348"><a name="p2750198348"></a><a name="p2750198348"></a><a href="../数学函数/half类型/half2类型比较函数/__hbneux2.md">__hbneux2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1312725014216"><a name="p1312725014216"></a><a name="p1312725014216"></a>比较两个half2类型数据的两个分量是否不相等，当两个分量均不相等时返回true。若任一输入的分量为nan，该分量的比较结果为true。</p>
</td>
</tr>
<tr id="row3591847739"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p97501081418"><a name="p97501081418"></a><a name="p97501081418"></a><a href="../数学函数/half类型/half2类型比较函数/__hbleux2.md">__hbleux2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1699817558157"><a name="p1699817558157"></a><a name="p1699817558157"></a>比较两个half2类型数据的两个分量，当两个分量均满足第一个数小于或等于第二个数时返回true。若任一输入的分量为nan，该分量的比较结果为true。</p>
</td>
</tr>
<tr id="row19927482312"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p7750781441"><a name="p7750781441"></a><a name="p7750781441"></a><a href="../数学函数/half类型/half2类型比较函数/__hbgeux2.md">__hbgeux2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p16924486310"><a name="p16924486310"></a><a name="p16924486310"></a>比较两个half2类型数据的两个分量，当两个分量均满足第一个数大于或等于第二个数时返回true。若任一输入的分量为nan，该分量的比较结果为true。</p>
</td>
</tr>
<tr id="row85656481311"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p8750138540"><a name="p8750138540"></a><a name="p8750138540"></a><a href="../数学函数/half类型/half2类型比较函数/__hbltux2.md">__hbltux2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p102417501611"><a name="p102417501611"></a><a name="p102417501611"></a>比较两个half2类型数据的两个分量，当两个分量均满足第一个数小于第二个数时返回true。若任一输入的分量为nan，该分量的比较结果为true。</p>
</td>
</tr>
<tr id="row697711481933"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p17501981242"><a name="p17501981242"></a><a name="p17501981242"></a><a href="../数学函数/half类型/half2类型比较函数/__hbgtux2.md">__hbgtux2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1644018917164"><a name="p1644018917164"></a><a name="p1644018917164"></a>比较两个half2类型数据的两个分量，当两个分量均满足第一个数大于第二个数时返回true。若任一输入的分量为nan，该分量的比较结果为true。</p>
</td>
</tr>
<tr id="row1760794915314"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p167501981149"><a name="p167501981149"></a><a name="p167501981149"></a><a href="../数学函数/half类型/half2类型比较函数/__heqx2.md">__heqx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p360710491039"><a name="p360710491039"></a><a name="p360710491039"></a>比较两个half2类型数据的两个分量，如果分量相等，则对应比较结果为1.0，否则为0.0。</p>
</td>
</tr>
<tr id="row119679491638"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p177503816412"><a name="p177503816412"></a><a name="p177503816412"></a><a href="../数学函数/half类型/half2类型比较函数/__hnex2.md">__hnex2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p4967449635"><a name="p4967449635"></a><a name="p4967449635"></a>比较两个half2类型数据的两个分量，如果分量不相等，则对应比较结果为1.0，否则为0.0。</p>
</td>
</tr>
<tr id="row1132085015312"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p175015811417"><a name="p175015811417"></a><a name="p175015811417"></a><a href="../数学函数/half类型/half2类型比较函数/__hlex2.md">__hlex2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1132017501930"><a name="p1132017501930"></a><a name="p1132017501930"></a>比较两个half2类型数据的两个分量，如果分量满足第一个数小于或等于第二个数，则对应比较结果为1.0，否则为0.0。</p>
</td>
</tr>
<tr id="row1176216501333"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p775117810410"><a name="p775117810410"></a><a name="p775117810410"></a><a href="../数学函数/half类型/half2类型比较函数/__hgex2.md">__hgex2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p12762950533"><a name="p12762950533"></a><a name="p12762950533"></a>比较两个half2类型数据的两个分量，如果分量满足第一个数大于或等于第二个数，则对应比较结果为1.0，否则为0.0。</p>
</td>
</tr>
<tr id="row5152851838"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p157516814412"><a name="p157516814412"></a><a name="p157516814412"></a><a href="../数学函数/half类型/half2类型比较函数/__hltx2.md">__hltx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p11521511436"><a name="p11521511436"></a><a name="p11521511436"></a>比较两个half2类型数据的两个分量，如果分量满足第一个数小于第二个数，则对应比较结果为1.0，否则为0.0。</p>
</td>
</tr>
<tr id="row35126519320"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p67511781745"><a name="p67511781745"></a><a name="p67511781745"></a><a href="../数学函数/half类型/half2类型比较函数/__hgtx2.md">__hgtx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p35129511431"><a name="p35129511431"></a><a name="p35129511431"></a>比较两个half2类型数据的两个分量，如果分量满足第一个数大于第二个数，则对应比较结果为1.0，否则为0.0。</p>
</td>
</tr>
<tr id="row152328520320"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p18751188146"><a name="p18751188146"></a><a name="p18751188146"></a><a href="../数学函数/half类型/half2类型比较函数/__hequx2.md">__hequx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p16232175215313"><a name="p16232175215313"></a><a name="p16232175215313"></a>比较两个half2类型数据的两个分量，如果分量相等，则对应比较结果为1.0，否则为0.0。若任一输入的分量为nan，该分量的比较结果为1.0。</p>
</td>
</tr>
<tr id="row151015213313"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p4751086416"><a name="p4751086416"></a><a name="p4751086416"></a><a href="../数学函数/half类型/half2类型比较函数/__hneux2.md">__hneux2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1067655561719"><a name="p1067655561719"></a><a name="p1067655561719"></a>比较两个half2类型数据的两个分量，如果分量不相等，则对应比较结果为1.0，否则为0.0。若任一输入的分量为nan，该分量的比较结果为1.0。</p>
</td>
</tr>
<tr id="row127521352332"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p187514817417"><a name="p187514817417"></a><a name="p187514817417"></a><a href="../数学函数/half类型/half2类型比较函数/__hleux2.md">__hleux2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1375212521038"><a name="p1375212521038"></a><a name="p1375212521038"></a>比较两个half2类型数据的两个分量，如果分量满足第一个数小于或等于第二个数，则对应比较结果为1.0，否则为0.0。若任一输入的分量为nan，该分量的比较结果为1.0。</p>
</td>
</tr>
<tr id="row62005531638"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p775114816412"><a name="p775114816412"></a><a name="p775114816412"></a><a href="../数学函数/half类型/half2类型比较函数/__hgeux2.md">__hgeux2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1720013532031"><a name="p1720013532031"></a><a name="p1720013532031"></a>比较两个half2类型数据的两个分量，如果分量满足第一个数大于或等于第二个数，则对应比较结果为1.0，否则为0.0。若任一输入的分量为nan，该分量的比较结果为1.0。</p>
</td>
</tr>
<tr id="row33873531636"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p187511181747"><a name="p187511181747"></a><a name="p187511181747"></a><a href="../数学函数/half类型/half2类型比较函数/__hltux2.md">__hltux2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p8387125318312"><a name="p8387125318312"></a><a name="p8387125318312"></a>比较两个half2类型数据的两个分量，如果分量满足第一个数小于第二个数，则对应比较结果为1.0，否则为0.0。若任一输入的分量为nan，该分量的比较结果为1.0。</p>
</td>
</tr>
<tr id="row12612175315311"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p137511789416"><a name="p137511789416"></a><a name="p137511789416"></a><a href="../数学函数/half类型/half2类型比较函数/__hgtux2.md">__hgtux2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1693841661819"><a name="p1693841661819"></a><a name="p1693841661819"></a>比较两个half2类型数据的两个分量，如果分量满足第一个数大于第二个数，则对应比较结果为1.0，否则为0.0。若任一输入的分量为nan，该分量的比较结果为1.0。</p>
</td>
</tr>
<tr id="row2852195314312"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1875111817418"><a name="p1875111817418"></a><a name="p1875111817418"></a><a href="../数学函数/half类型/half2类型比较函数/__heqx2_mask.md">__heqx2_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p197123910389"><a name="p197123910389"></a><a name="p197123910389"></a>比较两个half2类型数据的两个分量，结果以unsigned int形式返回，低16位为第一个分量的掩码结果，高16位为第二个分量的掩码结果。如果分量相等，则对应16位掩码为0xFFFF，否则为0x0。</p>
</td>
</tr>
<tr id="row339218548311"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p197511289419"><a name="p197511289419"></a><a name="p197511289419"></a><a href="../数学函数/half类型/half2类型比较函数/__hnex2_mask.md">__hnex2_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p12392195413315"><a name="p12392195413315"></a><a name="p12392195413315"></a>比较两个half2类型数据的两个分量，结果以unsigned int形式返回，低16位为第一个分量的掩码结果，高16位为第二个分量的掩码结果。如果分量不相等，则对应16位掩码为0xFFFF，否则为0x0。</p>
</td>
</tr>
<tr id="row13602554238"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p197514817411"><a name="p197514817411"></a><a name="p197514817411"></a><a href="../数学函数/half类型/half2类型比较函数/__hlex2_mask.md">__hlex2_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p19949194419182"><a name="p19949194419182"></a><a name="p19949194419182"></a>比较两个half2类型数据的两个分量，结果以unsigned int形式返回，低16位为第一个分量的掩码结果，高16位为第二个分量的掩码结果。如果分量满足第一个数小于或等于第二个数，则对应16位掩码为0xFFFF，否则为0x0。</p>
</td>
</tr>
<tr id="row1579065416314"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p67511982418"><a name="p67511982418"></a><a name="p67511982418"></a><a href="../数学函数/half类型/half2类型比较函数/__hgex2_mask.md">__hgex2_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1370849201811"><a name="p1370849201811"></a><a name="p1370849201811"></a>比较两个half2类型数据的两个分量，结果以unsigned int形式返回，低16位为第一个分量的掩码结果，高16位为第二个分量的掩码结果。如果分量满足第一个数大于或等于第二个数，则对应16位掩码为0xFFFF，否则为0x0。</p>
</td>
</tr>
<tr id="row1427710551635"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p14751118745"><a name="p14751118745"></a><a name="p14751118745"></a><a href="../数学函数/half类型/half2类型比较函数/__hltx2_mask.md">__hltx2_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p927745519317"><a name="p927745519317"></a><a name="p927745519317"></a>比较两个half2类型数据的两个分量，结果以unsigned int形式返回，低16位为第一个分量的掩码结果，高16位为第二个分量的掩码结果。如果分量满足第一个数小于第二个数，则对应16位掩码为0xFFFF，否则为0x0。</p>
</td>
</tr>
<tr id="row12465155520317"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p147522080414"><a name="p147522080414"></a><a name="p147522080414"></a><a href="../数学函数/half类型/half2类型比较函数/__hgtx2_mask.md">__hgtx2_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p137031121919"><a name="p137031121919"></a><a name="p137031121919"></a>比较两个half2类型数据的两个分量，结果以unsigned int形式返回，低16位为第一个分量的掩码结果，高16位为第二个分量的掩码结果。如果分量满足第一个数大于第二个数，则对应16位掩码为0xFFFF，否则为0x0。</p>
</td>
</tr>
<tr id="row147041055835"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p975228240"><a name="p975228240"></a><a name="p975228240"></a><a href="../数学函数/half类型/half2类型比较函数/__hequx2_mask.md">__hequx2_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p10558125325516"><a name="p10558125325516"></a><a name="p10558125325516"></a>比较两个half2类型数据的两个分量，结果以unsigned int形式返回，低16位为第一个分量的掩码结果，高16位为第二个分量的掩码结果。如果分量相等，则对应16位掩码为0xFFFF，否则为0x0。若任一输入的分量为nan，对应16位掩码为0xFFFF。</p>
</td>
</tr>
<tr id="row102911565310"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p9752688414"><a name="p9752688414"></a><a name="p9752688414"></a><a href="../数学函数/half类型/half2类型比较函数/__hneux2_mask.md">__hneux2_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p18291756332"><a name="p18291756332"></a><a name="p18291756332"></a>比较两个half2类型数据的两个分量，结果以unsigned int形式返回，低16位为第一个分量的掩码结果，高16位为第二个分量的掩码结果。如果分量不相等，则对应16位掩码为0xFFFF，否则为0x0。若任一输入的分量为nan，对应16位掩码为0xFFFF。</p>
</td>
</tr>
<tr id="row1620013569312"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p177521481145"><a name="p177521481145"></a><a name="p177521481145"></a><a href="../数学函数/half类型/half2类型比较函数/__hleux2_mask.md">__hleux2_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p22003561639"><a name="p22003561639"></a><a name="p22003561639"></a>比较两个half2类型数据的两个分量，结果以unsigned int形式返回，低16位为第一个分量的掩码结果，高16位为第二个分量的掩码结果。如果分量满足第一个数小于或等于第二个数，则对应16位掩码为0xFFFF，否则为0x0。若任一输入的分量为nan，对应16位掩码为0xFFFF。</p>
</td>
</tr>
<tr id="row113871256732"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p375278842"><a name="p375278842"></a><a name="p375278842"></a><a href="../数学函数/half类型/half2类型比较函数/__hgeux2_mask.md">__hgeux2_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p938815561312"><a name="p938815561312"></a><a name="p938815561312"></a>比较两个half2类型数据的两个分量，结果以unsigned int形式返回，低16位为第一个分量的掩码结果，高16位为第二个分量的掩码结果。如果分量满足第一个数大于或等于第二个数，则对应16位掩码为0xFFFF，否则为0x0。若任一输入的分量为nan，对应16位掩码为0xFFFF。</p>
</td>
</tr>
<tr id="row6717195618310"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p3752882042"><a name="p3752882042"></a><a name="p3752882042"></a><a href="../数学函数/half类型/half2类型比较函数/__hltux2_mask.md">__hltux2_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1271795615317"><a name="p1271795615317"></a><a name="p1271795615317"></a>比较两个half2类型数据的两个分量，结果以unsigned int形式返回，低16位为第一个分量的掩码结果，高16位为第二个分量的掩码结果。如果分量满足第一个数小于第二个数，则对应16位掩码为0xFFFF，否则为0x0。若任一输入的分量为nan，对应16位掩码为0xFFFF。</p>
</td>
</tr>
<tr id="row1189011561037"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p177521081042"><a name="p177521081042"></a><a name="p177521081042"></a><a href="../数学函数/half类型/half2类型比较函数/__hgtux2_mask.md">__hgtux2_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p138902561539"><a name="p138902561539"></a><a name="p138902561539"></a>比较两个half2类型数据的两个分量，结果以unsigned int形式返回，低16位为第一个分量的掩码结果，高16位为第二个分量的掩码结果。如果分量满足第一个数大于第二个数，则对应16位掩码为0xFFFF，否则为0x0。若任一输入的分量为nan，对应16位掩码为0xFFFF。</p>
</td>
</tr>
<tr id="row977657232"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1752208941"><a name="p1752208941"></a><a name="p1752208941"></a><a href="../数学函数/half类型/half2类型比较函数/__isnanx2.md">__isnanx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p208486506196"><a name="p208486506196"></a><a name="p208486506196"></a>判断half2类型数据的两个分量是否为nan。</p>
</td>
</tr>
<tr id="row1066914264712"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p106701827472"><a name="p106701827472"></a><a name="p106701827472"></a><a href="../数学函数/half类型/half2类型比较函数/__hmaxx2.md">__hmaxx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p3670220473"><a name="p3670220473"></a><a name="p3670220473"></a>获取两个half2类型数据各分量的最大值。</p>
</td>
</tr>
<tr id="row3158946192517"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p191591746152513"><a name="p191591746152513"></a><a name="p191591746152513"></a><a href="../数学函数/half类型/half2类型比较函数/__hmaxx2_nan.md">__hmaxx2_nan</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1159946192518"><a name="p1159946192518"></a><a name="p1159946192518"></a>获取两个half2类型数据各分量的最大值。任一分量为nan时对应结果为nan。</p>
</td>
</tr>
<tr id="row430812468259"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p130874612517"><a name="p130874612517"></a><a name="p130874612517"></a><a href="../数学函数/half类型/half2类型比较函数/__hminx2.md">__hminx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p19308164614253"><a name="p19308164614253"></a><a name="p19308164614253"></a>获取两个half2类型数据各分量的最小值。</p>
</td>
</tr>
<tr id="row13497134610254"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p7497046112519"><a name="p7497046112519"></a><a name="p7497046112519"></a><a href="../数学函数/half类型/half2类型比较函数/__hminx2_nan.md">__hminx2_nan</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p149794602515"><a name="p149794602515"></a><a name="p149794602515"></a>获取两个half2类型数据各分量的最小值。任一分量为nan时对应结果为nan。</p>
</td>
</tr>
</tbody>
</table>

**表 13**  half2类型数学库函数

<a name="table18325144919420"></a>
<table><thead align="left"><tr id="row8325194910429"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p432517498429"><a name="p432517498429"></a><a name="p432517498429"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p1832544911421"><a name="p1832544911421"></a><a name="p1832544911421"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row73258494421"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1158968182016"><a name="p1158968182016"></a><a name="p1158968182016"></a><a href="../数学函数/half类型/half2类型数学库函数/h2tanh.md">h2tanh</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p14589188102011"><a name="p14589188102011"></a><a name="p14589188102011"></a>获取输入数据各元素的三角函数双曲正切值。</p>
</td>
</tr>
<tr id="row19325204954220"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p204799560420"><a name="p204799560420"></a><a name="p204799560420"></a><a href="../数学函数/half类型/half2类型数学库函数/h2exp.md">h2exp</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p13479056049"><a name="p13479056049"></a><a name="p13479056049"></a>指定输入x，对x的各元素，获取e的该元素次方。</p>
</td>
</tr>
<tr id="row183251249124217"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p147455509616"><a name="p147455509616"></a><a name="p147455509616"></a><a href="../数学函数/half类型/half2类型数学库函数/h2exp2.md">h2exp2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p2745145016616"><a name="p2745145016616"></a><a name="p2745145016616"></a>指定输入x，对x的各元素，获取2的该元素次方。</p>
</td>
</tr>
<tr id="row13251149204218"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p0569152118154"><a name="p0569152118154"></a><a name="p0569152118154"></a><a href="../数学函数/half类型/half2类型数学库函数/h2exp10.md">h2exp10</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p356912219157"><a name="p356912219157"></a><a name="p356912219157"></a>指定输入x，对x的各元素，获取10的该元素次方。</p>
</td>
</tr>
<tr id="row123251549144215"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p626615631511"><a name="p626615631511"></a><a name="p626615631511"></a><a href="../数学函数/half类型/half2类型数学库函数/h2log.md">h2log</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p18266185621513"><a name="p18266185621513"></a><a name="p18266185621513"></a>获取以e为底，输入数据各元素的对数。</p>
</td>
</tr>
<tr id="row13259492426"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p34761920131611"><a name="p34761920131611"></a><a name="p34761920131611"></a><a href="../数学函数/half类型/half2类型数学库函数/h2log2.md">h2log2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p194762206163"><a name="p194762206163"></a><a name="p194762206163"></a>获取以2为底，输入数据各元素的对数。</p>
</td>
</tr>
<tr id="row17326154910421"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p118592517166"><a name="p118592517166"></a><a name="p118592517166"></a><a href="../数学函数/half类型/half2类型数学库函数/h2log10.md">h2log10</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p485945171614"><a name="p485945171614"></a><a name="p485945171614"></a>获取以10为底，输入数据各元素的对数。</p>
</td>
</tr>
<tr id="row10326154915422"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12854122421717"><a name="p12854122421717"></a><a name="p12854122421717"></a><a href="../数学函数/half类型/half2类型数学库函数/h2cos.md">h2cos</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p285442471714"><a name="p285442471714"></a><a name="p285442471714"></a>获取输入数据各元素的三角函数余弦值。</p>
</td>
</tr>
<tr id="row17326114914428"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p757913480179"><a name="p757913480179"></a><a name="p757913480179"></a><a href="../数学函数/half类型/half2类型数学库函数/h2sin.md">h2sin</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p115807481174"><a name="p115807481174"></a><a name="p115807481174"></a>获取输入数据各元素的三角函数正弦值。</p>
</td>
</tr>
<tr id="row1432674915425"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p2698935171814"><a name="p2698935171814"></a><a name="p2698935171814"></a><a href="../数学函数/half类型/half2类型数学库函数/h2sqrt.md">h2sqrt</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p12698183541814"><a name="p12698183541814"></a><a name="p12698183541814"></a>获取输入数据x各元素的平方根。</p>
</td>
</tr>
<tr id="row183261349154212"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p56681056131817"><a name="p56681056131817"></a><a name="p56681056131817"></a><a href="../数学函数/half类型/half2类型数学库函数/h2rsqrt.md">h2rsqrt</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p466818566188"><a name="p466818566188"></a><a name="p466818566188"></a>获取输入数据x各元素的平方根的倒数。</p>
</td>
</tr>
<tr id="row15326549144217"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1274952271918"><a name="p1274952271918"></a><a name="p1274952271918"></a><a href="../数学函数/half类型/half2类型数学库函数/h2rcp.md">h2rcp</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p4749122251920"><a name="p4749122251920"></a><a name="p4749122251920"></a>获取输入数据x各元素的倒数。</p>
</td>
</tr>
<tr id="row63261493424"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p130849102011"><a name="p130849102011"></a><a name="p130849102011"></a><a href="../数学函数/half类型/half2类型数学库函数/h2rint.md">h2rint</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1901749162013"><a name="p1901749162013"></a><a name="p1901749162013"></a>获取与输入数据各元素最接近的整数，若存在两个同样接近的整数，则获取其中的偶数。</p>
</td>
</tr>
<tr id="row117611038165719"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p5972175362012"><a name="p5972175362012"></a><a name="p5972175362012"></a><a href="../数学函数/half类型/half2类型数学库函数/h2floor.md">h2floor</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p997245312209"><a name="p997245312209"></a><a name="p997245312209"></a>获取小于或等于输入数据各元素的最大整数值。</p>
</td>
</tr>
<tr id="row622153965714"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p17564125762010"><a name="p17564125762010"></a><a name="p17564125762010"></a><a href="../数学函数/half类型/half2类型数学库函数/h2ceil.md">h2ceil</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p125644579203"><a name="p125644579203"></a><a name="p125644579203"></a>获取大于或等于输入数据各元素的最小整数值。</p>
</td>
</tr>
<tr id="row11308113945713"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p182481506212"><a name="p182481506212"></a><a name="p182481506212"></a><a href="../数学函数/half类型/half2类型数学库函数/h2trunc.md">h2trunc</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p62485015213"><a name="p62485015213"></a><a name="p62485015213"></a>获取对输入数据各元素的浮点数截断后的整数。</p>
</td>
</tr>
</tbody>
</table>

**表 14**  bfloat16类型算术函数

<a name="table17736301002"></a>
<table><thead align="left"><tr id="row107361201807"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p1273612016010"><a name="p1273612016010"></a><a name="p1273612016010"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p67361201908"><a name="p67361201908"></a><a name="p67361201908"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1555715389209"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1998383916203"><a name="p1998383916203"></a><a name="p1998383916203"></a><a href="../数学函数/bfloat16类型/bfloat16类型算术函数/__habs-150.md">__habs</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1998383910207"><a name="p1998383910207"></a><a name="p1998383910207"></a>获取输入数据的绝对值。</p>
</td>
</tr>
<tr id="row163221238172018"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1698319397204"><a name="p1698319397204"></a><a name="p1698319397204"></a><a href="../数学函数/bfloat16类型/bfloat16类型算术函数/__hfma-151.md">__hfma</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p17983539172019"><a name="p17983539172019"></a><a name="p17983539172019"></a>对输入数据x、y、z，计算x与y相乘加上z的结果。</p>
</td>
</tr>
<tr id="row273670500"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p113491286273"><a name="p113491286273"></a><a name="p113491286273"></a><a href="../数学函数/bfloat16类型/bfloat16类型算术函数/__hadd-152.md">__hadd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p107361902014"><a name="p107361902014"></a><a name="p107361902014"></a>计算两个bfloat16类型数据的相加结果，并遵循CAST_RINT模式舍入。</p>
</td>
</tr>
<tr id="row10736300018"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p19736602011"><a name="p19736602011"></a><a name="p19736602011"></a><a href="../数学函数/bfloat16类型/bfloat16类型算术函数/__hsub-153.md">__hsub</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p15469163992712"><a name="p15469163992712"></a><a name="p15469163992712"></a>计算两个bfloat16类型数据的相减结果，并遵循CAST_RINT模式舍入。</p>
</td>
</tr>
<tr id="row107361801800"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p4736301012"><a name="p4736301012"></a><a name="p4736301012"></a><a href="../数学函数/bfloat16类型/bfloat16类型算术函数/__hmul-154.md">__hmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p5737208018"><a name="p5737208018"></a><a name="p5737208018"></a>计算两个bfloat16类型数据的相乘结果，并遵循CAST_RINT模式舍入。</p>
</td>
</tr>
<tr id="row107371601501"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p77371707015"><a name="p77371707015"></a><a name="p77371707015"></a><a href="../数学函数/bfloat16类型/bfloat16类型算术函数/__hdiv-155.md">__hdiv</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p3737307019"><a name="p3737307019"></a><a name="p3737307019"></a>计算两个bfloat16类型数据的相除结果，并遵循CAST_RINT模式舍入。</p>
</td>
</tr>
<tr id="row147372003015"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p773716020014"><a name="p773716020014"></a><a name="p773716020014"></a><a href="../数学函数/bfloat16类型/bfloat16类型算术函数/__hneg-156.md">__hneg</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1385714213281"><a name="p1385714213281"></a><a name="p1385714213281"></a>获取输入bfloat16类型数据的负值。</p>
</td>
</tr>
<tr id="row17371501705"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p7737601017"><a name="p7737601017"></a><a name="p7737601017"></a><a href="../数学函数/bfloat16类型/bfloat16类型算术函数/__hfma_relu-157.md">__hfma_relu</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p107371901308"><a name="p107371901308"></a><a name="p107371901308"></a>对输入bfloat16类型数据x、y、z，计算x与y相乘加上z的结果，并遵循CAST_RINT模式舍入。负数结果置为0。</p>
</td>
</tr>
</tbody>
</table>

**表 15**  bfloat16类型比较函数

<a name="table125075351603"></a>
<table><thead align="left"><tr id="row12507935304"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p1150720351907"><a name="p1150720351907"></a><a name="p1150720351907"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p205071935903"><a name="p205071935903"></a><a name="p205071935903"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row132554922119"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p595113106223"><a name="p595113106223"></a><a name="p595113106223"></a><a href="../数学函数/bfloat16类型/bfloat16类型比较函数/__hmax-158.md">__hmax</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p139511810162216"><a name="p139511810162216"></a><a name="p139511810162216"></a>获取两个输入数据中的最大值。</p>
</td>
</tr>
<tr id="row1711514919215"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p395115103228"><a name="p395115103228"></a><a name="p395115103228"></a><a href="../数学函数/bfloat16类型/bfloat16类型比较函数/__hmin-159.md">__hmin</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1895101042219"><a name="p1895101042219"></a><a name="p1895101042219"></a>获取两个输入数据中的最小值。</p>
</td>
</tr>
<tr id="row1050719351609"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p8165730619"><a name="p8165730619"></a><a name="p8165730619"></a><a href="../数学函数/bfloat16类型/bfloat16类型比较函数/__hisnan-160.md">__hisnan</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p12165163013114"><a name="p12165163013114"></a><a name="p12165163013114"></a>判断浮点数是否为nan。</p>
</td>
</tr>
<tr id="row3507153515016"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p01651830917"><a name="p01651830917"></a><a name="p01651830917"></a><a href="../数学函数/bfloat16类型/bfloat16类型比较函数/__hisinf-161.md">__hisinf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p216513306115"><a name="p216513306115"></a><a name="p216513306115"></a>判断浮点数是否为无穷。</p>
</td>
</tr>
<tr id="row450712351400"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p6746546134717"><a name="p6746546134717"></a><a name="p6746546134717"></a><a href="../数学函数/bfloat16类型/bfloat16类型比较函数/__heq-162.md">__heq</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1574654618475"><a name="p1574654618475"></a><a name="p1574654618475"></a>比较两个bfloat16类型数据是否相等，相等时返回true。</p>
</td>
</tr>
<tr id="row1350718351704"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p15746134612477"><a name="p15746134612477"></a><a name="p15746134612477"></a><a href="../数学函数/bfloat16类型/bfloat16类型比较函数/__hne-163.md">__hne</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p574611462476"><a name="p574611462476"></a><a name="p574611462476"></a>比较两个bfloat16类型数据是否不相等，不相等时返回true。</p>
</td>
</tr>
<tr id="row105071835200"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p67461146184716"><a name="p67461146184716"></a><a name="p67461146184716"></a><a href="../数学函数/bfloat16类型/bfloat16类型比较函数/__hle-164.md">__hle</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p674613469472"><a name="p674613469472"></a><a name="p674613469472"></a>比较两个bfloat16类型数据，仅当第一个数小于或等于第二个数时返回true。</p>
</td>
</tr>
<tr id="row205075354020"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p127461146194715"><a name="p127461146194715"></a><a name="p127461146194715"></a><a href="../数学函数/bfloat16类型/bfloat16类型比较函数/__hge-165.md">__hge</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p16746134634714"><a name="p16746134634714"></a><a name="p16746134634714"></a>比较两个bfloat16类型数据，仅当第一个数大于或等于第二个数时返回true。</p>
</td>
</tr>
<tr id="row13508235202"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p13746164634717"><a name="p13746164634717"></a><a name="p13746164634717"></a><a href="../数学函数/bfloat16类型/bfloat16类型比较函数/__hlt-166.md">__hlt</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p174684616479"><a name="p174684616479"></a><a name="p174684616479"></a>比较两个bfloat16类型数据，仅当第一个数小于第二个数时返回true。</p>
</td>
</tr>
<tr id="row35081351012"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p91071355194912"><a name="p91071355194912"></a><a name="p91071355194912"></a><a href="../数学函数/bfloat16类型/bfloat16类型比较函数/__hgt-167.md">__hgt</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p131077556498"><a name="p131077556498"></a><a name="p131077556498"></a>比较两个bfloat16类型数据，仅当第一个数大于第二个数时返回true。</p>
</td>
</tr>
<tr id="row35086351202"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p63377191505"><a name="p63377191505"></a><a name="p63377191505"></a><a href="../数学函数/bfloat16类型/bfloat16类型比较函数/__hequ-168.md">__hequ</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1786614395010"><a name="p1786614395010"></a><a name="p1786614395010"></a>比较两个bfloat16类型数据是否相等，相等时返回true。若任一输入为nan，返回true。</p>
</td>
</tr>
<tr id="row1650817359013"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p17337161919507"><a name="p17337161919507"></a><a name="p17337161919507"></a><a href="../数学函数/bfloat16类型/bfloat16类型比较函数/__hneu-169.md">__hneu</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p412616265014"><a name="p412616265014"></a><a name="p412616265014"></a>比较两个bfloat16类型数据是否不相等，不相等时返回true。若任一输入为nan，返回true。</p>
</td>
</tr>
<tr id="row1450813351603"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p23388197504"><a name="p23388197504"></a><a name="p23388197504"></a><a href="../数学函数/bfloat16类型/bfloat16类型比较函数/__hleu-170.md">__hleu</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p14491120175010"><a name="p14491120175010"></a><a name="p14491120175010"></a>比较两个bfloat16类型数据，当第一个数小于或等于第二个数时返回true。若任一输入为nan，返回true。</p>
</td>
</tr>
<tr id="row2508535407"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p833821918501"><a name="p833821918501"></a><a name="p833821918501"></a><a href="../数学函数/bfloat16类型/bfloat16类型比较函数/__hgeu-171.md">__hgeu</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p586213582493"><a name="p586213582493"></a><a name="p586213582493"></a>比较两个bfloat16类型数据，当第一个数大于或等于第二个数时返回true。若任一输入为nan，返回true。</p>
</td>
</tr>
<tr id="row85229549298"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p153383194502"><a name="p153383194502"></a><a name="p153383194502"></a><a href="../数学函数/bfloat16类型/bfloat16类型比较函数/__hltu-172.md">__hltu</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p112191557194917"><a name="p112191557194917"></a><a name="p112191557194917"></a>比较两个bfloat16类型数据，当第一个数小于第二个数时返回true。若任一输入为nan，返回true。</p>
</td>
</tr>
<tr id="row1068165472918"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p171057354502"><a name="p171057354502"></a><a name="p171057354502"></a><a href="../数学函数/bfloat16类型/bfloat16类型比较函数/__hgtu-173.md">__hgtu</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p111051235185011"><a name="p111051235185011"></a><a name="p111051235185011"></a>比较两个bfloat16类型数据，当第一个数大于第二个数时返回true。若任一输入为nan，返回true。</p>
</td>
</tr>
<tr id="row4845754132915"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1184585432911"><a name="p1184585432911"></a><a name="p1184585432911"></a><a href="../数学函数/bfloat16类型/bfloat16类型比较函数/__hmax_nan-174.md">__hmax_nan</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1984575414299"><a name="p1984575414299"></a><a name="p1984575414299"></a>获取两个输入数据中的最大值。任一输入为nan时返回nan。</p>
</td>
</tr>
<tr id="row1535558293"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p11395582914"><a name="p11395582914"></a><a name="p11395582914"></a><a href="../数学函数/bfloat16类型/bfloat16类型比较函数/__hmin_nan-175.md">__hmin_nan</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p17313553293"><a name="p17313553293"></a><a name="p17313553293"></a>获取两个输入数据中的最小值。任一输入为nan时返回nan。</p>
</td>
</tr>
</tbody>
</table>

**表 16**  bfloat16数学库函数

<a name="table1421820931"></a>
<table><thead align="left"><tr id="row1921152016310"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p92118207310"><a name="p92118207310"></a><a name="p92118207310"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p17211201135"><a name="p17211201135"></a><a name="p17211201135"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row17210201632"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1221162014314"><a name="p1221162014314"></a><a name="p1221162014314"></a><a href="../数学函数/bfloat16类型/bfloat16类型数学库函数/htanh-176.md">htanh</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p12211202316"><a name="p12211202316"></a><a name="p12211202316"></a>获取输入数据的三角函数双曲正切值。</p>
</td>
</tr>
<tr id="row321220330"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1821420630"><a name="p1821420630"></a><a name="p1821420630"></a><a href="../数学函数/bfloat16类型/bfloat16类型数学库函数/hexp-177.md">hexp</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p3211206311"><a name="p3211206311"></a><a name="p3211206311"></a>指定输入x，获取e的x次方。</p>
</td>
</tr>
<tr id="row142113201438"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p182122015312"><a name="p182122015312"></a><a name="p182122015312"></a><a href="../数学函数/bfloat16类型/bfloat16类型数学库函数/hexp2-178.md">hexp2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p9217204312"><a name="p9217204312"></a><a name="p9217204312"></a>指定输入x，获取2的x次方。</p>
</td>
</tr>
<tr id="row1421132011312"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12115202033"><a name="p12115202033"></a><a name="p12115202033"></a><a href="../数学函数/bfloat16类型/bfloat16类型数学库函数/hexp10-179.md">hexp10</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p621142011318"><a name="p621142011318"></a><a name="p621142011318"></a>指定输入x，获取10的x次方。</p>
</td>
</tr>
<tr id="row112119201032"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p7221520335"><a name="p7221520335"></a><a name="p7221520335"></a><a href="../数学函数/bfloat16类型/bfloat16类型数学库函数/hlog-180.md">hlog</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p182213201339"><a name="p182213201339"></a><a name="p182213201339"></a>获取以e为底，输入数据的对数。</p>
</td>
</tr>
<tr id="row102262020319"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p122520633"><a name="p122520633"></a><a name="p122520633"></a><a href="../数学函数/bfloat16类型/bfloat16类型数学库函数/hlog2-181.md">hlog2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p18227205311"><a name="p18227205311"></a><a name="p18227205311"></a>获取以2为底，输入数据的对数。</p>
</td>
</tr>
<tr id="row132220201539"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p13221520434"><a name="p13221520434"></a><a name="p13221520434"></a><a href="../数学函数/bfloat16类型/bfloat16类型数学库函数/hlog10-182.md">hlog10</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p92214202039"><a name="p92214202039"></a><a name="p92214202039"></a>获取以10为底，输入数据的对数。</p>
</td>
</tr>
<tr id="row422172011310"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p5227208319"><a name="p5227208319"></a><a name="p5227208319"></a><a href="../数学函数/bfloat16类型/bfloat16类型数学库函数/hcos-183.md">hcos</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p15221720332"><a name="p15221720332"></a><a name="p15221720332"></a>获取输入数据的三角函数余弦值。</p>
</td>
</tr>
<tr id="row10221520434"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p19228206310"><a name="p19228206310"></a><a name="p19228206310"></a><a href="../数学函数/bfloat16类型/bfloat16类型数学库函数/hsin-184.md">hsin</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p322102016312"><a name="p322102016312"></a><a name="p322102016312"></a>获取输入数据的三角函数正弦值。</p>
</td>
</tr>
<tr id="row1722182018318"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p8223204313"><a name="p8223204313"></a><a name="p8223204313"></a><a href="../数学函数/bfloat16类型/bfloat16类型数学库函数/hsqrt-185.md">hsqrt</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p172214201538"><a name="p172214201538"></a><a name="p172214201538"></a>获取输入数据x的平方根。</p>
</td>
</tr>
<tr id="row8228203319"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p142212015318"><a name="p142212015318"></a><a name="p142212015318"></a><a href="../数学函数/bfloat16类型/bfloat16类型数学库函数/hrsqrt-186.md">hrsqrt</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p102222019310"><a name="p102222019310"></a><a name="p102222019310"></a>获取输入数据x的平方根的倒数。</p>
</td>
</tr>
<tr id="row152210203314"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p822152013315"><a name="p822152013315"></a><a name="p822152013315"></a><a href="../数学函数/bfloat16类型/bfloat16类型数学库函数/hrcp-187.md">hrcp</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p2022720132"><a name="p2022720132"></a><a name="p2022720132"></a>获取输入数据x的倒数。</p>
</td>
</tr>
<tr id="row1222205319"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1822162014319"><a name="p1822162014319"></a><a name="p1822162014319"></a><a href="../数学函数/bfloat16类型/bfloat16类型数学库函数/hrint-188.md">hrint</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p11221320739"><a name="p11221320739"></a><a name="p11221320739"></a>获取与输入数据最接近的整数，若存在两个同样接近的整数，则获取其中的偶数。</p>
</td>
</tr>
<tr id="row1022122015310"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p6221420537"><a name="p6221420537"></a><a name="p6221420537"></a><a href="../数学函数/bfloat16类型/bfloat16类型数学库函数/hfloor-189.md">hfloor</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p522142010317"><a name="p522142010317"></a><a name="p522142010317"></a>获取小于或等于输入数据的最大整数值。</p>
</td>
</tr>
<tr id="row4221820631"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p192218201838"><a name="p192218201838"></a><a name="p192218201838"></a><a href="../数学函数/bfloat16类型/bfloat16类型数学库函数/hceil-190.md">hceil</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p22372012315"><a name="p22372012315"></a><a name="p22372012315"></a>获取大于或等于输入数据的最小整数值。</p>
</td>
</tr>
<tr id="row17239201932"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p162382011318"><a name="p162382011318"></a><a name="p162382011318"></a><a href="../数学函数/bfloat16类型/bfloat16类型数学库函数/htrunc-191.md">htrunc</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p12231920838"><a name="p12231920838"></a><a name="p12231920838"></a>获取对输入数据的浮点数截断后的整数。</p>
</td>
</tr>
</tbody>
</table>

**表 17**  bfloat16类型精度转换函数

<a name="table6134181212173"></a>
<table><thead align="left"><tr id="row613481251712"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p13134612111716"><a name="p13134612111716"></a><a name="p13134612111716"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p513411221717"><a name="p513411221717"></a><a name="p513411221717"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row313414129171"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1354664444719"><a name="p1354664444719"></a><a name="p1354664444719"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__float2bfloat16.md">__float2bfloat16</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p19801112612247"><a name="p19801112612247"></a><a name="p19801112612247"></a>获取输入遵循CAST_RINT模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row813416129179"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p75463441475"><a name="p75463441475"></a><a name="p75463441475"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__float2bfloat16_rn.md">__float2bfloat16_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p78011265249"><a name="p78011265249"></a><a name="p78011265249"></a>获取输入遵循CAST_RINT模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row20134121211178"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p7776194355910"><a name="p7776194355910"></a><a name="p7776194355910"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__float2bfloat16_rn_sat.md">__float2bfloat16_rn_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p158011526172414"><a name="p158011526172414"></a><a name="p158011526172414"></a>饱和模式下获取输入遵循CAST_RINT模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row7135111220174"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p16561111818228"><a name="p16561111818228"></a><a name="p16561111818228"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__float22bfloat162_rn_sat.md">__float22bfloat162_rn_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p198041926102418"><a name="p198041926102418"></a><a name="p198041926102418"></a>饱和模式下获取输入的两个分量遵循CAST_RINT模式转换成的bfloat16x2_t类型数据。</p>
</td>
</tr>
<tr id="row14135191214179"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12547194414712"><a name="p12547194414712"></a><a name="p12547194414712"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__float2bfloat16_rz.md">__float2bfloat16_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p580182615248"><a name="p580182615248"></a><a name="p580182615248"></a>获取输入遵循CAST_TRUNC模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row713581214176"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p822754614593"><a name="p822754614593"></a><a name="p822754614593"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__float2bfloat16_rz_sat.md">__float2bfloat16_rz_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p680117269248"><a name="p680117269248"></a><a name="p680117269248"></a>饱和模式下获取输入遵循CAST_TRUNC模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row1613518126172"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1579082010223"><a name="p1579082010223"></a><a name="p1579082010223"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__float22bfloat162_rz.md">__float22bfloat162_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p17805142612247"><a name="p17805142612247"></a><a name="p17805142612247"></a>获取输入的两个分量遵循CAST_TRUNC模式转换成的bfloat16x2_t类型数据。</p>
</td>
</tr>
<tr id="row141351912181713"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p11297183882214"><a name="p11297183882214"></a><a name="p11297183882214"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__float22bfloat162_rz_sat.md">__float22bfloat162_rz_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p18056261247"><a name="p18056261247"></a><a name="p18056261247"></a>饱和模式下获取输入的两个分量遵循CAST_TRUNC模式转换成的bfloat16x2_t类型数据。</p>
</td>
</tr>
<tr id="row313541261720"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p3547154410476"><a name="p3547154410476"></a><a name="p3547154410476"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__float2bfloat16_rd.md">__float2bfloat16_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p198019268243"><a name="p198019268243"></a><a name="p198019268243"></a>获取输入遵循CAST_FLOOR模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row1113519122171"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12965014595"><a name="p12965014595"></a><a name="p12965014595"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__float2bfloat16_rd_sat.md">__float2bfloat16_rd_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1580132672413"><a name="p1580132672413"></a><a name="p1580132672413"></a>饱和模式下获取输入遵循CAST_FLOOR模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row14135171251712"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p19295426142219"><a name="p19295426142219"></a><a name="p19295426142219"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__float22bfloat162_rd.md">__float22bfloat162_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p2080512266245"><a name="p2080512266245"></a><a name="p2080512266245"></a>获取输入的两个分量遵循CAST_FLOOR模式转换成的bfloat16x2_t类型数据。</p>
</td>
</tr>
<tr id="row4135131215178"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p11708735132212"><a name="p11708735132212"></a><a name="p11708735132212"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__float22bfloat162_rd_sat.md">__float22bfloat162_rd_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p480517268241"><a name="p480517268241"></a><a name="p480517268241"></a>饱和模式下获取输入的两个分量遵循CAST_FLOOR模式转换成的bfloat16x2_t类型数据。</p>
</td>
</tr>
<tr id="row655365962114"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1654784404718"><a name="p1654784404718"></a><a name="p1654784404718"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__float2bfloat16_ru.md">__float2bfloat16_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p7801192614244"><a name="p7801192614244"></a><a name="p7801192614244"></a>获取输入遵循CAST_CEIL模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row8825459162119"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p113125510599"><a name="p113125510599"></a><a name="p113125510599"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__float2bfloat16_ru_sat.md">__float2bfloat16_ru_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p178016261248"><a name="p178016261248"></a><a name="p178016261248"></a>饱和模式下获取输入遵循CAST_CEIL模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row1119190192218"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p586003314221"><a name="p586003314221"></a><a name="p586003314221"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__float22bfloat162_ru.md">__float22bfloat162_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1380582612410"><a name="p1380582612410"></a><a name="p1380582612410"></a>获取输入的两个分量遵循CAST_CEIL模式转换成的bfloat16x2_t类型数据。</p>
</td>
</tr>
<tr id="row82334010226"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p25401924122219"><a name="p25401924122219"></a><a name="p25401924122219"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__float22bfloat162_ru_sat.md">__float22bfloat162_ru_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p180512614247"><a name="p180512614247"></a><a name="p180512614247"></a>饱和模式下获取输入的两个分量遵循CAST_CEIL模式转换成的bfloat16x2_t类型数据。</p>
</td>
</tr>
<tr id="row19427100102213"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p5873192717244"><a name="p5873192717244"></a><a name="p5873192717244"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__float2bfloat16_rna.md">__float2bfloat16_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1133721464319"><a name="p1133721464319"></a><a name="p1133721464319"></a>获取输入遵循CAST_ROUND模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row9616140182219"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1870635218599"><a name="p1870635218599"></a><a name="p1870635218599"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__float2bfloat16_rna_sat.md">__float2bfloat16_rna_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1802132614240"><a name="p1802132614240"></a><a name="p1802132614240"></a>饱和模式下获取输入遵循CAST_ROUND模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row1882020202213"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p6627163114229"><a name="p6627163114229"></a><a name="p6627163114229"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__float22bfloat162_rna.md">__float22bfloat162_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1280592682417"><a name="p1280592682417"></a><a name="p1280592682417"></a>获取输入的两个分量遵循CAST_ROUND模式转换成的bfloat16x2_t类型数据。</p>
</td>
</tr>
<tr id="row149017113229"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p5612192272214"><a name="p5612192272214"></a><a name="p5612192272214"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__float22bfloat162_rna_sat.md">__float22bfloat162_rna_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p168051026192417"><a name="p168051026192417"></a><a name="p168051026192417"></a>饱和模式下获取输入的两个分量遵循CAST_ROUND模式转换成的bfloat16x2_t类型数据。</p>
</td>
</tr>
<tr id="row2403144672415"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p164027364476"><a name="p164027364476"></a><a name="p164027364476"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__half2bfloat16_rn.md">__half2bfloat16_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p138069262249"><a name="p138069262249"></a><a name="p138069262249"></a>获取输入遵循CAST_RINT模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row18636164619243"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p240293634717"><a name="p240293634717"></a><a name="p240293634717"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__half2bfloat16_rz.md">__half2bfloat16_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p108066260248"><a name="p108066260248"></a><a name="p108066260248"></a>获取输入遵循CAST_TRUNC模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row083113463247"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1040217367478"><a name="p1040217367478"></a><a name="p1040217367478"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__half2bfloat16_rd.md">__half2bfloat16_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p14806142619244"><a name="p14806142619244"></a><a name="p14806142619244"></a>获取输入遵循CAST_FLOOR模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row102964715244"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p164028361472"><a name="p164028361472"></a><a name="p164028361472"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__half2bfloat16_ru.md">__half2bfloat16_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p480614263247"><a name="p480614263247"></a><a name="p480614263247"></a>获取输入遵循CAST_CEIL模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row8231447122416"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1636015521246"><a name="p1636015521246"></a><a name="p1636015521246"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__half2bfloat16_rna.md">__half2bfloat16_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p6806726102410"><a name="p6806726102410"></a><a name="p6806726102410"></a>获取输入遵循CAST_ROUND模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row04522476243"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1540433674714"><a name="p1540433674714"></a><a name="p1540433674714"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162float.md">__bfloat162float</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p2810162613241"><a name="p2810162613241"></a><a name="p2810162613241"></a>获取输入转换为浮点数的结果。</p>
</td>
</tr>
<tr id="row1393792017262"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p15214947115810"><a name="p15214947115810"></a><a name="p15214947115810"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162bfloat16_rn.md">__bfloat162bfloat16_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1681012612243"><a name="p1681012612243"></a><a name="p1681012612243"></a>获取输入遵循CAST_RINT模式取整后的bfloat16_t类型数据。</p>
</td>
</tr>
<tr id="row1522142132620"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1783044995814"><a name="p1783044995814"></a><a name="p1783044995814"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162bfloat16_rz.md">__bfloat162bfloat16_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p10810192611247"><a name="p10810192611247"></a><a name="p10810192611247"></a>获取输入遵循CAST_TRUNC模式取整后的bfloat16_t类型数据。</p>
</td>
</tr>
<tr id="row25151921112610"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p101655625812"><a name="p101655625812"></a><a name="p101655625812"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162bfloat16_rd.md">__bfloat162bfloat16_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p168101526142413"><a name="p168101526142413"></a><a name="p168101526142413"></a>获取输入遵循CAST_FLOOR模式取整后的bfloat16_t类型数据。</p>
</td>
</tr>
<tr id="row182162172618"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p87954533587"><a name="p87954533587"></a><a name="p87954533587"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162bfloat16_ru.md">__bfloat162bfloat16_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p4810826182416"><a name="p4810826182416"></a><a name="p4810826182416"></a>获取输入遵循CAST_CEIL模式取整后的bfloat16_t类型数据。</p>
</td>
</tr>
<tr id="row111731748192414"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p16921105115581"><a name="p16921105115581"></a><a name="p16921105115581"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162bfloat16_rna.md">__bfloat162bfloat16_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p14810126102415"><a name="p14810126102415"></a><a name="p14810126102415"></a>获取输入遵循CAST_ROUND模式取整后的bfloat16_t类型数据。</p>
</td>
</tr>
<tr id="row13828958122613"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p18404133615473"><a name="p18404133615473"></a><a name="p18404133615473"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162uint_rn.md">__bfloat162uint_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p3810132682410"><a name="p3810132682410"></a><a name="p3810132682410"></a>获取输入遵循CAST_RINT模式转换成的无符号整数。</p>
</td>
</tr>
<tr id="row41363591263"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p8404123612476"><a name="p8404123612476"></a><a name="p8404123612476"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162uint_rz.md">__bfloat162uint_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p118101226182420"><a name="p118101226182420"></a><a name="p118101226182420"></a>获取输入遵循CAST_TRUNC模式转换成的无符号整数。</p>
</td>
</tr>
<tr id="row8496145952610"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p8404936144720"><a name="p8404936144720"></a><a name="p8404936144720"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162uint_rd.md">__bfloat162uint_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p10935434135914"><a name="p10935434135914"></a><a name="p10935434135914"></a>获取输入遵循CAST_FLOOR模式转换成的无符号整数。</p>
</td>
</tr>
<tr id="row18471959142616"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p5404143619476"><a name="p5404143619476"></a><a name="p5404143619476"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162uint_ru.md">__bfloat162uint_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1781002642417"><a name="p1781002642417"></a><a name="p1781002642417"></a>获取输入遵循CAST_CEIL模式转换成的无符号整数。</p>
</td>
</tr>
<tr id="row151721203274"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12649191222519"><a name="p12649191222519"></a><a name="p12649191222519"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162uint_rna.md">__bfloat162uint_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p16811162619242"><a name="p16811162619242"></a><a name="p16811162619242"></a>获取输入遵循CAST_ROUND模式转换成的无符号整数。</p>
</td>
</tr>
<tr id="row173178239272"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1340411363471"><a name="p1340411363471"></a><a name="p1340411363471"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162int_rn.md">__bfloat162int_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p98111026192412"><a name="p98111026192412"></a><a name="p98111026192412"></a>获取输入遵循CAST_RINT模式转换成的有符号整数。</p>
</td>
</tr>
<tr id="row9602122317274"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p440519365470"><a name="p440519365470"></a><a name="p440519365470"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162int_rz.md">__bfloat162int_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p12811102617246"><a name="p12811102617246"></a><a name="p12811102617246"></a>获取输入遵循CAST_TRUNC模式转换成的有符号整数。</p>
</td>
</tr>
<tr id="row1390213239273"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p7405143611478"><a name="p7405143611478"></a><a name="p7405143611478"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162int_rd.md">__bfloat162int_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1481122622419"><a name="p1481122622419"></a><a name="p1481122622419"></a>获取输入遵循CAST_FLOOR模式转换成的有符号整数。</p>
</td>
</tr>
<tr id="row8198224142712"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p440513369470"><a name="p440513369470"></a><a name="p440513369470"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162int_ru.md">__bfloat162int_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p2811172682417"><a name="p2811172682417"></a><a name="p2811172682417"></a>获取输入遵循CAST_CEIL模式转换成的有符号整数。</p>
</td>
</tr>
<tr id="row12569112432720"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p35290154251"><a name="p35290154251"></a><a name="p35290154251"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162int_rna.md">__bfloat162int_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p3811142622412"><a name="p3811142622412"></a><a name="p3811142622412"></a>获取输入遵循CAST_ROUND模式转换成的有符号整数。</p>
</td>
</tr>
<tr id="row2079764111278"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p154051736204717"><a name="p154051736204717"></a><a name="p154051736204717"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162ull_rn.md">__bfloat162ull_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p12811172612248"><a name="p12811172612248"></a><a name="p12811172612248"></a>获取输入遵循CAST_RINT模式转换成的64位无符号整数。</p>
</td>
</tr>
<tr id="row64511428277"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p840514362473"><a name="p840514362473"></a><a name="p840514362473"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162ull_rz.md">__bfloat162ull_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p68111426162413"><a name="p68111426162413"></a><a name="p68111426162413"></a><span>获取输入遵循CAST_TRUNC模式转换成的64位无符号整数。</span></p>
</td>
</tr>
<tr id="row1129514292718"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p6405123644718"><a name="p6405123644718"></a><a name="p6405123644718"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162ull_rd.md">__bfloat162ull_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1681112619243"><a name="p1681112619243"></a><a name="p1681112619243"></a>获取输入遵循CAST_FLOOR模式转换成的64位无符号整数。</p>
</td>
</tr>
<tr id="row1857124211278"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p4405183694712"><a name="p4405183694712"></a><a name="p4405183694712"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162ull_ru.md">__bfloat162ull_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1581142682414"><a name="p1581142682414"></a><a name="p1581142682414"></a>获取输入遵循CAST_CEIL模式转换成的64位无符号整数。</p>
</td>
</tr>
<tr id="row1583624210270"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p33131618152518"><a name="p33131618152518"></a><a name="p33131618152518"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162ull_rna.md">__bfloat162ull_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1781142632416"><a name="p1781142632416"></a><a name="p1781142632416"></a>获取输入遵循CAST_ROUND模式转换成的64位无符号整数。</p>
</td>
</tr>
<tr id="row132721475288"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1240615365476"><a name="p1240615365476"></a><a name="p1240615365476"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162ll_rn.md">__bfloat162ll_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p335471214015"><a name="p335471214015"></a><a name="p335471214015"></a>获取输入遵循CAST_RINT模式转换成的64位有符号整数。</p>
</td>
</tr>
<tr id="row257210762811"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1540653618478"><a name="p1540653618478"></a><a name="p1540653618478"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162ll_rz.md">__bfloat162ll_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p2812202618247"><a name="p2812202618247"></a><a name="p2812202618247"></a>获取输入遵循CAST_TRUNC模式转换成的64位有符号整数。</p>
</td>
</tr>
<tr id="row117422762810"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p0406103664710"><a name="p0406103664710"></a><a name="p0406103664710"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162ll_rd.md">__bfloat162ll_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p19812112615249"><a name="p19812112615249"></a><a name="p19812112615249"></a>获取输入遵循CAST_FLOOR模式转换成的64位有符号整数。</p>
</td>
</tr>
<tr id="row12115862810"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p540653644713"><a name="p540653644713"></a><a name="p540653644713"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162ll_ru.md">__bfloat162ll_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p981222613242"><a name="p981222613242"></a><a name="p981222613242"></a>获取输入遵循CAST_CEIL模式转换成的64位有符号整数。</p>
</td>
</tr>
<tr id="row63210820285"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p884919238256"><a name="p884919238256"></a><a name="p884919238256"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162ll_rna.md">__bfloat162ll_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p2812142613242"><a name="p2812142613242"></a><a name="p2812142613242"></a>获取输入遵循CAST_ROUND模式转换成的64位有符号整数。</p>
</td>
</tr>
<tr id="row13916158112818"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p64071736184710"><a name="p64071736184710"></a><a name="p64071736184710"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__uint2bfloat16_rn.md">__uint2bfloat16_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p2081616264248"><a name="p2081616264248"></a><a name="p2081616264248"></a>获取输入遵循CAST_RINT模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row1214415596284"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p17407123615473"><a name="p17407123615473"></a><a name="p17407123615473"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__uint2bfloat16_rz.md">__uint2bfloat16_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p10816426182412"><a name="p10816426182412"></a><a name="p10816426182412"></a>获取输入遵循CAST_TRUNC模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row341220593288"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1407133612471"><a name="p1407133612471"></a><a name="p1407133612471"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__uint2bfloat16_rd.md">__uint2bfloat16_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p4817326102414"><a name="p4817326102414"></a><a name="p4817326102414"></a>获取输入遵循CAST_FLOOR模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row1971565912281"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1440717367474"><a name="p1440717367474"></a><a name="p1440717367474"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__uint2bfloat16_ru.md">__uint2bfloat16_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p68171326182420"><a name="p68171326182420"></a><a name="p68171326182420"></a>获取输入遵循CAST_CEIL模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row112515014296"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p3185433142516"><a name="p3185433142516"></a><a name="p3185433142516"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__uint2bfloat16_rna.md">__uint2bfloat16_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p158171126132410"><a name="p158171126132410"></a><a name="p158171126132410"></a>获取输入遵循CAST_ROUND模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row7287142615297"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p17408936184711"><a name="p17408936184711"></a><a name="p17408936184711"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__int2bfloat16_rn.md">__int2bfloat16_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p14819526122417"><a name="p14819526122417"></a><a name="p14819526122417"></a>获取输入遵循CAST_RINT模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row17567626152912"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p3409153684712"><a name="p3409153684712"></a><a name="p3409153684712"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__int2bfloat16_rz.md">__int2bfloat16_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1781902642414"><a name="p1781902642414"></a><a name="p1781902642414"></a>获取输入遵循CAST_TRUNC模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row17823132642910"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p204098366478"><a name="p204098366478"></a><a name="p204098366478"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__int2bfloat16_rd.md">__int2bfloat16_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p12819132652420"><a name="p12819132652420"></a><a name="p12819132652420"></a>获取输入遵循CAST_FLOOR模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row985192732918"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1740943604716"><a name="p1740943604716"></a><a name="p1740943604716"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__int2bfloat16_ru.md">__int2bfloat16_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1782018266248"><a name="p1782018266248"></a><a name="p1782018266248"></a>获取输入遵循CAST_CEIL模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row1736382702916"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12864441192512"><a name="p12864441192512"></a><a name="p12864441192512"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__int2bfloat16_rna.md">__int2bfloat16_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p15820122682416"><a name="p15820122682416"></a><a name="p15820122682416"></a>获取输入遵循CAST_ROUND模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row610055842911"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1141013613476"><a name="p1141013613476"></a><a name="p1141013613476"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__ull2bfloat16_rn.md">__ull2bfloat16_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1582115261241"><a name="p1582115261241"></a><a name="p1582115261241"></a>获取输入遵循CAST_RINT模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row7364155818298"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p0410936124718"><a name="p0410936124718"></a><a name="p0410936124718"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__ull2bfloat16_rz.md">__ull2bfloat16_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1282122615246"><a name="p1282122615246"></a><a name="p1282122615246"></a>获取输入遵循CAST_TRUNC模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row2062825842917"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1341003644714"><a name="p1341003644714"></a><a name="p1341003644714"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__ull2bfloat16_rd.md">__ull2bfloat16_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p178211926182415"><a name="p178211926182415"></a><a name="p178211926182415"></a>获取输入遵循CAST_FLOOR模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row1288012582292"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12410113614717"><a name="p12410113614717"></a><a name="p12410113614717"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__ull2bfloat16_ru.md">__ull2bfloat16_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p98221426152417"><a name="p98221426152417"></a><a name="p98221426152417"></a>获取输入遵循CAST_CEIL模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row16156185962917"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p879125210252"><a name="p879125210252"></a><a name="p879125210252"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__ull2bfloat16_rna.md">__ull2bfloat16_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p118220267240"><a name="p118220267240"></a><a name="p118220267240"></a>获取输入遵循CAST_ROUND模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row132782016010"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1241183654713"><a name="p1241183654713"></a><a name="p1241183654713"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__ll2bfloat16_rn.md">__ll2bfloat16_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p12823142611244"><a name="p12823142611244"></a><a name="p12823142611244"></a>获取输入遵循CAST_RINT模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row858614204010"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p2411193619479"><a name="p2411193619479"></a><a name="p2411193619479"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__ll2bfloat16_rz.md">__ll2bfloat16_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p11823162616244"><a name="p11823162616244"></a><a name="p11823162616244"></a>获取输入遵循CAST_TRUNC模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row176145203017"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p17411536174710"><a name="p17411536174710"></a><a name="p17411536174710"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__ll2bfloat16_rd.md">__ll2bfloat16_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p98231526172415"><a name="p98231526172415"></a><a name="p98231526172415"></a>获取输入遵循CAST_FLOOR模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row15164102115012"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1141112367474"><a name="p1141112367474"></a><a name="p1141112367474"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__ll2bfloat16_ru.md">__ll2bfloat16_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p3824102618249"><a name="p3824102618249"></a><a name="p3824102618249"></a>获取输入遵循CAST_CEIL模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row157534211017"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1583731142610"><a name="p1583731142610"></a><a name="p1583731142610"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__ll2bfloat16_rna.md">__ll2bfloat16_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p158241326182420"><a name="p158241326182420"></a><a name="p158241326182420"></a>获取输入遵循CAST_ROUND模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row692912201216"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p352911716104"><a name="p352911716104"></a><a name="p352911716104"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__float2bfloat162_rn.md">__float2bfloat162_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p88241226132411"><a name="p88241226132411"></a><a name="p88241226132411"></a>将float类型数据遵循CAST_RINT模式转换为bfloat16类型并填充到bfloat16x2的前后两部分，返回填充后的bfloat16x2类型数据。</p>
</td>
</tr>
<tr id="row530413214215"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p175681532111011"><a name="p175681532111011"></a><a name="p175681532111011"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__floats2bfloat162_rn.md">__floats2bfloat162_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p482519268246"><a name="p482519268246"></a><a name="p482519268246"></a>将输入的数据x，y遵循CAST_RINT模式分别转换为bfloat16类型并填充到bfloat16x2的前后两部分，返回转换后的bfloat16x2类型数据。</p>
</td>
</tr>
<tr id="row694113211721"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p143988309102"><a name="p143988309102"></a><a name="p143988309102"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__float22bfloat162_rn.md">__float22bfloat162_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p17825192652413"><a name="p17825192652413"></a><a name="p17825192652413"></a>将float2类型数据遵循CAST_RINT模式转换为bfloat16x2类型，返回转换后的bfloat16x2类型数据。</p>
</td>
</tr>
<tr id="row17848226216"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p11383162811100"><a name="p11383162811100"></a><a name="p11383162811100"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat162bfloat162.md">__bfloat162bfloat162</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p14825122620243"><a name="p14825122620243"></a><a name="p14825122620243"></a>将输入的数据的填充为bfloat16x2前后两个分量，返回转换后的bfloat16x2类型数据。</p>
</td>
</tr>
<tr id="row15629102613215"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p18399202613101"><a name="p18399202613101"></a><a name="p18399202613101"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__halves2bfloat162.md">__halves2bfloat162</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p158251126152418"><a name="p158251126152418"></a><a name="p158251126152418"></a>将输入的数据分别填充为bfloat16x2前后两个分量，返回填充后数据。</p>
</td>
</tr>
<tr id="row131441228827"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p153369245109"><a name="p153369245109"></a><a name="p153369245109"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__high2bfloat16.md">__high2bfloat16</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p4825162611245"><a name="p4825162611245"></a><a name="p4825162611245"></a>提取输入bfloat16x2的高16位，并返回。</p>
</td>
</tr>
<tr id="row11572122812210"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p2361922131011"><a name="p2361922131011"></a><a name="p2361922131011"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__high2bfloat162.md">__high2bfloat162</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p682532620247"><a name="p682532620247"></a><a name="p682532620247"></a>将输入数据的高16位填充到bfloat16x2并返回结果。</p>
</td>
</tr>
<tr id="row203911299215"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p107521119121011"><a name="p107521119121011"></a><a name="p107521119121011"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__high2float.md">__high2float</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p6825192682410"><a name="p6825192682410"></a><a name="p6825192682410"></a>将输入数据的高16位转换为float类型并返回结果。</p>
</td>
</tr>
<tr id="row1444614290215"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p5529132119143"><a name="p5529132119143"></a><a name="p5529132119143"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__highs2bfloat162.md">__highs2bfloat162</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1682616268249"><a name="p1682616268249"></a><a name="p1682616268249"></a>分别提取两个bfloat162输入的高16位，并填充到bfloat162中。返回填充后的数据。</p>
</td>
</tr>
<tr id="row1495117293214"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p543316194140"><a name="p543316194140"></a><a name="p543316194140"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__low2bfloat16.md">__low2bfloat16</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p88262026192410"><a name="p88262026192410"></a><a name="p88262026192410"></a>返回输入数据的低16位。</p>
</td>
</tr>
<tr id="row24053111214"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p21522017171418"><a name="p21522017171418"></a><a name="p21522017171418"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__low2bfloat162.md">__low2bfloat162</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1982611267249"><a name="p1982611267249"></a><a name="p1982611267249"></a>将输入数据的低16位填充到bfloat16x2并返回。</p>
</td>
</tr>
<tr id="row2497431126"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p6827161431416"><a name="p6827161431416"></a><a name="p6827161431416"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__low2float.md">__low2float</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p148269268241"><a name="p148269268241"></a><a name="p148269268241"></a>将输入数据的低16位转换为浮点数并返回结果。</p>
</td>
</tr>
<tr id="row55323220218"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p4653111213147"><a name="p4653111213147"></a><a name="p4653111213147"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__lowhigh2highlow.md">__lowhigh2highlow</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1082682682419"><a name="p1082682682419"></a><a name="p1082682682419"></a><span>将输入数据的高低16位进行交换并返回</span>。</p>
</td>
</tr>
<tr id="row458483211211"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p710816273153"><a name="p710816273153"></a><a name="p710816273153"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__lows2bfloat162.md">__lows2bfloat162</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1482622622418"><a name="p1482622622418"></a><a name="p1482622622418"></a>分别提取两个bfloat162输入的低16位，并填充到bfloat162中。返回填充后的数据。</p>
</td>
</tr>
<tr id="row219918331215"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p593132521514"><a name="p593132521514"></a><a name="p593132521514"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__bfloat1622float2.md">__bfloat1622float2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p15826162652413"><a name="p15826162652413"></a><a name="p15826162652413"></a>将bfloat16x2的两个分量分别转换为float，并填充到float2返回。</p>
</td>
</tr>
<tr id="row1612019281309"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1688842514190"><a name="p1688842514190"></a><a name="p1688842514190"></a><a href="../数学函数/bfloat16类型/bfloat16类型精度转换函数/__ushort_as_bfloat16.md">__ushort_as_bfloat16</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p158291326202415"><a name="p158291326202415"></a><a name="p158291326202415"></a><span>将unsigned short int的按位重新解释为bfloat16，即将unsigned short int的数据存储的位按照bfloat16的格式进行读取。</span></p>
</td>
</tr>
</tbody>
</table>

**表 18**  bfloat16x2类型算术函数

<a name="table55213191438"></a>
<table><thead align="left"><tr id="row17526199317"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p3522191434"><a name="p3522191434"></a><a name="p3522191434"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p3522019733"><a name="p3522019733"></a><a name="p3522019733"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row85241912317"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p17523195310"><a name="p17523195310"></a><a name="p17523195310"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型算术函数/__haddx2-192.md">__haddx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p2052919236"><a name="p2052919236"></a><a name="p2052919236"></a>计算两个bfloat16x2_t类型数据各分量的相加结果，并遵循CAST_RINT模式舍入。</p>
</td>
</tr>
<tr id="row1752319136"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p205291915310"><a name="p205291915310"></a><a name="p205291915310"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型算术函数/__hsubx2-193.md">__hsubx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p252019939"><a name="p252019939"></a><a name="p252019939"></a>计算两个bfloat16x2_t类型数据各分量的相减结果，并遵循CAST_RINT模式舍入。</p>
</td>
</tr>
<tr id="row125211915310"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p135211191836"><a name="p135211191836"></a><a name="p135211191836"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型算术函数/__hmulx2-194.md">__hmulx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p205215191139"><a name="p205215191139"></a><a name="p205215191139"></a>计算两个bfloat16x2_t类型数据各分量的相乘结果，并遵循CAST_RINT模式舍入。</p>
</td>
</tr>
<tr id="row1952419336"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p25211191310"><a name="p25211191310"></a><a name="p25211191310"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型算术函数/__hdivx2-195.md">__hdivx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p452519330"><a name="p452519330"></a><a name="p452519330"></a>计算两个bfloat16x2_t类型数据各分量的相除结果，并遵循CAST_RINT模式舍入。</p>
</td>
</tr>
<tr id="row14532195312"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p85312191838"><a name="p85312191838"></a><a name="p85312191838"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型算术函数/__habsx2-196.md">__habsx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p155318191311"><a name="p155318191311"></a><a name="p155318191311"></a>计算输入bfloat16x2_t类型数据各分量的绝对值。</p>
</td>
</tr>
<tr id="row5531619934"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1153171920313"><a name="p1153171920313"></a><a name="p1153171920313"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型算术函数/__hfmax2-197.md">__hfmax2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p05319191134"><a name="p05319191134"></a><a name="p05319191134"></a>计算两个bfloat16x2_t类型数据各分量的乘加的结果（前两个输入相乘后与第三个输入相加），并遵循CAST_RINT模式舍入。</p>
</td>
</tr>
<tr id="row453191918311"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p175391916320"><a name="p175391916320"></a><a name="p175391916320"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型算术函数/__hnegx2-198.md">__hnegx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p165314199310"><a name="p165314199310"></a><a name="p165314199310"></a>获取输入bfloat16x2_t类型数据各分量的负值。</p>
</td>
</tr>
<tr id="row8535193318"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p185310193314"><a name="p185310193314"></a><a name="p185310193314"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型算术函数/__hfmax2_relu-199.md">__hfmax2_relu</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p185361919311"><a name="p185361919311"></a><a name="p185361919311"></a>计算两个bfloat16x2_t类型数据各分量的乘加的结果（前两个输入相乘后与第三个输入相加），并遵循CAST_RINT模式舍入。负数结果置为0。</p>
</td>
</tr>
<tr id="row1253191910318"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p5530191334"><a name="p5530191334"></a><a name="p5530191334"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型算术函数/__hcmadd-200.md">__hcmadd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p19532191737"><a name="p19532191737"></a><a name="p19532191737"></a>将三个bfloat16x2_t输入视为复数（第一个分量为实部，第二个分量为虚部），执行复数乘加运算x*y+z。</p>
</td>
</tr>
</tbody>
</table>

**表 19**  bfloat16x2类型比较函数

<a name="table6881815165412"></a>
<table><thead align="left"><tr id="row1888171535416"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="p48918155546"><a name="p48918155546"></a><a name="p48918155546"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="p589171519545"><a name="p589171519545"></a><a name="p589171519545"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row0891115105410"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p6891154541"><a name="p6891154541"></a><a name="p6891154541"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hbeqx2-201.md">__hbeqx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1889131565418"><a name="p1889131565418"></a><a name="p1889131565418"></a>比较两个bfloat16x2_t类型数据的两个分量是否相等，仅当两个分量均相等时返回true。</p>
</td>
</tr>
<tr id="row889515185416"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p2089161514549"><a name="p2089161514549"></a><a name="p2089161514549"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hbnex2-202.md">__hbnex2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1189131535411"><a name="p1189131535411"></a><a name="p1189131535411"></a>比较两个bfloat16x2_t类型数据的两个分量是否不相等，仅当两个分量均不相等时返回true。</p>
</td>
</tr>
<tr id="row08971595412"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p78914159549"><a name="p78914159549"></a><a name="p78914159549"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hblex2-203.md">__hblex2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p7891515155419"><a name="p7891515155419"></a><a name="p7891515155419"></a>比较两个bfloat16x2_t类型数据的两个分量，仅当两个分量均满足第一个数小于或等于第二个数时返回true。</p>
</td>
</tr>
<tr id="row689215135419"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p208911512549"><a name="p208911512549"></a><a name="p208911512549"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hbgex2-204.md">__hbgex2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p98901545419"><a name="p98901545419"></a><a name="p98901545419"></a>比较两个bfloat16x2_t类型数据的两个分量，仅当两个分量均满足第一个数大于或等于第二个数时返回true。</p>
</td>
</tr>
<tr id="row13898156542"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p9891115125413"><a name="p9891115125413"></a><a name="p9891115125413"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hbltx2-205.md">__hbltx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1789151515414"><a name="p1789151515414"></a><a name="p1789151515414"></a>比较两个bfloat16x2_t类型数据的两个分量，仅当两个分量均满足第一个数小于第二个数时返回true。</p>
</td>
</tr>
<tr id="row78951520542"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p689915155414"><a name="p689915155414"></a><a name="p689915155414"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hbgtx2-206.md">__hbgtx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p08991515546"><a name="p08991515546"></a><a name="p08991515546"></a>比较两个bfloat16x2_t类型数据的两个分量，仅当两个分量均满足第一个数大于第二个数时返回true。</p>
</td>
</tr>
<tr id="row118912158549"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p12891215155411"><a name="p12891215155411"></a><a name="p12891215155411"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hbequx2-207.md">__hbequx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p58910158544"><a name="p58910158544"></a><a name="p58910158544"></a>比较两个bfloat16x2_t类型数据的两个分量是否相等，当两个分量均相等时返回true。若任一输入的分量为nan，该分量的比较结果为true。</p>
</td>
</tr>
<tr id="row7747192015517"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1974722055517"><a name="p1974722055517"></a><a name="p1974722055517"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hbneux2-208.md">__hbneux2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p18427193011467"><a name="p18427193011467"></a><a name="p18427193011467"></a>比较两个bfloat16x2_t类型数据的两个分量是否不相等，当两个分量均不相等时返回true。若任一输入的分量为nan，该分量的比较结果为true。</p>
</td>
</tr>
<tr id="row1921614227555"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p521622225515"><a name="p521622225515"></a><a name="p521622225515"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hbleux2-209.md">__hbleux2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p10544191986"><a name="p10544191986"></a><a name="p10544191986"></a>比较两个bfloat16x2_t类型数据的两个分量，当两个分量均满足第一个数小于或等于第二个数时返回true。若任一输入的分量为nan，该分量的比较结果为true。</p>
</td>
</tr>
<tr id="row0452123912550"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p194521839105514"><a name="p194521839105514"></a><a name="p194521839105514"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hbgeux2-210.md">__hbgeux2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p184528399559"><a name="p184528399559"></a><a name="p184528399559"></a>比较两个bfloat16x2_t类型数据的两个分量，当两个分量均满足第一个数大于或等于第二个数时返回true。若任一输入的分量为nan，该分量的比较结果为true。</p>
</td>
</tr>
<tr id="row6127740155510"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1512717408554"><a name="p1512717408554"></a><a name="p1512717408554"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hbltux2-211.md">__hbltux2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p3349161313814"><a name="p3349161313814"></a><a name="p3349161313814"></a>比较两个bfloat16x2_t类型数据的两个分量，当两个分量均满足第一个数小于第二个数时返回true。若任一输入的分量为nan，该分量的比较结果为true。</p>
</td>
</tr>
<tr id="row12991740165511"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p14299440145513"><a name="p14299440145513"></a><a name="p14299440145513"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hbgtux2-212.md">__hbgtux2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p229974055510"><a name="p229974055510"></a><a name="p229974055510"></a>比较两个bfloat16x2_t类型数据的两个分量，当两个分量均满足第一个数大于第二个数时返回true。若任一输入的分量为nan，该分量的比较结果为true。</p>
</td>
</tr>
<tr id="row20484204065512"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p184841540105516"><a name="p184841540105516"></a><a name="p184841540105516"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__heqx2-213.md">__heqx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p54848402555"><a name="p54848402555"></a><a name="p54848402555"></a>比较两个bfloat16x2_t类型数据的两个分量，如果分量相等，则对应比较结果为1.0，否则为0.0。</p>
</td>
</tr>
<tr id="row662954075518"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p136291640185514"><a name="p136291640185514"></a><a name="p136291640185514"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hnex2-214.md">__hnex2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p106441931683"><a name="p106441931683"></a><a name="p106441931683"></a>比较两个bfloat16x2_t类型数据的两个分量，如果分量不相等，则对应比较结果为1.0，否则为0.0。</p>
</td>
</tr>
<tr id="row979417407558"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p67949407559"><a name="p67949407559"></a><a name="p67949407559"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hlex2-215.md">__hlex2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p15709123618817"><a name="p15709123618817"></a><a name="p15709123618817"></a>比较两个bfloat16x2_t类型数据的两个分量，如果分量满足第一个数小于或等于第二个数，则对应比较结果为1.0，否则为0.0。</p>
</td>
</tr>
<tr id="row1894317402553"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p16944154045514"><a name="p16944154045514"></a><a name="p16944154045514"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hgex2-216.md">__hgex2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p3944154025514"><a name="p3944154025514"></a><a name="p3944154025514"></a>比较两个bfloat16x2_t类型数据的两个分量，如果分量满足第一个数大于或等于第二个数，则对应比较结果为1.0，否则为0.0。</p>
</td>
</tr>
<tr id="row202371141145520"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p2023774145515"><a name="p2023774145515"></a><a name="p2023774145515"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hltx2-217.md">__hltx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p169755213816"><a name="p169755213816"></a><a name="p169755213816"></a>比较两个bfloat16x2_t类型数据的两个分量，如果分量满足第一个数小于第二个数，则对应比较结果为1.0，否则为0.0。</p>
</td>
</tr>
<tr id="row4814223195516"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p198143235559"><a name="p198143235559"></a><a name="p198143235559"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hgtx2-218.md">__hgtx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p20814192345520"><a name="p20814192345520"></a><a name="p20814192345520"></a>比较两个bfloat16x2_t类型数据的两个分量，如果分量满足第一个数大于第二个数，则对应比较结果为1.0，否则为0.0。</p>
</td>
</tr>
<tr id="row1724191818557"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1224201855511"><a name="p1224201855511"></a><a name="p1224201855511"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hequx2-219.md">__hequx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p22491818553"><a name="p22491818553"></a><a name="p22491818553"></a>比较两个bfloat16x2_t类型数据的两个分量，如果分量相等，则对应比较结果为1.0，否则为0.0。若任一输入的分量为nan，该分量的比较结果为1.0。</p>
</td>
</tr>
<tr id="row760135310567"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p760165311566"><a name="p760165311566"></a><a name="p760165311566"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hneux2-220.md">__hneux2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p26055312560"><a name="p26055312560"></a><a name="p26055312560"></a>比较两个bfloat16x2_t类型数据的两个分量，如果分量不相等，则对应比较结果为1.0，否则为0.0。若任一输入的分量为nan，该分量的比较结果为1.0。</p>
</td>
</tr>
<tr id="row15452145315567"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p144521253105616"><a name="p144521253105616"></a><a name="p144521253105616"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hleux2-221.md">__hleux2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p10452145315564"><a name="p10452145315564"></a><a name="p10452145315564"></a>比较两个bfloat16x2_t类型数据的两个分量，如果分量满足第一个数小于或等于第二个数，则对应比较结果为1.0，否则为0.0。若任一输入的分量为nan，该分量的比较结果为1.0。</p>
</td>
</tr>
<tr id="row158005411561"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p680205435611"><a name="p680205435611"></a><a name="p680205435611"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hgeux2-222.md">__hgeux2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p15591235997"><a name="p15591235997"></a><a name="p15591235997"></a>比较两个bfloat16x2_t类型数据的两个分量，如果分量满足第一个数大于或等于第二个数，则对应比较结果为1.0，否则为0.0。若任一输入的分量为nan，该分量的比较结果为1.0。</p>
</td>
</tr>
<tr id="row8268125405616"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p22681154155616"><a name="p22681154155616"></a><a name="p22681154155616"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hltux2-223.md">__hltux2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p727711411898"><a name="p727711411898"></a><a name="p727711411898"></a>比较两个bfloat16x2_t类型数据的两个分量，如果分量满足第一个数小于第二个数，则对应比较结果为1.0，否则为0.0。若任一输入的分量为nan，该分量的比较结果为1.0。</p>
</td>
</tr>
<tr id="row1450318541567"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p050316544567"><a name="p050316544567"></a><a name="p050316544567"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hgtux2-224.md">__hgtux2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p132584463918"><a name="p132584463918"></a><a name="p132584463918"></a>比较两个bfloat16x2_t类型数据的两个分量，如果分量满足第一个数大于第二个数，则对应比较结果为1.0，否则为0.0。若任一输入的分量为nan，该分量的比较结果为1.0。</p>
</td>
</tr>
<tr id="row195961251577"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1259652595717"><a name="p1259652595717"></a><a name="p1259652595717"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__heqx2_mask-225.md">__heqx2_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1359682511572"><a name="p1359682511572"></a><a name="p1359682511572"></a>比较两个bfloat16x2_t类型数据的两个分量，结果以unsigned int形式返回，低16位为第一个分量的掩码结果，高16位为第二个分量的掩码结果。如果分量相等，则对应16位掩码为0xFFFF，否则为0x0。</p>
</td>
</tr>
<tr id="row496462535711"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p159645255575"><a name="p159645255575"></a><a name="p159645255575"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hnex2_mask-226.md">__hnex2_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p596402505714"><a name="p596402505714"></a><a name="p596402505714"></a>比较两个bfloat16x2_t类型数据的两个分量，结果以unsigned int形式返回，低16位为第一个分量的掩码结果，高16位为第二个分量的掩码结果。如果分量不相等，则对应16位掩码为0xFFFF，否则为0x0。</p>
</td>
</tr>
<tr id="row172941626135710"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1329522618575"><a name="p1329522618575"></a><a name="p1329522618575"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hlex2_mask-227.md">__hlex2_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p52957268571"><a name="p52957268571"></a><a name="p52957268571"></a>比较两个bfloat16x2_t类型数据的两个分量，结果以unsigned int形式返回，低16位为第一个分量的掩码结果，高16位为第二个分量的掩码结果。如果分量满足第一个数小于或等于第二个数，则对应16位掩码为0xFFFF，否则为0x0。</p>
</td>
</tr>
<tr id="row118641526145714"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1486514264573"><a name="p1486514264573"></a><a name="p1486514264573"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hgex2_mask-228.md">__hgex2_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1865152610571"><a name="p1865152610571"></a><a name="p1865152610571"></a>比较两个bfloat16x2_t类型数据的两个分量，结果以unsigned int形式返回，低16位为第一个分量的掩码结果，高16位为第二个分量的掩码结果。如果分量满足第一个数大于或等于第二个数，则对应16位掩码为0xFFFF，否则为0x0。</p>
</td>
</tr>
<tr id="row424742718573"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p16247132725712"><a name="p16247132725712"></a><a name="p16247132725712"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hltx2_mask-229.md">__hltx2_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p10836114181016"><a name="p10836114181016"></a><a name="p10836114181016"></a>比较两个bfloat16x2_t类型数据的两个分量，结果以unsigned int形式返回，低16位为第一个分量的掩码结果，高16位为第二个分量的掩码结果。如果分量满足第一个数小于第二个数，则对应16位掩码为0xFFFF，否则为0x0。</p>
</td>
</tr>
<tr id="row1462972725712"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1630192775710"><a name="p1630192775710"></a><a name="p1630192775710"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hgtx2_mask-230.md">__hgtx2_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p82143717470"><a name="p82143717470"></a><a name="p82143717470"></a>比较两个bfloat16x2_t类型数据的两个分量，结果以unsigned int形式返回，低16位为第一个分量的掩码结果，高16位为第二个分量的掩码结果。如果分量满足第一个数大于第二个数，则对应16位掩码为0xFFFF，否则为0x0。</p>
</td>
</tr>
<tr id="row160020549576"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p360035465720"><a name="p360035465720"></a><a name="p360035465720"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hequx2_mask-231.md">__hequx2_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p186002547577"><a name="p186002547577"></a><a name="p186002547577"></a>比较两个bfloat16x2_t类型数据的两个分量，结果以unsigned int形式返回，低16位为第一个分量的掩码结果，高16位为第二个分量的掩码结果。如果分量相等，则对应16位掩码为0xFFFF，否则为0x0。若任一输入的分量为nan，对应16位掩码为0xFFFF。</p>
</td>
</tr>
<tr id="row189601754185711"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p6960155405717"><a name="p6960155405717"></a><a name="p6960155405717"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hneux2_mask-232.md">__hneux2_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p8960125435713"><a name="p8960125435713"></a><a name="p8960125435713"></a>比较两个bfloat16x2_t类型数据的两个分量，结果以unsigned int形式返回，低16位为第一个分量的掩码结果，高16位为第二个分量的掩码结果。如果分量不相等，则对应16位掩码为0xFFFF，否则为0x0。若任一输入的分量为nan，对应16位掩码为0xFFFF。</p>
</td>
</tr>
<tr id="row22451552573"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p9245175517572"><a name="p9245175517572"></a><a name="p9245175517572"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hleux2_mask-233.md">__hleux2_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p824575517573"><a name="p824575517573"></a><a name="p824575517573"></a>比较两个bfloat16x2_t类型数据的两个分量，结果以unsigned int形式返回，低16位为第一个分量的掩码结果，高16位为第二个分量的掩码结果。如果分量满足第一个数小于或等于第二个数，则对应16位掩码为0xFFFF，否则为0x0。若任一输入的分量为nan，对应16位掩码为0xFFFF。</p>
</td>
</tr>
<tr id="row137551955105713"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p9755145535711"><a name="p9755145535711"></a><a name="p9755145535711"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hgeux2_mask-234.md">__hgeux2_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p175685513571"><a name="p175685513571"></a><a name="p175685513571"></a>比较两个bfloat16x2_t类型数据的两个分量，结果以unsigned int形式返回，低16位为第一个分量的掩码结果，高16位为第二个分量的掩码结果。如果分量满足第一个数大于或等于第二个数，则对应16位掩码为0xFFFF，否则为0x0。若任一输入的分量为nan，对应16位掩码为0xFFFF。</p>
</td>
</tr>
<tr id="row1186156165719"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p9862056195713"><a name="p9862056195713"></a><a name="p9862056195713"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hltux2_mask-235.md">__hltux2_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p19861156175714"><a name="p19861156175714"></a><a name="p19861156175714"></a>比较两个bfloat16x2_t类型数据的两个分量，结果以unsigned int形式返回，低16位为第一个分量的掩码结果，高16位为第二个分量的掩码结果。如果分量满足第一个数小于第二个数，则对应16位掩码为0xFFFF，否则为0x0。若任一输入的分量为nan，对应16位掩码为0xFFFF。</p>
</td>
</tr>
<tr id="row15423156165713"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p94231156185713"><a name="p94231156185713"></a><a name="p94231156185713"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hgtux2_mask-236.md">__hgtux2_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p9423125613578"><a name="p9423125613578"></a><a name="p9423125613578"></a>比较两个bfloat16x2_t类型数据的两个分量，结果以unsigned int形式返回，低16位为第一个分量的掩码结果，高16位为第二个分量的掩码结果。如果分量满足第一个数大于第二个数，则对应16位掩码为0xFFFF，否则为0x0。若任一输入的分量为nan，对应16位掩码为0xFFFF。</p>
</td>
</tr>
<tr id="row14429320145819"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p154291820115819"><a name="p154291820115819"></a><a name="p154291820115819"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__isnanx2-237.md">__isnanx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p7462921141117"><a name="p7462921141117"></a><a name="p7462921141117"></a>判断bfloat16x2_t类型数据的两个分量是否为nan。</p>
</td>
</tr>
<tr id="row3618541151911"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p12619041161918"><a name="p12619041161918"></a><a name="p12619041161918"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hmaxx2-238.md">__hmaxx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p961915415191"><a name="p961915415191"></a><a name="p961915415191"></a>获取两个bfloat16x2_t类型数据各分量的最大值。</p>
</td>
</tr>
<tr id="row254594883514"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p954514893517"><a name="p954514893517"></a><a name="p954514893517"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hmaxx2_nan-239.md">__hmaxx2_nan</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p154554853519"><a name="p154554853519"></a><a name="p154554853519"></a>获取两个bfloat16x2_t类型数据各分量的最大值。任一分量为nan时对应结果为nan。</p>
</td>
</tr>
<tr id="row17740194819355"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1574014811353"><a name="p1574014811353"></a><a name="p1574014811353"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hminx2-240.md">__hminx2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p157405481354"><a name="p157405481354"></a><a name="p157405481354"></a>获取两个bfloat16x2_t类型数据各分量的最小值。</p>
</td>
</tr>
<tr id="row12930848183512"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p393017485353"><a name="p393017485353"></a><a name="p393017485353"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型比较函数/__hminx2_nan-241.md">__hminx2_nan</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1693074873511"><a name="p1693074873511"></a><a name="p1693074873511"></a>获取两个bfloat16x2_t类型数据各分量的最小值。任一分量为nan时对应结果为nan。</p>
</td>
</tr>
</tbody>
</table>

**表 20**  bfloat16x2类型数学库函数

<a name="table12598123918200"></a>
<table><thead align="left"><tr id="row125998399205"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p12599639192010"><a name="p12599639192010"></a><a name="p12599639192010"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p195991639172011"><a name="p195991639172011"></a><a name="p195991639172011"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row15991139192013"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1686118365212"><a name="p1686118365212"></a><a name="p1686118365212"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型数学库函数/h2tanh-242.md">h2tanh</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p20861103614216"><a name="p20861103614216"></a><a name="p20861103614216"></a>获取输入数据各元素的三角函数双曲正切值。</p>
</td>
</tr>
<tr id="row1759917392201"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12861336112119"><a name="p12861336112119"></a><a name="p12861336112119"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型数学库函数/h2exp-243.md">h2exp</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1886117364214"><a name="p1886117364214"></a><a name="p1886117364214"></a>指定输入x，对x的各元素，获取e的该元素次方。</p>
</td>
</tr>
<tr id="row65990391200"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p486117369217"><a name="p486117369217"></a><a name="p486117369217"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型数学库函数/h2exp2-244.md">h2exp2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1686193616218"><a name="p1686193616218"></a><a name="p1686193616218"></a>指定输入x，对x的各元素，获取2的该元素次方。</p>
</td>
</tr>
<tr id="row15599143982019"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p11861193662119"><a name="p11861193662119"></a><a name="p11861193662119"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型数学库函数/h2exp10-245.md">h2exp10</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1486183672117"><a name="p1486183672117"></a><a name="p1486183672117"></a>指定输入x，对x的各元素，获取10的该元素次方。</p>
</td>
</tr>
<tr id="row185991839152013"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p16861143682110"><a name="p16861143682110"></a><a name="p16861143682110"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型数学库函数/h2log-246.md">h2log</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p16861636202116"><a name="p16861636202116"></a><a name="p16861636202116"></a>获取以e为底，输入数据各元素的对数。</p>
</td>
</tr>
<tr id="row10599133942018"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p486113365218"><a name="p486113365218"></a><a name="p486113365218"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型数学库函数/h2log2-247.md">h2log2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p12861536202119"><a name="p12861536202119"></a><a name="p12861536202119"></a>获取以2为底，输入数据各元素的对数。</p>
</td>
</tr>
<tr id="row160012391207"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p386115369211"><a name="p386115369211"></a><a name="p386115369211"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型数学库函数/h2log10-248.md">h2log10</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1286143618214"><a name="p1286143618214"></a><a name="p1286143618214"></a>获取以10为底，输入数据各元素的对数。</p>
</td>
</tr>
<tr id="row36001239132011"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1386243632118"><a name="p1386243632118"></a><a name="p1386243632118"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型数学库函数/h2cos-249.md">h2cos</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p4862193610218"><a name="p4862193610218"></a><a name="p4862193610218"></a>获取输入数据各元素的三角函数余弦值。</p>
</td>
</tr>
<tr id="row4600739152017"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1386223612113"><a name="p1386223612113"></a><a name="p1386223612113"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型数学库函数/h2sin-250.md">h2sin</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p158621836102116"><a name="p158621836102116"></a><a name="p158621836102116"></a>获取输入数据各元素的三角函数正弦值。</p>
</td>
</tr>
<tr id="row136002399204"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1286203652117"><a name="p1286203652117"></a><a name="p1286203652117"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型数学库函数/h2sqrt-251.md">h2sqrt</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p16862113692111"><a name="p16862113692111"></a><a name="p16862113692111"></a>获取输入数据x各元素的平方根。</p>
</td>
</tr>
<tr id="row13600739142011"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p178626368219"><a name="p178626368219"></a><a name="p178626368219"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型数学库函数/h2rsqrt-252.md">h2rsqrt</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p3862173652119"><a name="p3862173652119"></a><a name="p3862173652119"></a>获取输入数据x各元素的平方根的倒数。</p>
</td>
</tr>
<tr id="row2509191314336"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p8862336162113"><a name="p8862336162113"></a><a name="p8862336162113"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型数学库函数/h2rcp-253.md">h2rcp</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p450911383313"><a name="p450911383313"></a><a name="p450911383313"></a>获取输入数据x各元素的倒数。</p>
</td>
</tr>
<tr id="row1962141414337"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p20629148335"><a name="p20629148335"></a><a name="p20629148335"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型数学库函数/h2rint-254.md">h2rint</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1262914173319"><a name="p1262914173319"></a><a name="p1262914173319"></a>获取与输入数据各元素最接近的整数，若存在两个同样接近的整数，则获取其中的偶数。</p>
</td>
</tr>
<tr id="row2431131416339"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1143211483315"><a name="p1143211483315"></a><a name="p1143211483315"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型数学库函数/h2floor-255.md">h2floor</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1243211463316"><a name="p1243211463316"></a><a name="p1243211463316"></a>获取小于或等于输入数据各元素的最大整数值。</p>
</td>
</tr>
<tr id="row1962201519339"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p156351563319"><a name="p156351563319"></a><a name="p156351563319"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型数学库函数/h2ceil-256.md">h2ceil</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1463115103316"><a name="p1463115103316"></a><a name="p1463115103316"></a>获取大于或等于输入数据各元素的最小整数值。</p>
</td>
</tr>
<tr id="row5600239182012"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p17862936112116"><a name="p17862936112116"></a><a name="p17862936112116"></a><a href="../数学函数/bfloat16类型/bfloat16x2类型数学库函数/h2trunc-257.md">h2trunc</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1086243612211"><a name="p1086243612211"></a><a name="p1086243612211"></a>获取对输入数据各元素的浮点数截断后的整数。</p>
</td>
</tr>
</tbody>
</table>

**表 21**  float类型数学库函数

<a name="table198466363363"></a>
<table><thead align="left"><tr id="row884653633612"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p5846636143619"><a name="p5846636143619"></a><a name="p5846636143619"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p1784616363364"><a name="p1784616363364"></a><a name="p1784616363364"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1846133612366"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p4909155410334"><a name="p4909155410334"></a><a name="p4909155410334"></a><a href="../数学函数/float类型数学库函数/tanf.md">tanf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p5909165415330"><a name="p5909165415330"></a><a name="p5909165415330"></a>获取输入数据的三角函数正切值。</p>
</td>
</tr>
<tr id="row17846133693612"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1990916547337"><a name="p1990916547337"></a><a name="p1990916547337"></a><a href="../数学函数/float类型数学库函数/tanhf.md">tanhf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p39094543334"><a name="p39094543334"></a><a name="p39094543334"></a>获取输入数据的三角函数双曲正切值。</p>
</td>
</tr>
<tr id="row1484613369361"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1090955416336"><a name="p1090955416336"></a><a name="p1090955416336"></a><a href="../数学函数/float类型数学库函数/tanpif.md">tanpif</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p9910654123318"><a name="p9910654123318"></a><a name="p9910654123318"></a>获取输入数据与π相乘的正切值。</p>
</td>
</tr>
<tr id="row1484643693619"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p191075410331"><a name="p191075410331"></a><a name="p191075410331"></a><a href="../数学函数/float类型数学库函数/atanf.md">atanf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1891012543332"><a name="p1891012543332"></a><a name="p1891012543332"></a>获取输入数据的反正切值。</p>
</td>
</tr>
<tr id="row1384719368361"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p99101654203315"><a name="p99101654203315"></a><a name="p99101654203315"></a><a href="../数学函数/float类型数学库函数/atan2f.md">atan2f</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p3910145418337"><a name="p3910145418337"></a><a name="p3910145418337"></a>获取输入数据y/x的反正切值。</p>
</td>
</tr>
<tr id="row178471636143620"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p16910145418333"><a name="p16910145418333"></a><a name="p16910145418333"></a><a href="../数学函数/float类型数学库函数/atanhf.md">atanhf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1291075483313"><a name="p1291075483313"></a><a name="p1291075483313"></a>获取输入数据的反双曲正切值。</p>
</td>
</tr>
<tr id="row18847143613362"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p39101254103319"><a name="p39101254103319"></a><a name="p39101254103319"></a><a href="../数学函数/float类型数学库函数/expf.md">expf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p15910554123318"><a name="p15910554123318"></a><a name="p15910554123318"></a>指定输入x，获取e的x次方。</p>
</td>
</tr>
<tr id="row984713364364"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p291015410334"><a name="p291015410334"></a><a name="p291015410334"></a><a href="../数学函数/float类型数学库函数/exp2f.md">exp2f</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p6910135412337"><a name="p6910135412337"></a><a name="p6910135412337"></a>指定输入x，获取2的x次方。</p>
</td>
</tr>
<tr id="row118474365369"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p139106548332"><a name="p139106548332"></a><a name="p139106548332"></a><a href="../数学函数/float类型数学库函数/exp10f.md">exp10f</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1910165423315"><a name="p1910165423315"></a><a name="p1910165423315"></a>指定输入x，获取10的x次方。</p>
</td>
</tr>
<tr id="row1084783618361"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p291012547334"><a name="p291012547334"></a><a name="p291012547334"></a><a href="../数学函数/float类型数学库函数/expm1f.md">expm1f</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1191005413310"><a name="p1191005413310"></a><a name="p1191005413310"></a>指定输入x，获取e的x次方减1。</p>
</td>
</tr>
<tr id="row8847736103620"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p991035418339"><a name="p991035418339"></a><a name="p991035418339"></a><a href="../数学函数/float类型数学库函数/logf.md">logf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p11910205418339"><a name="p11910205418339"></a><a name="p11910205418339"></a>获取以e为底，输入数据的对数。</p>
</td>
</tr>
<tr id="row1984718361362"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1291095410333"><a name="p1291095410333"></a><a name="p1291095410333"></a><a href="../数学函数/float类型数学库函数/log2f.md">log2f</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p179101547334"><a name="p179101547334"></a><a name="p179101547334"></a>获取以2为底，输入数据的对数。</p>
</td>
</tr>
<tr id="row659843411413"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1091015411331"><a name="p1091015411331"></a><a name="p1091015411331"></a><a href="../数学函数/float类型数学库函数/log10f.md">log10f</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p17910145463312"><a name="p17910145463312"></a><a name="p17910145463312"></a>获取以10为底，输入数据的对数。</p>
</td>
</tr>
<tr id="row1386643474113"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1911185411339"><a name="p1911185411339"></a><a name="p1911185411339"></a><a href="../数学函数/float类型数学库函数/log1pf.md">log1pf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p291195411335"><a name="p291195411335"></a><a name="p291195411335"></a>获取以e为底，输入数据加1的对数。</p>
</td>
</tr>
<tr id="row14276357417"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1991185473313"><a name="p1991185473313"></a><a name="p1991185473313"></a><a href="../数学函数/float类型数学库函数/logbf.md">logbf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p129111454183318"><a name="p129111454183318"></a><a name="p129111454183318"></a>计算以2为底，输入数据的对数，并对结果向下取整，返回浮点数。</p>
</td>
</tr>
<tr id="row1019343564115"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12911145419333"><a name="p12911145419333"></a><a name="p12911145419333"></a><a href="../数学函数/float类型数学库函数/ilogbf.md">ilogbf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p39111654113314"><a name="p39111654113314"></a><a name="p39111654113314"></a>计算以2为底，输入数据的对数，并对结果向下取整，返回整数。</p>
</td>
</tr>
<tr id="row183651235194113"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p891175410338"><a name="p891175410338"></a><a name="p891175410338"></a><a href="../数学函数/float类型数学库函数/cosf.md">cosf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p791135473319"><a name="p791135473319"></a><a name="p791135473319"></a>获取输入数据的三角函数余弦值。</p>
</td>
</tr>
<tr id="row256713511419"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p10911115463319"><a name="p10911115463319"></a><a name="p10911115463319"></a><a href="../数学函数/float类型数学库函数/coshf.md">coshf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p119111545339"><a name="p119111545339"></a><a name="p119111545339"></a>获取输入数据的双曲余弦值。</p>
</td>
</tr>
<tr id="row77391535134119"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p99113543339"><a name="p99113543339"></a><a name="p99113543339"></a><a href="../数学函数/float类型数学库函数/cospif.md">cospif</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p591115420332"><a name="p591115420332"></a><a name="p591115420332"></a>获取输入数据与π相乘的余弦值。</p>
</td>
</tr>
<tr id="row15922103518415"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p29118548332"><a name="p29118548332"></a><a name="p29118548332"></a><a href="../数学函数/float类型数学库函数/acosf.md">acosf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1391185433312"><a name="p1391185433312"></a><a name="p1391185433312"></a>获取输入数据的反余弦值。</p>
</td>
</tr>
<tr id="row9107193664116"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1091185423313"><a name="p1091185423313"></a><a name="p1091185423313"></a><a href="../数学函数/float类型数学库函数/acoshf.md">acoshf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1391115543333"><a name="p1391115543333"></a><a name="p1391115543333"></a>获取输入数据的双曲反余弦值。</p>
</td>
</tr>
<tr id="row1928715360416"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p991117540339"><a name="p991117540339"></a><a name="p991117540339"></a><a href="../数学函数/float类型数学库函数/sinf.md">sinf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p491145483319"><a name="p491145483319"></a><a name="p491145483319"></a>获取输入数据的三角函数正弦值。</p>
</td>
</tr>
<tr id="row13483203618412"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p39111854153318"><a name="p39111854153318"></a><a name="p39111854153318"></a><a href="../数学函数/float类型数学库函数/sinhf.md">sinhf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p10911854183317"><a name="p10911854183317"></a><a name="p10911854183317"></a>获取输入数据的双曲正弦值。</p>
</td>
</tr>
<tr id="row116770364412"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p59115544339"><a name="p59115544339"></a><a name="p59115544339"></a><a href="../数学函数/float类型数学库函数/sinpif.md">sinpif</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1991275410335"><a name="p1991275410335"></a><a name="p1991275410335"></a>获取输入数据与π相乘的正弦值。</p>
</td>
</tr>
<tr id="row10879136164119"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p169121654143312"><a name="p169121654143312"></a><a name="p169121654143312"></a><a href="../数学函数/float类型数学库函数/asinf.md">asinf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p15912135433312"><a name="p15912135433312"></a><a name="p15912135433312"></a>获取输入数据的反正弦值。</p>
</td>
</tr>
<tr id="row189077422569"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p169121054133310"><a name="p169121054133310"></a><a name="p169121054133310"></a><a href="../数学函数/float类型数学库函数/asinhf.md">asinhf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1091215453318"><a name="p1091215453318"></a><a name="p1091215453318"></a>获取输入数据的双曲反正弦值。</p>
</td>
</tr>
<tr id="row1943174310560"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p169124541336"><a name="p169124541336"></a><a name="p169124541336"></a><a href="../数学函数/float类型数学库函数/sincosf.md">sincosf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p591275412338"><a name="p591275412338"></a><a name="p591275412338"></a>获取输入数据的三角函数正弦值和余弦值。</p>
</td>
</tr>
<tr id="row1216184314561"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p09121354123318"><a name="p09121354123318"></a><a name="p09121354123318"></a><a href="../数学函数/float类型数学库函数/sincospif.md">sincospif</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p14912135416337"><a name="p14912135416337"></a><a name="p14912135416337"></a>获取输入数据与π相乘的三角函数正弦值和余弦值。</p>
</td>
</tr>
<tr id="row141220434567"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p991225416331"><a name="p991225416331"></a><a name="p991225416331"></a><a href="../数学函数/float类型数学库函数/frexpf.md">frexpf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p59121254153318"><a name="p59121254153318"></a><a name="p59121254153318"></a>将x转换为归一化[1/2, 1)的有符号数乘以2的积分幂。</p>
</td>
</tr>
<tr id="row5590104335619"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p491215414331"><a name="p491215414331"></a><a name="p491215414331"></a><a href="../数学函数/float类型数学库函数/ldexpf.md">ldexpf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1291275463317"><a name="p1291275463317"></a><a name="p1291275463317"></a>获取输入x乘以2的exp次幂的结果。</p>
</td>
</tr>
<tr id="row480434395611"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p591275473312"><a name="p591275473312"></a><a name="p591275473312"></a><a href="../数学函数/float类型数学库函数/sqrtf.md">sqrtf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p15912125473316"><a name="p15912125473316"></a><a name="p15912125473316"></a>获取输入数据x的平方根。</p>
</td>
</tr>
<tr id="row199951943175618"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p59121054113311"><a name="p59121054113311"></a><a name="p59121054113311"></a><a href="../数学函数/float类型数学库函数/rsqrtf.md">rsqrtf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p2091210546339"><a name="p2091210546339"></a><a name="p2091210546339"></a>获取输入数据x的平方根的倒数。</p>
</td>
</tr>
<tr id="row1419254485612"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p17912195414335"><a name="p17912195414335"></a><a name="p17912195414335"></a><a href="../数学函数/float类型数学库函数/hypotf.md">hypotf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p69122549339"><a name="p69122549339"></a><a name="p69122549339"></a>获取输入数据x、y的平方和x^2 + y^2的平方根。</p>
</td>
</tr>
<tr id="row238784419565"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p291215414337"><a name="p291215414337"></a><a name="p291215414337"></a><a href="../数学函数/float类型数学库函数/rhypotf.md">rhypotf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p13912175411334"><a name="p13912175411334"></a><a name="p13912175411334"></a>获取输入数据x、y的平方和x^2 + y^2的平方根的倒数。</p>
</td>
</tr>
<tr id="row3595544195618"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p189135546330"><a name="p189135546330"></a><a name="p189135546330"></a><a href="../数学函数/float类型数学库函数/powf.md">powf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p89131548338"><a name="p89131548338"></a><a name="p89131548338"></a>获取输入数据x的y次幂。</p>
</td>
</tr>
<tr id="row8783844195614"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1991316546338"><a name="p1991316546338"></a><a name="p1991316546338"></a><a href="../数学函数/float类型数学库函数/norm3df.md">norm3df</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p6913165483312"><a name="p6913165483312"></a><a name="p6913165483312"></a>获取输入数据a、b、c的平方和a^2 + b^2 + c^2的平方根。</p>
</td>
</tr>
<tr id="row18979444125611"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p149135547337"><a name="p149135547337"></a><a name="p149135547337"></a><a href="../数学函数/float类型数学库函数/rnorm3df.md">rnorm3df</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1591315547332"><a name="p1591315547332"></a><a name="p1591315547332"></a>获取输入数据a、b、c的平方和a^2 + b^2 + c^2的平方根的倒数。</p>
</td>
</tr>
<tr id="row71883451561"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p79133542332"><a name="p79133542332"></a><a name="p79133542332"></a><a href="../数学函数/float类型数学库函数/norm4df.md">norm4df</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p159133545335"><a name="p159133545335"></a><a name="p159133545335"></a>获取输入数据a、b、c、d的平方和a^2 + b^2+ c^2+ d^2的平方根。</p>
</td>
</tr>
<tr id="row3511174514561"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p109137543339"><a name="p109137543339"></a><a name="p109137543339"></a><a href="../数学函数/float类型数学库函数/rnorm4df.md">rnorm4df</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p2091305415337"><a name="p2091305415337"></a><a name="p2091305415337"></a>获取输入数据a、b、c、d的平方和a^2 + b^2 + c^2 + d^2的平方根的倒数。</p>
</td>
</tr>
<tr id="row14714174512566"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p791335412331"><a name="p791335412331"></a><a name="p791335412331"></a><a href="../数学函数/float类型数学库函数/normf.md">normf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p89131754173312"><a name="p89131754173312"></a><a name="p89131754173312"></a>获取输入数据a中前n个元素的平方和a[0]^2 + a[1]^2 +...+ a[n-1]^2的平方根。</p>
</td>
</tr>
<tr id="row109086458561"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p791375413314"><a name="p791375413314"></a><a name="p791375413314"></a><a href="../数学函数/float类型数学库函数/rnormf.md">rnormf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p20913154183315"><a name="p20913154183315"></a><a name="p20913154183315"></a>获取输入数据a中前n个元素的平方和a[0]^2 + a[1]^2 + ...+ a[n-1]^2的平方根的倒数。</p>
</td>
</tr>
<tr id="row13881346155614"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p149136545335"><a name="p149136545335"></a><a name="p149136545335"></a><a href="../数学函数/float类型数学库函数/cbrtf.md">cbrtf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p69134548337"><a name="p69134548337"></a><a name="p69134548337"></a>获取输入数据x的立方根。</p>
</td>
</tr>
<tr id="row1527614619568"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p79141854143319"><a name="p79141854143319"></a><a name="p79141854143319"></a><a href="../数学函数/float类型数学库函数/rcbrtf.md">rcbrtf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p5914195423310"><a name="p5914195423310"></a><a name="p5914195423310"></a>获取输入数据x的立方根的倒数。</p>
</td>
</tr>
<tr id="row1531739145912"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p0914185463320"><a name="p0914185463320"></a><a name="p0914185463320"></a><a href="../数学函数/float类型数学库函数/erff.md">erff</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p791420542339"><a name="p791420542339"></a><a name="p791420542339"></a>获取输入数据的误差函数值。</p>
</td>
</tr>
<tr id="row1799893915914"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12914125411339"><a name="p12914125411339"></a><a name="p12914125411339"></a><a href="../数学函数/float类型数学库函数/erfcf.md">erfcf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1091465419339"><a name="p1091465419339"></a><a name="p1091465419339"></a>获取输入数据的互补误差函数值。</p>
</td>
</tr>
<tr id="row84569406599"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p2091415414331"><a name="p2091415414331"></a><a name="p2091415414331"></a><a href="../数学函数/float类型数学库函数/erfinvf.md">erfinvf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p10914115423319"><a name="p10914115423319"></a><a name="p10914115423319"></a>获取输入数据的逆误差函数值。</p>
</td>
</tr>
<tr id="row5935640165918"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1691419544339"><a name="p1691419544339"></a><a name="p1691419544339"></a><a href="../数学函数/float类型数学库函数/erfcinvf.md">erfcinvf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p9914155411339"><a name="p9914155411339"></a><a name="p9914155411339"></a>获取输入数据的逆互补误差函数值。</p>
</td>
</tr>
<tr id="row1736117413597"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p7914145493311"><a name="p7914145493311"></a><a name="p7914145493311"></a><a href="../数学函数/float类型数学库函数/erfcxf.md">erfcxf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p191413543333"><a name="p191413543333"></a><a name="p191413543333"></a>获取输入数据的缩放互补误差函数值。</p>
</td>
</tr>
<tr id="row587294118591"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1691414548339"><a name="p1691414548339"></a><a name="p1691414548339"></a><a href="../数学函数/float类型数学库函数/tgammaf.md">tgammaf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1091413548331"><a name="p1091413548331"></a><a name="p1091413548331"></a>获取输入数据x的伽马函数值。</p>
</td>
</tr>
<tr id="row1323174275918"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p7915135493311"><a name="p7915135493311"></a><a name="p7915135493311"></a><a href="../数学函数/float类型数学库函数/lgammaf.md">lgammaf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p3915754143313"><a name="p3915754143313"></a><a name="p3915754143313"></a>获取输入数据x伽马值的绝对值并求自然对数。</p>
</td>
</tr>
<tr id="row16816134219595"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1891595412338"><a name="p1891595412338"></a><a name="p1891595412338"></a><a href="../数学函数/float类型数学库函数/cyl_bessel_i0f.md">cyl_bessel_i0f</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p20915054133314"><a name="p20915054133314"></a><a name="p20915054133314"></a>获取输入数据x的0阶常规修正圆柱贝塞尔函数的值。</p>
</td>
</tr>
<tr id="row20341114345918"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p091514544334"><a name="p091514544334"></a><a name="p091514544334"></a><a href="../数学函数/float类型数学库函数/cyl_bessel_i1f.md">cyl_bessel_i1f</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1915145410334"><a name="p1915145410334"></a><a name="p1915145410334"></a>获取输入数据x的1阶常规修正圆柱贝塞尔函数的值。</p>
</td>
</tr>
<tr id="row19728164445916"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p0915125413319"><a name="p0915125413319"></a><a name="p0915125413319"></a><a href="../数学函数/float类型数学库函数/normcdff.md">normcdff</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p119151454103312"><a name="p119151454103312"></a><a name="p119151454103312"></a>获取输入数据x的标准正态分布的累积分布函数值。</p>
</td>
</tr>
<tr id="row1218724510594"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p10301542341"><a name="p10301542341"></a><a name="p10301542341"></a><a href="../数学函数/float类型数学库函数/normcdfinvf.md">normcdfinvf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p15301346347"><a name="p15301346347"></a><a name="p15301346347"></a>获取输入数据x的标准正态累积分布的逆函数</p>
</td>
</tr>
<tr id="row19644114513596"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p159151549334"><a name="p159151549334"></a><a name="p159151549334"></a><a href="../数学函数/float类型数学库函数/j0f.md">j0f</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p891555443311"><a name="p891555443311"></a><a name="p891555443311"></a>获取输入数据x的0阶第一类贝塞尔函数j0的值。</p>
</td>
</tr>
<tr id="row6116546115912"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p19151554153310"><a name="p19151554153310"></a><a name="p19151554153310"></a><a href="../数学函数/float类型数学库函数/j1f.md">j1f</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1891595418336"><a name="p1891595418336"></a><a name="p1891595418336"></a>获取输入数据x的1阶第一类贝塞尔函数j1的值。</p>
</td>
</tr>
<tr id="row1968014695913"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p2091595483312"><a name="p2091595483312"></a><a name="p2091595483312"></a><a href="../数学函数/float类型数学库函数/jnf.md">jnf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p12915454173316"><a name="p12915454173316"></a><a name="p12915454173316"></a>获取输入数据x的n阶第一类贝塞尔函数jn的值。</p>
</td>
</tr>
<tr id="row018919476592"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p10915185414338"><a name="p10915185414338"></a><a name="p10915185414338"></a><a href="../数学函数/float类型数学库函数/y0f.md">y0f</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p991514545332"><a name="p991514545332"></a><a name="p991514545332"></a>获取输入数据x的0阶第二类贝塞尔函数y0的值。</p>
</td>
</tr>
<tr id="row569217478591"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p29151854203311"><a name="p29151854203311"></a><a name="p29151854203311"></a><a href="../数学函数/float类型数学库函数/y1f.md">y1f</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p13915155483317"><a name="p13915155483317"></a><a name="p13915155483317"></a>获取输入数据x的1阶第二类贝塞尔函数y1的值。</p>
</td>
</tr>
<tr id="row7251154855910"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p19915165418337"><a name="p19915165418337"></a><a name="p19915165418337"></a><a href="../数学函数/float类型数学库函数/ynf.md">ynf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p189151754133313"><a name="p189151754133313"></a><a name="p189151754133313"></a>获取输入数据x的n阶第二类贝塞尔函数yn的值。</p>
</td>
</tr>
<tr id="row16839134855919"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p14916175416338"><a name="p14916175416338"></a><a name="p14916175416338"></a><a href="../数学函数/float类型数学库函数/fabsf.md">fabsf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p119161754193319"><a name="p119161754193319"></a><a name="p119161754193319"></a>获取输入数据的绝对值。</p>
</td>
</tr>
<tr id="row368717103"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p991615546339"><a name="p991615546339"></a><a name="p991615546339"></a><a href="../数学函数/float类型数学库函数/fmaf.md">fmaf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p3916105413333"><a name="p3916105413333"></a><a name="p3916105413333"></a>对输入数据x、y、z，计算x与y相乘加上z的结果。</p>
</td>
</tr>
<tr id="row28853715018"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p2916115413339"><a name="p2916115413339"></a><a name="p2916115413339"></a><a href="../数学函数/float类型数学库函数/fmaxf.md">fmaxf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p091616545335"><a name="p091616545335"></a><a name="p091616545335"></a>获取两个输入数据中的最大值。</p>
</td>
</tr>
<tr id="row875589015"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p291645411338"><a name="p291645411338"></a><a name="p291645411338"></a><a href="../数学函数/float类型数学库函数/fminf.md">fminf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p891675463312"><a name="p891675463312"></a><a name="p891675463312"></a>获取两个输入数据中的最小值。</p>
</td>
</tr>
<tr id="row17267178601"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p4916115411332"><a name="p4916115411332"></a><a name="p4916115411332"></a><a href="../数学函数/float类型数学库函数/fdimf.md">fdimf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p13916125433311"><a name="p13916125433311"></a><a name="p13916125433311"></a>获取输入数据的差值，差值小于0时，返回0。</p>
</td>
</tr>
<tr id="row14445981805"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12916195415332"><a name="p12916195415332"></a><a name="p12916195415332"></a><a href="../数学函数/float类型数学库函数/remquof.md">remquof</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1791695411333"><a name="p1791695411333"></a><a name="p1791695411333"></a>获取输入数据x除以y的余数。求余数时，商取最接近x除以y浮点数结果的整数，当x除以y的浮点数结果与左右最接近的整数距离相等时，商取偶数，同时将商赋值给指针变量quo。</p>
</td>
</tr>
<tr id="row3642198604"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p14916254133316"><a name="p14916254133316"></a><a name="p14916254133316"></a><a href="../数学函数/float类型数学库函数/fmodf.md">fmodf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p491615453316"><a name="p491615453316"></a><a name="p491615453316"></a>获取输入数据x除以y的余数。求余数时，商取x除以y浮点数结果的整数部分。</p>
</td>
</tr>
<tr id="row1582088503"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p139161541336"><a name="p139161541336"></a><a name="p139161541336"></a><a href="../数学函数/float类型数学库函数/remainderf.md">remainderf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p5916185418337"><a name="p5916185418337"></a><a name="p5916185418337"></a>获取输入数据x除以y的余数。求余数时，商取最接近x除以y浮点数结果的整数，当x除以y的浮点数结果与左右最接近的整数距离相等时，商取偶数。</p>
</td>
</tr>
<tr id="row638991502"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p39166542332"><a name="p39166542332"></a><a name="p39166542332"></a><a href="../数学函数/float类型数学库函数/copysignf.md">copysignf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p209161154133315"><a name="p209161154133315"></a><a name="p209161154133315"></a>获取由第一个输入x的数值部分和第二个输入y的符号部分拼接得到的浮点数。</p>
</td>
</tr>
<tr id="row422719504"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1091665433317"><a name="p1091665433317"></a><a name="p1091665433317"></a><a href="../数学函数/float类型数学库函数/nearbyintf.md">nearbyIntf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p16917354163318"><a name="p16917354163318"></a><a name="p16917354163318"></a>获取与输入浮点数最接近的整数，输入浮点数与左右整数的距离相等时，返回偶数。</p>
</td>
</tr>
<tr id="row1844212913016"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p191775483318"><a name="p191775483318"></a><a name="p191775483318"></a><a href="../数学函数/float类型数学库函数/nextafterf.md">nextafterf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p18917185418334"><a name="p18917185418334"></a><a name="p18917185418334"></a>如果y大于x，返回比x大的下一个可表示的浮点值，即浮点数二进制最低位加1。</p>
<p id="p8917354133310"><a name="p8917354133310"></a><a name="p8917354133310"></a>如果y小于x，返回比x小的下一个可表示的浮点值，即浮点数二进制最低位减1。</p>
<p id="p1991745433314"><a name="p1991745433314"></a><a name="p1991745433314"></a>如果y等于x，返回x。</p>
</td>
</tr>
<tr id="row1642159208"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p169175549335"><a name="p169175549335"></a><a name="p169175549335"></a><a href="../数学函数/float类型数学库函数/scalbnf.md">scalbnf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p5917135413314"><a name="p5917135413314"></a><a name="p5917135413314"></a>获取输入数据x与2的n次方的乘积。</p>
</td>
</tr>
<tr id="row1784717911014"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p4917125413310"><a name="p4917125413310"></a><a name="p4917125413310"></a><a href="../数学函数/float类型数学库函数/scalblnf.md">scalblnf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p119172054133314"><a name="p119172054133314"></a><a name="p119172054133314"></a>获取输入数据x与2的n次方的乘积。</p>
</td>
</tr>
<tr id="row1139615108013"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p7832443201414"><a name="p7832443201414"></a><a name="p7832443201414"></a><a href="../数学函数/float类型数学库函数/modff.md">modff</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p883214361417"><a name="p883214361417"></a><a name="p883214361417"></a><span>将输入数据分解为小数部分和整数部分</span>。</p>
</td>
</tr>
<tr id="row955171262315"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1216132319516"><a name="p1216132319516"></a><a name="p1216132319516"></a><a href="../数学函数/float类型数学库函数/fdividef.md">fdivdef</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p116152315515"><a name="p116152315515"></a><a name="p116152315515"></a>获取两个输入数据相除的结果。</p>
</td>
</tr>
<tr id="row178710122235"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p582012523817"><a name="p582012523817"></a><a name="p582012523817"></a><a href="../数学函数/float类型数学库函数/signbit.md">signbit</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p108201952186"><a name="p108201952186"></a><a name="p108201952186"></a>获取输入数据的符号位。</p>
</td>
</tr>
<tr id="row16988423112317"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p0435454162019"><a name="p0435454162019"></a><a name="p0435454162019"></a><a href="../数学函数/float类型数学库函数/__saturatef.md">__saturatef</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p12435954122017"><a name="p12435954122017"></a><a name="p12435954122017"></a>将输入数据钳位到[0.0, 1.0]区间。</p>
</td>
</tr>
<tr id="row975811191259"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1683754011412"><a name="p1683754011412"></a><a name="p1683754011412"></a><a href="../数学函数/float类型数学库函数/__fdividef.md">__fdividef</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1271719232246"><a name="p1271719232246"></a><a name="p1271719232246"></a>获取两个输入数据相除的结果。</p>
</td>
</tr>
<tr id="row55427131720"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1245915462484"><a name="p1245915462484"></a><a name="p1245915462484"></a><a href="../数学函数/float类型数学库函数/rintf.md">rintf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p19459124613489"><a name="p19459124613489"></a><a name="p19459124613489"></a>获取与输入数据最接近的整数，若存在两个同样接近的整数，则获取其中的偶数。</p>
</td>
</tr>
<tr id="row77533139215"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p18459204617484"><a name="p18459204617484"></a><a name="p18459204617484"></a><a href="../数学函数/float类型数学库函数/lrintf.md">lrintf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p20459104684810"><a name="p20459104684810"></a><a name="p20459104684810"></a>获取与输入数据最接近的整数，若存在两个同样接近的整数，则获取其中的偶数。</p>
</td>
</tr>
<tr id="row194710136219"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1745974634812"><a name="p1745974634812"></a><a name="p1745974634812"></a><a href="../数学函数/float类型数学库函数/llrintf.md">llrintf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p245954654812"><a name="p245954654812"></a><a name="p245954654812"></a>获取与输入数据最接近的整数，若存在两个同样接近的整数，则获取其中的偶数。</p>
</td>
</tr>
<tr id="row91201814526"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1545934694812"><a name="p1545934694812"></a><a name="p1545934694812"></a><a href="../数学函数/float类型数学库函数/roundf.md">roundf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p445916469481"><a name="p445916469481"></a><a name="p445916469481"></a>获取对输入数据四舍五入后的整数。</p>
</td>
</tr>
<tr id="row192981148217"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1045944624815"><a name="p1045944624815"></a><a name="p1045944624815"></a><a href="../数学函数/float类型数学库函数/lroundf.md">lroundf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p3459046164817"><a name="p3459046164817"></a><a name="p3459046164817"></a>获取对输入数据四舍五入后的整数。</p>
</td>
</tr>
<tr id="row348661413217"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p194590469486"><a name="p194590469486"></a><a name="p194590469486"></a><a href="../数学函数/float类型数学库函数/llroundf.md">llroundf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p445924614485"><a name="p445924614485"></a><a name="p445924614485"></a>获取对输入数据四舍五入后的整数。</p>
</td>
</tr>
<tr id="row267415144217"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p16459246174813"><a name="p16459246174813"></a><a name="p16459246174813"></a><a href="../数学函数/float类型数学库函数/floorf.md">floorf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p445924616480"><a name="p445924616480"></a><a name="p445924616480"></a>获取小于或等于输入数据的最大整数值。</p>
</td>
</tr>
<tr id="row1185371410211"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p42781135185115"><a name="p42781135185115"></a><a name="p42781135185115"></a><a href="../数学函数/float类型数学库函数/ceilf.md">ceilf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p8278103517513"><a name="p8278103517513"></a><a name="p8278103517513"></a>获取大于或等于输入数据的最小整数值。</p>
</td>
</tr>
<tr id="row84491518217"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p427916352511"><a name="p427916352511"></a><a name="p427916352511"></a><a href="../数学函数/float类型数学库函数/truncf.md">truncf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p12279133555111"><a name="p12279133555111"></a><a name="p12279133555111"></a>获取对输入数据的浮点数截断后的整数。</p>
</td>
</tr>
<tr id="row10243161517216"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p439610186253"><a name="p439610186253"></a><a name="p439610186253"></a><a href="../数学函数/float类型数学库函数/isfinite1.md">isfinite</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1396181813253"><a name="p1396181813253"></a><a name="p1396181813253"></a>判断浮点数是否为有限数（非inf、非nan）。</p>
</td>
</tr>
<tr id="row154639151219"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p2396218192518"><a name="p2396218192518"></a><a name="p2396218192518"></a><a href="../数学函数/float类型数学库函数/isnan1.md">isnan</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p2039631814253"><a name="p2039631814253"></a><a name="p2039631814253"></a>判断浮点数是否为nan。</p>
</td>
</tr>
<tr id="row865113151722"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p93961918152514"><a name="p93961918152514"></a><a name="p93961918152514"></a><a href="../数学函数/float类型数学库函数/isinf1.md">isinf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p18215105443017"><a name="p18215105443017"></a><a name="p18215105443017"></a>判断浮点数是否为无穷。</p>
</td>
</tr>
</tbody>
</table>

**表 22**  类型转换函数

<a name="table10619151412419"></a>
<table><thead align="left"><tr id="row862011415415"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p66201114749"><a name="p66201114749"></a><a name="p66201114749"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p1262011145414"><a name="p1262011145414"></a><a name="p1262011145414"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row462012141143"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p174771914103211"><a name="p174771914103211"></a><a name="p174771914103211"></a><a href="../数学函数/数据类型转换/类型转换函数/__float2float_rn.md">__float2float_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1939214217487"><a name="p1939214217487"></a><a name="p1939214217487"></a>获取输入遵循CAST_RINT模式取整后的浮点数。</p>
</td>
</tr>
<tr id="row56206141445"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p9941919163220"><a name="p9941919163220"></a><a name="p9941919163220"></a><a href="../数学函数/数据类型转换/类型转换函数/__float2float_rz.md">__float2float_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p89891442114818"><a name="p89891442114818"></a><a name="p89891442114818"></a>获取输入遵循CAST_TRUNC模式取整后的浮点数。</p>
</td>
</tr>
<tr id="row2062011420411"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p72778226328"><a name="p72778226328"></a><a name="p72778226328"></a><a href="../数学函数/数据类型转换/类型转换函数/__float2float_rd.md">__float2float_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p451244364810"><a name="p451244364810"></a><a name="p451244364810"></a>获取输入遵循CAST_FLOOR模式取整后的浮点数。</p>
</td>
</tr>
<tr id="row15620114249"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p14495924123219"><a name="p14495924123219"></a><a name="p14495924123219"></a><a href="../数学函数/数据类型转换/类型转换函数/__float2float_ru.md">__float2float_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1813744134814"><a name="p1813744134814"></a><a name="p1813744134814"></a>获取输入遵循CAST_CEIL模式取整后的浮点数。</p>
</td>
</tr>
<tr id="row116209148417"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1919162623211"><a name="p1919162623211"></a><a name="p1919162623211"></a><a href="../数学函数/数据类型转换/类型转换函数/__float2float_rna.md">__float2float_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p13800426202411"><a name="p13800426202411"></a><a name="p13800426202411"></a>获取输入遵循CAST_ROUND模式取整后的浮点数。</p>
</td>
</tr>
<tr id="row1362016144416"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p654810448473"><a name="p654810448473"></a><a name="p654810448473"></a><a href="../数学函数/数据类型转换/类型转换函数/__float2uint_rn.md">__float2uint_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p198021926192411"><a name="p198021926192411"></a><a name="p198021926192411"></a>获取输入遵循CAST_RINT模式转换成的无符号整数。</p>
</td>
</tr>
<tr id="row116200141048"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p154815446477"><a name="p154815446477"></a><a name="p154815446477"></a><a href="../数学函数/数据类型转换/类型转换函数/__float2uint_rz.md">__float2uint_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p68025263249"><a name="p68025263249"></a><a name="p68025263249"></a>获取输入遵循CAST_TRUNC模式转换成的无符号整数。</p>
</td>
</tr>
<tr id="row1562015148416"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p74001636134710"><a name="p74001636134710"></a><a name="p74001636134710"></a><a href="../数学函数/数据类型转换/类型转换函数/__float2uint_rd.md">__float2uint_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p4802132632419"><a name="p4802132632419"></a><a name="p4802132632419"></a>获取输入遵循<span>CAST_FLOOR</span>模式转换成的无符号整数。</p>
</td>
</tr>
<tr id="row106211814940"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p14400143684711"><a name="p14400143684711"></a><a name="p14400143684711"></a><a href="../数学函数/数据类型转换/类型转换函数/__float2uint_ru.md">__float2uint_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p38022269247"><a name="p38022269247"></a><a name="p38022269247"></a>获取输入遵循<span>CAST_CEIL</span>模式转换成的无符号整数。</p>
</td>
</tr>
<tr id="row1362120141549"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p4136163242415"><a name="p4136163242415"></a><a name="p4136163242415"></a><a href="../数学函数/数据类型转换/类型转换函数/__float2uint_rna.md">__float2uint_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p0998074441"><a name="p0998074441"></a><a name="p0998074441"></a>获取输入遵循CAST_ROUND模式转换成的无符号整数。</p>
</td>
</tr>
<tr id="row762120141545"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p20400113634718"><a name="p20400113634718"></a><a name="p20400113634718"></a><a href="../数学函数/数据类型转换/类型转换函数/__float2int_rn.md">__float2int_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1880232613245"><a name="p1880232613245"></a><a name="p1880232613245"></a>获取输入遵循CAST_RINT模式转换成的有符号整数。</p>
</td>
</tr>
<tr id="row0346123315132"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p184004360478"><a name="p184004360478"></a><a name="p184004360478"></a><a href="../数学函数/数据类型转换/类型转换函数/__float2int_rz.md">__float2int_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1080212672414"><a name="p1080212672414"></a><a name="p1080212672414"></a>获取输入遵循<span>CAST_TRUNC</span>模式转换成的有符号整数。</p>
</td>
</tr>
<tr id="row4749123391317"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p3400936114712"><a name="p3400936114712"></a><a name="p3400936114712"></a><a href="../数学函数/数据类型转换/类型转换函数/__float2int_rd.md">__float2int_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p380282612246"><a name="p380282612246"></a><a name="p380282612246"></a>获取输入遵循<span>CAST_FLOOR</span>模式转换成的有符号整数。</p>
</td>
</tr>
<tr id="row18266153491316"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p9401136114720"><a name="p9401136114720"></a><a name="p9401136114720"></a><a href="../数学函数/数据类型转换/类型转换函数/__float2int_ru.md">__float2int_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p19802426162412"><a name="p19802426162412"></a><a name="p19802426162412"></a>获取输入遵循<span>CAST_CEIL</span>模式转换成的有符号整数。</p>
</td>
</tr>
<tr id="row16211114441"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p18923143719248"><a name="p18923143719248"></a><a name="p18923143719248"></a><a href="../数学函数/数据类型转换/类型转换函数/__float2int_rna.md">__float2int_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p380292617243"><a name="p380292617243"></a><a name="p380292617243"></a>获取输入遵循CAST_ROUND模式转换成的有符号整数。</p>
</td>
</tr>
<tr id="row09021819145"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p134015361474"><a name="p134015361474"></a><a name="p134015361474"></a><a href="../数学函数/数据类型转换/类型转换函数/__float2ull_rn.md">__float2ull_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1780222672410"><a name="p1780222672410"></a><a name="p1780222672410"></a>获取输入遵循CAST_RINT模式转换成的64位无符号整数。</p>
</td>
</tr>
<tr id="row19408191891412"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p54011336104715"><a name="p54011336104715"></a><a name="p54011336104715"></a><a href="../数学函数/数据类型转换/类型转换函数/__float2ull_rz.md">__float2ull_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p158032026182417"><a name="p158032026182417"></a><a name="p158032026182417"></a>获取输入遵循<span>CAST_TRUNC</span>模式转换成的64位无符号整数。</p>
</td>
</tr>
<tr id="row11714101817148"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p15401163624710"><a name="p15401163624710"></a><a name="p15401163624710"></a><a href="../数学函数/数据类型转换/类型转换函数/__float2ull_rd.md">__float2ull_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p12803162612415"><a name="p12803162612415"></a><a name="p12803162612415"></a>获取输入遵循<span>CAST_FLOOR</span>模式转换成的64位无符号整数。</p>
</td>
</tr>
<tr id="row1135171918142"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p440123604719"><a name="p440123604719"></a><a name="p440123604719"></a><a href="../数学函数/数据类型转换/类型转换函数/__float2ull_ru.md">__float2ull_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p8803122612248"><a name="p8803122612248"></a><a name="p8803122612248"></a>获取输入遵循<span>CAST_CEIL</span>模式转换成的64位无符号整数。</p>
</td>
</tr>
<tr id="row163811019171413"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1616824392417"><a name="p1616824392417"></a><a name="p1616824392417"></a><a href="../数学函数/数据类型转换/类型转换函数/__float2ull_rna.md">__float2ull_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p7803726182412"><a name="p7803726182412"></a><a name="p7803726182412"></a>获取输入遵循<span>CAST_ROUND</span>模式转换成的64位无符号整数。</p>
</td>
</tr>
<tr id="row139885343141"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12401103614471"><a name="p12401103614471"></a><a name="p12401103614471"></a><a href="../数学函数/数据类型转换/类型转换函数/__float2ll_rn.md">__float2ll_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p10803826102410"><a name="p10803826102410"></a><a name="p10803826102410"></a>获取输入遵循CAST_RINT模式转换成的64位有符号整数。</p>
</td>
</tr>
<tr id="row5273535171415"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1940143654717"><a name="p1940143654717"></a><a name="p1940143654717"></a><a href="../数学函数/数据类型转换/类型转换函数/__float2ll_rz.md">__float2ll_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p5803152614249"><a name="p5803152614249"></a><a name="p5803152614249"></a>获取输入遵循<span>CAST_TRUNC</span>模式转换成的64位有符号整数。</p>
</td>
</tr>
<tr id="row10625173510145"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p194014360474"><a name="p194014360474"></a><a name="p194014360474"></a><a href="../数学函数/数据类型转换/类型转换函数/__float2ll_rd.md">__float2ll_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1080316261241"><a name="p1080316261241"></a><a name="p1080316261241"></a>获取输入遵循<span>CAST_FLOOR</span>模式转换成的64位有符号整数。</p>
</td>
</tr>
<tr id="row8926163581411"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p15401133664711"><a name="p15401133664711"></a><a name="p15401133664711"></a><a href="../数学函数/数据类型转换/类型转换函数/__float2ll_ru.md">__float2ll_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1580317266244"><a name="p1580317266244"></a><a name="p1580317266244"></a>获取输入遵循<span>CAST_CEIL</span>模式转换成的64位有符号整数。</p>
</td>
</tr>
<tr id="row19240173691419"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p123711848132412"><a name="p123711848132412"></a><a name="p123711848132412"></a><a href="../数学函数/数据类型转换/类型转换函数/__float2ll_rna.md">__float2ll_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p20803182672415"><a name="p20803182672415"></a><a name="p20803182672415"></a>获取输入遵循<span>CAST_ROUND</span>模式转换成的64位有符号整数。</p>
</td>
</tr>
<tr id="row56411611513"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p16406036194719"><a name="p16406036194719"></a><a name="p16406036194719"></a><a href="../数学函数/数据类型转换/类型转换函数/__uint2float_rn.md">__uint2float_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p18812226162417"><a name="p18812226162417"></a><a name="p18812226162417"></a>获取输入遵循CAST_RINT模式转换成的浮点数。</p>
</td>
</tr>
<tr id="row19908166141516"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p13406153694711"><a name="p13406153694711"></a><a name="p13406153694711"></a><a href="../数学函数/数据类型转换/类型转换函数/__uint2float_rz.md">__uint2float_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p726720511721"><a name="p726720511721"></a><a name="p726720511721"></a>获取输入遵循CAST_TRUNC模式转换成的浮点数。</p>
</td>
</tr>
<tr id="row141595771519"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p114069361472"><a name="p114069361472"></a><a name="p114069361472"></a><a href="../数学函数/数据类型转换/类型转换函数/__uint2float_rd.md">__uint2float_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p0812202613247"><a name="p0812202613247"></a><a name="p0812202613247"></a>获取输入遵循CAST_FLOOR模式转换成的浮点数。</p>
</td>
</tr>
<tr id="row9475973152"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p440615369477"><a name="p440615369477"></a><a name="p440615369477"></a><a href="../数学函数/数据类型转换/类型转换函数/__uint2float_ru.md">__uint2float_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p17812226192417"><a name="p17812226192417"></a><a name="p17812226192417"></a>获取输入遵循CAST_CEIL模式转换成的浮点数。</p>
</td>
</tr>
<tr id="row15823187141516"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1759027152517"><a name="p1759027152517"></a><a name="p1759027152517"></a><a href="../数学函数/数据类型转换/类型转换函数/__uint2float_rna.md">__uint2float_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p178121626152418"><a name="p178121626152418"></a><a name="p178121626152418"></a>获取输入遵循CAST_ROUND模式转换成的浮点数。</p>
</td>
</tr>
<tr id="row2271113118155"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p15407036144715"><a name="p15407036144715"></a><a name="p15407036144715"></a><a href="../数学函数/数据类型转换/类型转换函数/__int2float_rn.md">__int2float_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p3817162613247"><a name="p3817162613247"></a><a name="p3817162613247"></a>获取输入遵循CAST_RINT模式转换成的浮点数。</p>
</td>
</tr>
<tr id="row10569631171513"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1040815363470"><a name="p1040815363470"></a><a name="p1040815363470"></a><a href="../数学函数/数据类型转换/类型转换函数/__int2float_rz.md">__int2float_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p78171826192412"><a name="p78171826192412"></a><a name="p78171826192412"></a>获取输入遵循CAST_TRUNC模式转换成的浮点数。</p>
</td>
</tr>
<tr id="row985417319155"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1540883614474"><a name="p1540883614474"></a><a name="p1540883614474"></a><a href="../数学函数/数据类型转换/类型转换函数/__int2float_rd.md">__int2float_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p5817626192410"><a name="p5817626192410"></a><a name="p5817626192410"></a>获取输入遵循CAST_FLOOR模式转换成的浮点数。</p>
</td>
</tr>
<tr id="row13199232191515"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p15408436184719"><a name="p15408436184719"></a><a name="p15408436184719"></a><a href="../数学函数/数据类型转换/类型转换函数/__int2float_ru.md">__int2float_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p148175262242"><a name="p148175262242"></a><a name="p148175262242"></a>获取输入遵循CAST_CEIL模式转换成的浮点数。</p>
</td>
</tr>
<tr id="row19569163241511"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p384233512512"><a name="p384233512512"></a><a name="p384233512512"></a><a href="../数学函数/数据类型转换/类型转换函数/__int2float_rna.md">__int2float_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p58180265248"><a name="p58180265248"></a><a name="p58180265248"></a>获取输入遵循CAST_ROUND模式转换成的浮点数。</p>
</td>
</tr>
<tr id="row559561919165"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p7409436164710"><a name="p7409436164710"></a><a name="p7409436164710"></a><a href="../数学函数/数据类型转换/类型转换函数/__ull2float_rn.md">__ull2float_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p282018267240"><a name="p282018267240"></a><a name="p282018267240"></a>获取输入遵循CAST_RINT模式转换成的浮点数。</p>
</td>
</tr>
<tr id="row1485111961612"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p6409123694715"><a name="p6409123694715"></a><a name="p6409123694715"></a><a href="../数学函数/数据类型转换/类型转换函数/__ull2float_rz.md">__ull2float_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p482082610249"><a name="p482082610249"></a><a name="p482082610249"></a>获取输入遵循CAST_TRUNC模式转换成的浮点数。</p>
</td>
</tr>
<tr id="row1111914204167"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p840953620478"><a name="p840953620478"></a><a name="p840953620478"></a><a href="../数学函数/数据类型转换/类型转换函数/__ull2float_rd.md">__ull2float_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p198201626182419"><a name="p198201626182419"></a><a name="p198201626182419"></a>获取输入遵循CAST_FLOOR模式转换成的浮点数。</p>
</td>
</tr>
<tr id="row1836902061616"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1440913365478"><a name="p1440913365478"></a><a name="p1440913365478"></a><a href="../数学函数/数据类型转换/类型转换函数/__ull2float_ru.md">__ull2float_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1382092619243"><a name="p1382092619243"></a><a name="p1382092619243"></a>获取输入遵循CAST_CEIL模式转换成的浮点数。</p>
</td>
</tr>
<tr id="row6642132020160"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p433714532516"><a name="p433714532516"></a><a name="p433714532516"></a><a href="../数学函数/数据类型转换/类型转换函数/__ull2float_rna.md">__ull2float_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1582012612412"><a name="p1582012612412"></a><a name="p1582012612412"></a>获取输入遵循CAST_ROUND模式转换成的浮点数。</p>
</td>
</tr>
<tr id="row1546963761611"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p04109367478"><a name="p04109367478"></a><a name="p04109367478"></a><a href="../数学函数/数据类型转换/类型转换函数/__ll2float_rn.md">__ll2float_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p282220261244"><a name="p282220261244"></a><a name="p282220261244"></a>获取输入遵循CAST_RINT模式转换成的浮点数。</p>
</td>
</tr>
<tr id="row27175372167"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p19410183684711"><a name="p19410183684711"></a><a name="p19410183684711"></a><a href="../数学函数/数据类型转换/类型转换函数/__ll2float_rz.md">__ll2float_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p7822122610248"><a name="p7822122610248"></a><a name="p7822122610248"></a>获取输入遵循CAST_TRUNC模式转换成的浮点数。</p>
</td>
</tr>
<tr id="row10985237141615"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p4410336174713"><a name="p4410336174713"></a><a name="p4410336174713"></a><a href="../数学函数/数据类型转换/类型转换函数/__ll2float_rd.md">__ll2float_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p4822142652410"><a name="p4822142652410"></a><a name="p4822142652410"></a>获取输入遵循CAST_FLOOR模式转换成的浮点数。</p>
</td>
</tr>
<tr id="row82856388163"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1841020365478"><a name="p1841020365478"></a><a name="p1841020365478"></a><a href="../数学函数/数据类型转换/类型转换函数/__ll2float_ru.md">__ll2float_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p982210263242"><a name="p982210263242"></a><a name="p982210263242"></a>获取输入遵循CAST_CEIL模式转换成的浮点数。</p>
</td>
</tr>
<tr id="row9607133881612"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p874335510253"><a name="p874335510253"></a><a name="p874335510253"></a><a href="../数学函数/数据类型转换/类型转换函数/__ll2float_rna.md">__ll2float_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p510694441610"><a name="p510694441610"></a><a name="p510694441610"></a>获取输入遵循CAST_ROUND模式转换成的浮点数。</p>
</td>
</tr>
<tr id="row1620338181717"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p184130363474"><a name="p184130363474"></a><a name="p184130363474"></a><a href="../数学函数/数据类型转换/类型转换函数/__int_as_float.md">__int_as_float</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p12828826202417"><a name="p12828826202417"></a><a name="p12828826202417"></a><span>将整数中的位重新解释为浮点数</span>。</p>
</td>
</tr>
<tr id="row1152013818178"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p3413103684714"><a name="p3413103684714"></a><a name="p3413103684714"></a><a href="../数学函数/数据类型转换/类型转换函数/__uint_as_float.md">__uint_as_float</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p482802642416"><a name="p482802642416"></a><a name="p482802642416"></a><span>将无符号整数中的位重新解释为浮点数</span>。</p>
</td>
</tr>
<tr id="row88261986178"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1198816478553"><a name="p1198816478553"></a><a name="p1198816478553"></a><a href="../数学函数/数据类型转换/类型转换函数/__float_as_int.md">__float_as_int</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p58285268248"><a name="p58285268248"></a><a name="p58285268248"></a><span>将浮点数中的位重新解释为有符号整数</span>。</p>
</td>
</tr>
<tr id="row1013416951712"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1413163611474"><a name="p1413163611474"></a><a name="p1413163611474"></a><a href="../数学函数/数据类型转换/类型转换函数/__float_as_uint.md">__float_as_uint</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p4829226162414"><a name="p4829226162414"></a><a name="p4829226162414"></a><span>将浮点数中的位重新解释为无符号整数</span>。</p>
</td>
</tr>
</tbody>
</table>

**表 23**  整型数学库函数

<a name="table1268552151814"></a>
<table><thead align="left"><tr id="row2068512241812"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p66851521183"><a name="p66851521183"></a><a name="p66851521183"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p126858210182"><a name="p126858210182"></a><a name="p126858210182"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1568517221812"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p719313201519"><a name="p719313201519"></a><a name="p719313201519"></a><a href="../数学函数/整型数学库函数/labs.md">labs</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1419316201258"><a name="p1419316201258"></a><a name="p1419316201258"></a>获取输入数据的绝对值。</p>
</td>
</tr>
<tr id="row8686624189"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1377314471756"><a name="p1377314471756"></a><a name="p1377314471756"></a><a href="../数学函数/整型数学库函数/llabs.md">llabs</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p157737471356"><a name="p157737471356"></a><a name="p157737471356"></a>获取输入数据的绝对值。</p>
</td>
</tr>
<tr id="row15686192101818"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p107292521754"><a name="p107292521754"></a><a name="p107292521754"></a><a href="../数学函数/整型数学库函数/llmax.md">llmax</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p14729752759"><a name="p14729752759"></a><a name="p14729752759"></a>获取两个输入数据中的最大值。</p>
</td>
</tr>
<tr id="row46861728188"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p103167505515"><a name="p103167505515"></a><a name="p103167505515"></a><a href="../数学函数/整型数学库函数/ullmax.md">ullmax</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p183163501555"><a name="p183163501555"></a><a name="p183163501555"></a>获取两个输入数据中的最大值。</p>
</td>
</tr>
<tr id="row76867201812"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p5648163513514"><a name="p5648163513514"></a><a name="p5648163513514"></a><a href="../数学函数/整型数学库函数/umax.md">umax</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p76484350516"><a name="p76484350516"></a><a name="p76484350516"></a>获取两个输入数据中的最大值。</p>
</td>
</tr>
<tr id="row116861231814"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p885293212512"><a name="p885293212512"></a><a name="p885293212512"></a><a href="../数学函数/整型数学库函数/llmin.md">llmin</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p28521327516"><a name="p28521327516"></a><a name="p28521327516"></a>获取两个输入数据中的最小值。</p>
</td>
</tr>
<tr id="row06861821186"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1320313308515"><a name="p1320313308515"></a><a name="p1320313308515"></a><a href="../数学函数/整型数学库函数/ullmin.md">ullmin</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p720343016512"><a name="p720343016512"></a><a name="p720343016512"></a>获取两个输入数据中的最小值。</p>
</td>
</tr>
<tr id="row1468672121817"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p192747271952"><a name="p192747271952"></a><a name="p192747271952"></a><a href="../数学函数/整型数学库函数/umin.md">umin</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p4274727352"><a name="p4274727352"></a><a name="p4274727352"></a>获取两个输入数据中的最小值。</p>
</td>
</tr>
<tr id="row186863281815"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12411535171917"><a name="p12411535171917"></a><a name="p12411535171917"></a><a href="../数学函数/整型数学库函数/__mulhi.md">__mulhi</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p54120358192"><a name="p54120358192"></a><a name="p54120358192"></a>获取输入int32类型数据x和y乘积的高32位。</p>
</td>
</tr>
<tr id="row186867219183"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p2158237101916"><a name="p2158237101916"></a><a name="p2158237101916"></a><a href="../数学函数/整型数学库函数/__umulhi.md">__umulhi</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p215819376197"><a name="p215819376197"></a><a name="p215819376197"></a>获取输入uint32类型数据x和y乘积的高32位。</p>
</td>
</tr>
<tr id="row7836221151912"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1585174131916"><a name="p1585174131916"></a><a name="p1585174131916"></a><a href="../数学函数/整型数学库函数/__mul64hi.md">__mul64hi</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p118511341111914"><a name="p118511341111914"></a><a name="p118511341111914"></a>获取输入int64类型数据x和y乘积的高64位。</p>
</td>
</tr>
<tr id="row152807224190"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p32401643181911"><a name="p32401643181911"></a><a name="p32401643181911"></a><a href="../数学函数/整型数学库函数/__umul64hi.md">__umul64hi</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p12240144319192"><a name="p12240144319192"></a><a name="p12240144319192"></a>获取输入uint64类型数据x和y乘积的高64位。</p>
</td>
</tr>
<tr id="row4726102213194"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p183421349191917"><a name="p183421349191917"></a><a name="p183421349191917"></a><a href="../数学函数/整型数学库函数/__mul_i32toi64.md">__mul_i32toi64</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p434218495199"><a name="p434218495199"></a><a name="p434218495199"></a>计算输入32位整数x和y的乘积，返回64位结果。</p>
</td>
</tr>
<tr id="row17396923151918"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p42744467196"><a name="p42744467196"></a><a name="p42744467196"></a><a href="../数学函数/整型数学库函数/__brev.md">__brev</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p727414651919"><a name="p727414651919"></a><a name="p727414651919"></a>将输入数据的位序反转，返回反转后的值。</p>
</td>
</tr>
<tr id="row47871923191916"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1864720475198"><a name="p1864720475198"></a><a name="p1864720475198"></a><a href="../数学函数/整型数学库函数/__clz.md">__clz</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1564754741916"><a name="p1564754741916"></a><a name="p1564754741916"></a>从输入数据的二进制最高有效位开始，返回连续的前导零的位数。</p>
</td>
</tr>
<tr id="row923652415197"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p20797174413193"><a name="p20797174413193"></a><a name="p20797174413193"></a><a href="../数学函数/整型数学库函数/__ffs.md">__ffs</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p3797164421915"><a name="p3797164421915"></a><a name="p3797164421915"></a>从二进制输入数据的最低位开始，查找第一个值为1的比特位的位置，并返回该位置的索引，索引从1开始计数；如果二进制数据中没有1，则返回0。</p>
</td>
</tr>
<tr id="row17656202461919"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1456104001913"><a name="p1456104001913"></a><a name="p1456104001913"></a><a href="../数学函数/整型数学库函数/__popc.md">__popc</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p124561640151916"><a name="p124561640151916"></a><a name="p124561640151916"></a>统计输入数据从二进制的高位到低位比特位为1的数量。</p>
</td>
</tr>
<tr id="row9157225121911"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p38848382195"><a name="p38848382195"></a><a name="p38848382195"></a><a href="../数学函数/整型数学库函数/__byte_perm.md">__byte_perm</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p178846384196"><a name="p178846384196"></a><a name="p178846384196"></a>由输入的两个4字节的uint32_t类型数据组成一个8个字节的64比特位的整数，通过选择器s指定选取其中的4个字节，将这4个字节从低位到高位拼成一个uint32_t类型的整数。</p>
</td>
</tr>
<tr id="row711915403411"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1611954019411"><a name="p1611954019411"></a><a name="p1611954019411"></a><a href="../数学函数/整型数学库函数/__sad.md">__sad</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p211915406414"><a name="p211915406414"></a><a name="p211915406414"></a>对输入数据x、y、z，计算|x - y|+z的结果，即第一个入参和第二个入参之差的绝对值与第三个入参的和。</p>
</td>
</tr>
<tr id="row9324940134113"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p14324194084115"><a name="p14324194084115"></a><a name="p14324194084115"></a><a href="../数学函数/整型数学库函数/__usad.md">__usad</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1432454012413"><a name="p1432454012413"></a><a name="p1432454012413"></a>对输入数据x、y、z，计算|x - y|+z的结果，即第一个入参和第二个入参之差的绝对值与第三个入参的和。</p>
</td>
</tr>
<tr id="row14487184084114"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1488164012410"><a name="p1488164012410"></a><a name="p1488164012410"></a><a href="../数学函数/整型数学库函数/__mul24.md">__mul24</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p184881240154118"><a name="p184881240154118"></a><a name="p184881240154118"></a>获取输入int32类型数据x和y低24位乘积的低32位结果。x和y的高8位被忽略。</p>
</td>
</tr>
<tr id="row365344084118"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p9653164094119"><a name="p9653164094119"></a><a name="p9653164094119"></a><a href="../数学函数/整型数学库函数/__umul24.md">__umul24</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p8653144011412"><a name="p8653144011412"></a><a name="p8653144011412"></a>获取输入uint32类型数据x和y低24位乘积的低32位结果。x和y的高8位被忽略。</p>
</td>
</tr>
<tr id="row9836240104117"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 ">&nbsp;&nbsp;</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 ">&nbsp;&nbsp;</td>
</tr>
<tr id="row129851340124110"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p498518405417"><a name="p498518405417"></a><a name="p498518405417"></a><a href="../数学函数/整型数学库函数/__hadd-259.md">__hadd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1598584094118"><a name="p1598584094118"></a><a name="p1598584094118"></a>获取输入int32类型数据x和y的平均值，避免中间求和溢出。</p>
</td>
</tr>
<tr id="row9162941144117"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p11162134144115"><a name="p11162134144115"></a><a name="p11162134144115"></a><a href="../数学函数/整型数学库函数/__rhadd.md">__rhadd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1116274112415"><a name="p1116274112415"></a><a name="p1116274112415"></a>获取输入int32类型数据x和y的向上取整平均值，避免中间求和溢出。</p>
</td>
</tr>
<tr id="row1031464116412"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p8314114134116"><a name="p8314114134116"></a><a name="p8314114134116"></a><a href="../数学函数/整型数学库函数/__uhadd.md">__uhadd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p160042953617"><a name="p160042953617"></a><a name="p160042953617"></a>获取输入uint32类型数据x和y的平均值，避免中间求和溢出。</p>
</td>
</tr>
<tr id="row14807411413"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1148010413418"><a name="p1148010413418"></a><a name="p1148010413418"></a><a href="../数学函数/整型数学库函数/__urhadd.md">__urhadd</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p54801841164111"><a name="p54801841164111"></a><a name="p54801841164111"></a>获取输入uint32类型数据x和y的向上取整平均值，避免中间求和溢出。</p>
</td>
</tr>
<tr id="row564317416419"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p13643194117416"><a name="p13643194117416"></a><a name="p13643194117416"></a><a href="../数学函数/整型数学库函数/max1.md">max</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1964364117415"><a name="p1964364117415"></a><a name="p1964364117415"></a>获取两个输入数据中的最大值。</p>
</td>
</tr>
<tr id="row17853114194114"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p285316410416"><a name="p285316410416"></a><a name="p285316410416"></a><a href="../数学函数/整型数学库函数/min1.md">min</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p3844161519416"><a name="p3844161519416"></a><a name="p3844161519416"></a>获取两个输入数据中的最小值。</p>
</td>
</tr>
</tbody>
</table>

## 地址空间谓词函数<a name="section97001946144014"></a>

**表 24**  地址空间谓词函数

<a name="table22661153112118"></a>
<table><thead align="left"><tr id="row1926617533219"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p226710533217"><a name="p226710533217"></a><a name="p226710533217"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p3267105310210"><a name="p3267105310210"></a><a name="p3267105310210"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row126715537214"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1981831015464"><a name="p1981831015464"></a><a name="p1981831015464"></a><a href="../地址空间谓词函数/__isGlobal.md">__isGlobal</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p19267115312118"><a name="p19267115312118"></a><a name="p19267115312118"></a>判断输入的指针是否指向<span id="ph128774221307"><a name="ph128774221307"></a><a name="ph128774221307"></a>Global Memory</span>内存空间的地址。</p>
</td>
</tr>
<tr id="row5267145312110"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p526719530215"><a name="p526719530215"></a><a name="p526719530215"></a><a href="../地址空间谓词函数/__isUbuf.md">__isUbuf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p18267753162115"><a name="p18267753162115"></a><a name="p18267753162115"></a>判断输入的指针是否指向<span id="ph38777222303"><a name="ph38777222303"></a><a name="ph38777222303"></a>Unified Buffer</span>内存空间的地址。</p>
</td>
</tr>
<tr id="row192671453102113"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1267105352111"><a name="p1267105352111"></a><a name="p1267105352111"></a><a href="../地址空间谓词函数/__isLocal.md">__isLocal</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p12993123212442"><a name="p12993123212442"></a><a name="p12993123212442"></a>判断输入的指针是否指向栈空间的地址。</p>
</td>
</tr>
</tbody>
</table>

## 地址空间转换函数<a name="section11840265418"></a>

**表 25**  地址空间转换函数

<a name="table1342852412222"></a>
<table><thead align="left"><tr id="row04281324162215"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p1642952411229"><a name="p1642952411229"></a><a name="p1642952411229"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p542922418224"><a name="p542922418224"></a><a name="p542922418224"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row24293248220"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p5429524102211"><a name="p5429524102211"></a><a name="p5429524102211"></a><a href="../地址空间转换函数/__cvta_generic_to_global.md">__cvta_generic_to_global</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1942992410225"><a name="p1942992410225"></a><a name="p1942992410225"></a>将输入的指针转换为其指向的<span id="ph20205471510"><a name="ph20205471510"></a><a name="ph20205471510"></a>Global Memory</span>内存空间的地址值并返回。</p>
</td>
</tr>
<tr id="row12429132419229"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p13429132432211"><a name="p13429132432211"></a><a name="p13429132432211"></a><a href="../地址空间转换函数/__cvta_generic_to_ubuf.md">__cvta_generic_to_ubuf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p842942492217"><a name="p842942492217"></a><a name="p842942492217"></a>将输入的指针转换为其指向的<span id="ph1973732815477"><a name="ph1973732815477"></a><a name="ph1973732815477"></a>Unified Buffer</span>内存空间的地址值并返回。</p>
</td>
</tr>
<tr id="row16429924132213"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p174291524182215"><a name="p174291524182215"></a><a name="p174291524182215"></a><a href="../地址空间转换函数/__cvta_generic_to_local.md">__cvta_generic_to_local</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p542992442211"><a name="p542992442211"></a><a name="p542992442211"></a>将输入的指针转换为其指向的栈空间地址的值并返回。</p>
</td>
</tr>
<tr id="row096214102236"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p196315103235"><a name="p196315103235"></a><a name="p196315103235"></a><a href="../地址空间转换函数/__cvta_global_to_generic.md">__cvta_global_to_generic</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p38937100557"><a name="p38937100557"></a><a name="p38937100557"></a>将<span id="ph1374624284712"><a name="ph1374624284712"></a><a name="ph1374624284712"></a>Global Memory</span>内存空间的地址值转换为对应的指针。</p>
</td>
</tr>
<tr id="row11714113237"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p81717118235"><a name="p81717118235"></a><a name="p81717118235"></a><a href="../地址空间转换函数/__cvta_ubuf_to_generic.md">__cvta_ubuf_to_generic</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1217151132313"><a name="p1217151132313"></a><a name="p1217151132313"></a>将<span id="ph385565119479"><a name="ph385565119479"></a><a name="ph385565119479"></a>Unified Buffer</span>内存空间的地址值转换为对应的指针。</p>
</td>
</tr>
<tr id="row20472161110237"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p13473121192312"><a name="p13473121192312"></a><a name="p13473121192312"></a><a href="../地址空间转换函数/__cvta_local_to_generic.md">__cvta_local_to_generic</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p54731711192315"><a name="p54731711192315"></a><a name="p54731711192315"></a>将栈空间的地址值转换为对应的指针。</p>
</td>
</tr>
</tbody>
</table>

