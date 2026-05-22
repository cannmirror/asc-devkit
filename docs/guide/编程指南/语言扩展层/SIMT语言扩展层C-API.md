# SIMT语言扩展层C API<a name="ZH-CN_TOPIC_0000002509743873"></a>

SIMT编程基于AI Core的硬件能力实现，可以使用[asc\_vf\_call](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/API/ascendcopapi/atlasascendc_api_07_10303.html)接口启动SIMT VF（Vector Function）子任务。当前，SIMT语言扩展层支持的C API类别如下：

-   [同步与内存栅栏](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/API/ascendcopapi/atlasascendc_api_07_10306.html)：提供内存管理与同步接口，解决不同核内的线程间可能存在的数据竞争以及线程的同步问题。
-   [原子操作](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/API/ascendcopapi/atlasascendc_api_07_10375.html)：提供对Unified Buffer或Global Memory上的数据与指定数据执行原子操作的一系列API接口。
-   [Warp函数](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/API/ascendcopapi/atlasascendc_api_07_10387.html)：提供对单个Warp内32个线程的数据进行处理的相关操作的一系列API接口。
-   [数学函数](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/API/ascendcopapi/atlasascendc_api_07_10630.html)：提供处理数学运算的函数接口集合。
-   [访存函数](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/API/ascendcopapi/atlasascendc_api_07_10556.html)：提供使能Cache Hints的Load/Store函数。
-   [向量类型构造函数](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/API/ascendcopapi/atlasascendc_api_07_10520.html)：向量类型构造相关接口。
-   [调测接口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/API/ascendcopapi/atlasascendc_api_07_10426.html)：SIMT VF调试场景下使用的相关接口。
