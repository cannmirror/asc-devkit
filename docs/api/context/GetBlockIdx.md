# GetBlockIdx<a name="ZH-CN_TOPIC_0000001560710516"></a>

## AI处理器支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>AI处理器类型</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p48327011813"><a name="p48327011813"></a><a name="p48327011813"></a><span id="ph583230201815"><a name="ph583230201815"></a><a name="ph583230201815"></a><term id="zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312391781_term1253731311225"></a>Ascend 910C</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p7948163910184"><a name="p7948163910184"></a><a name="p7948163910184"></a>√</p>
</td>
</tr>
<tr id="row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p14832120181815"><a name="p14832120181815"></a><a name="p14832120181815"></a><span id="ph1483216010188"><a name="ph1483216010188"></a><a name="ph1483216010188"></a><term id="zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312391781_term11962195213215"></a>Ascend 910B</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p19948143911820"><a name="p19948143911820"></a><a name="p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

获取当前核的index，用于代码内部的多核逻辑控制及多核偏移量计算等。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline int64_t GetBlockIdx()
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

当前核的index。

index的范围为\[0, 用户配置的BlockDim数量 - 1\]。

## 约束说明<a name="section633mcpsimp"></a>

GetBlockIdx为一个系统内置函数，返回当前核的index。

## 调用示例<a name="section177231425115410"></a>

```
#include "kernel_operator.h"
constexpr int32_t SINGLE_CORE_OFFSET = 256;
class KernelGetBlockIdx {
public:
    __aicore__ inline KernelGetBlockIdx () {}
    __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
    {
        // 根据index对每个核进行地址偏移
        src0Global.SetGlobalBuffer((__gm__ float*)src0Gm + AscendC::GetBlockIdx() * SINGLE_CORE_OFFSET);
        src1Global.SetGlobalBuffer((__gm__ float*)src1Gm + AscendC::GetBlockIdx() * SINGLE_CORE_OFFSET);
        dstGlobal.SetGlobalBuffer((__gm__ float*)dstGm + AscendC::GetBlockIdx() * SINGLE_CORE_OFFSET);
        pipe.InitBuffer(inQueueSrc0, 1, 256 * sizeof(float));
        pipe.InitBuffer(inQueueSrc1, 1, 256 * sizeof(float));
        pipe.InitBuffer(selMask, 1, 256);
        pipe.InitBuffer(outQueueDst, 1, 256 * sizeof(float));
    }
    ......
};
```

