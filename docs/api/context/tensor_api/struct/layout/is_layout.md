# is\_layout

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

判断输入的数据结构是否为Layout数据结构，可通过检查其成员常量value的值来判断。当value为true时，表示输入的数据结构是Layout类型；反之则为非Layout类型。

## 函数原型

```cpp
template <typename T> struct is_layout
```

## 参数说明

**表 1**  模板参数说明


<table><thead align="left"><tr id="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_row6223476444"><th class="cellrowborder" valign="top" width="20.34%" id="mcps1.2.3.1.1"><p id="p1085176175119">参数名</p>
</th>
<th class="cellrowborder" valign="top" width="79.66%" id="mcps1.2.3.1.2"><p id="p148519610515">描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1392614743211"><td class="cellrowborder" valign="top" width="20.34%" headers="mcps1.2.3.1.1 "><p id="p17843141016336">T</p>
</td>
<td class="cellrowborder" valign="top" width="79.66%" headers="mcps1.2.3.1.2 "><p id="p1192718714328">根据输入的数据类型，判断是否为Layout数据结构。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明

无

## 约束说明

无

## 调用示例

```cpp
using namespace AscendC::Te;

// 初始化Layout数据结构并判断其类型
auto shape = MakeShape(10, 20, 30);
auto stride = MakeStride(1, 100, 200);

auto layoutMake = MakeLayout(shape, stride);
Layout<decltype(shape), decltype(stride)> layoutInit(shape, stride);

bool value = is_layout<decltype(shape)>::value; // value = false
value = is_layout<decltype(stride)>::value; // value = false

value = is_layout<decltype(layoutMake)>::value; // value = true
value = is_layout<decltype(layoutInit)>::value; // value = true
```

