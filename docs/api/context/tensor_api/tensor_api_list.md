# Tensor API

Tensor API文档目录，整体使用时可以引入tensor.h，Tensor API列表如下：

## 数据结构

| 结构名 | 说明 |
| --------- | ---------------- |
| [Shape](./tensor/Shape.md) | 定义Shape |
| [Stride](./tensor/Stride.md) | 定义Stride |
| [Coord](./tensor/Coord.md) | 定义Coord |
| [Layout](./tensor/Layout简介.md) | 定义Layout |
| [Tensor](./tensor/Tensor.md) | 定义Tensor |
| [Pointer](./tensor/Pointer.md) | 定义迭代器 |
| [Pointer](./tensor/Tile.md) | 张量分块 |

## 数据接口

基础结构类API，可以高效地管理内存，进行复杂的张量操作。此类API列表如下：

| API名称 | 说明 |
| ---------- | ----------- |
| [MakeShape] | 构造Shape |
| [MakeStride] | 构造Stride |
| [GetShape](struct/constructor/GetShape.md) | 构造Stride |
| [GetStride](struct/constructor/GetStride.md) | 构造Stride |
| [MakeCoord] | 构造Coord |
| [MakeTile] | 构造Tile |
| [MakeLayout](./tensor/layout_fractal//MakeLayout.md) | 构造Layoute |
| [MakeNZLayout](./tensor/layout_fractal/MakeNZLayout.md) | 构造NZ Layoute |
| [MakeZNLayout](./tensor/layout_fractal/MakeZNLayout.md) | 构造Layoute |
| [is_Layout](./tensor/layout_fractal/is_layout.md) | 判断Layoute |
| [Get] | 构造子Layout |
| [Select](./tensor/Select.md) | 构造子Layout |
| [GetShape](./tensor/GetShape.md) | 返回Shape |
| [GetLayout](./tensor/GetLayout.md)  | 返回Layout对象 |
| [GetStride](./tensor/GetStride.md)  | 返回Stride |
| [Size] | 返回Shape的总大小 |
| [Rank](./tensor/Rank.md) | 返回Layout中的秩 |
| [Coshape] | 返回实际Shape空间 |
| [Cosize] | 返回实际占用的内存 |
| [Crd2Idx](./tensor/Crd2Idx.md) | Coordinate转Index |
| [MakeTensor] | 构造Tensor |
| [ZippedDivided] | 通过Tile切分Tensor，返回Tensor |
| [InnerPartition] | 通过Tile切分Tensor，返回Tensor |
| [LocalTile] | 通过Tile切分Tensor，返回Tensor |
| [MakeGlobalPtr] | 构造GlobalPointer迭代器 |
| [RecastlPtr] | Ptr强制转换 |
| [MakeL1Ptr] | 构造L1Pointer迭代器 |
| [MakeL0APtr] | 构造L0APointer迭代器 |
| [MakeL0BPtr] | 构造L0BPointer迭代器 |
| [MakeL0CPtr] | 构造L0CPointer迭代器 |

## 数据搬运

数据搬运类API，单独使用时可以引入tensor_tile_arch.h，此类API列表如下：

|   API名称   |   说明   |
|----------|-----------|
| [DataCopy](./arch/cube_datamove/DataCopy.md) | CUBE计算，支持GM2L1搬运处理 |
| [TileCopy] | CUBE计算，支持L12L0搬运 |
| [FixPipe](./arch/cube_datamove/Fixpipe.md) | FixPIPE搬出接口 |
| [LoadData](./arch/cube_datamove/LoadData.md) | L12L0加载接口 |

## 矩阵计算

标量操作类API，单独使用时可以引入tensor_tile_arch.h，此类API列表如下：

|   API名称   |   说明   |
|----------|-----------|
| [Mmad](./arch/cube_compute/Mad.md) | 完成矩阵乘加操作。 |


