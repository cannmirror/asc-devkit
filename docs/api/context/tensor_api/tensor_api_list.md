# Tensor API

Tensor API文档目录，整体使用时可以引入tensor.h，Tensor API列表如下：

## 数据结构

| 结构名 | 说明 |
| --------- | ---------------- |
| [Shape](struct/Shape.md) | 定义Shape |
| [Stride](struct/Stride.md) | 定义Stride |
| [Coord](struct/coord/Coord.md) | 定义Coord |
| [Layout](struct/layout/Layout简介.md) | 定义Layout |
| [LocalTensor](struct/tensor/LocalTensor.md) | 定义LocalTensor |
| [Pointer](struct/pointer/Pointer.md) | 定义迭代器 |
| [Tile](struct/tile/Tile.md) | 张量分块 |

## 数据接口

基础结构类API，可以高效地管理内存，进行复杂的张量操作。此类API列表如下：

| API名称 | 说明 |
| ---------- | ----------- |
| [MakeShape] | 构造Shape |
| [MakeStride] | 构造Stride |
| [GetShape](struct/layout/GetShape.md) | 获取Shape |
| [GetStride](struct/layout/GetStride.md) | 获取Stride |
| [MakeCoord] | 构造Coord |
| [MakeTile] | 构造Tile |
| [MakeLayout](struct/layout/layout_fractal/MakeLayout.md) | 构造Layout |
| [MakeNZLayout](struct/layout/layout_fractal/MakeNZLayout.md) | 构造NZ Layout |
| [MakeZNLayout](struct/layout/layout_fractal/MakeZNLayout.md) | 构造ZN Layout |
| [is_Layout](struct/layout/is_layout.md) | 判断Layout |
| [Get] | 构造子Layout |
| [Select](struct/layout/Select.md) | 构造子Layout |
| [GetShape](struct/layout/GetShape.md) | 返回Shape |
| [GetLayout](struct/tensor/GetLayout.md)  | 返回Layout对象 |
| [GetStride](struct/layout/GetStride.md)  | 返回Stride |
| [Size] | 返回Shape的总大小 |
| [Rank](struct/layout/Rank.md) | 返回Layout中的秩 |
| [Coshape] | 返回实际Shape空间 |
| [Cosize] | 返回实际占用的内存 |
| [Crd2Idx](struct/tensor/Crd2Idx.md) | Coordinate转Index |
| [MakeTensor](struct/tensor/MakeTensor.md) | 构造LocalTensor |
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
| [DataCopy](data_move/DataCopyGM2L1.md) | CUBE计算，支持GM2L1搬运处理 |
| [TileCopy] | CUBE计算，支持L12L0搬运 |
| [FixPipe](data_move/Fixpipe.md) | FixPIPE搬出接口 |
| [LoadData](data_move/LoadData.md) | L12L0加载接口 |

## 矩阵计算

标量操作类API，单独使用时可以引入tensor_tile_arch.h，此类API列表如下：

|   API名称   |   说明   |
|----------|-----------|
| [Mmad](cube_compute/Mmad.md) | 完成矩阵乘加操作。 |
