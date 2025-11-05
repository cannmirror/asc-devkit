# Copy Policies说明
CopyPolicy是BlockMmad的一个重要模板参数，各个CopyPolicy定义在[`tile_copy_policy.h`](../include/matmul/tile/tile_copy_policy.h)中。本文档对下列CopyPolicy的进行简单介绍。
- CopyWithLayout
- CopyEnUnitFlagWithLayout
- CopySparseWithLayout
- CopyWithParams
- CopyOutSplitMWithParams
- CopyOutSplitNWithParams
- CopyNoGmIn
- CopyInAndCopyOutSplitMWithParams

## CopyWithLayout
功能：基础拷贝策略，支持Ascend910B/Ascend910_95环境且使用限制相同。

使用限制：
- CopyIn的TPosition和Format组合支持：GM，ND/ND_ALIGN/NZ。
- 在Ascend910B环境下CopyOut的TPosition，Format和是否量化组合支持：GM，ND/ND_ALIGN/NZ，非量化。
- 在Ascend910_95环境下CopyOut的TPosition，Format和是否量化组合支持：GM，ND/ND_ALIGN/NZ，非量化；UB，ND/NZ，非量化。

## CopyEnUnitFlagWithLayout
功能：带`unitFlag`参数的`CopyCo1ToOut`，详见[《Ascend C算子开发接口》](https://www.hiascend.com/document/redirect/CannCommunityAscendCApi)的“基础API > 矩阵计算(ISASI) > Fixpipe”章节中的`unitFlag`参数，支持Ascend910B环境。

使用限制：
- CopyIn的TPosition和Format组合支持：GM，ND/ND_ALIGN/NZ。
- CopyOut的TPosition，Format和是否量化组合支持：GM，ND/ND_ALIGN/NZ，非量化。

## CopySparseWithLayout
功能：SparseMatmul专用，支持从L1 Buffer搬运B矩阵到L0B Buffer的同时搬运稀疏索引矩阵，支持Ascend910B环境。

使用限制：
- CopyIn的TPosition和Format组合支持：GM，ND/ND_ALIGN/NZ。
- CopyOut的TPosition，Format和是否量化组合支持：GM，ND/ND_ALIGN/NZ，非量化。

## CopyWithParams
功能：带有额外参数的拷贝策略，支持Ascend910B/Ascend910_95环境，不同环境使用限制不同。

使用限制：
- 在Ascend910B环境下CopyIn的TPosition和Format组合支持：GM，ND/NZ。
- 在Ascend910B环境下CopyOut的TPosition，Format和是否量化组合支持：GM，ND/ND_ALIGN/NZ，量化/非量化。
- 在Ascend910_95环境下CopyIn的TPosition和Format组合支持：GM，ND/NZ。
- 在Ascend910_95环境下CopyOut的TPosition，Format和是否量化组合支持：GM/UB，ND/ND_ALIGN/NZ，量化/非量化。

## CopyOutSplitMWithParams
功能：沿M维度分割输出的拷贝策略，带有参数，支持Ascend910_95环境。

使用限制：
- CopyOut的TPosition，Format和是否量化组合支持：UB，ND/ND_ALIGN/NZ，非量化。

## CopyOutSplitNWithParams
功能：沿N维度分割输出的拷贝策略，带有参数，支持Ascend910_95环境。

使用限制：
- CopyOut的TPosition，Format和是否量化组合支持：UB，ND/ND_ALIGN/NZ，非量化。

## CopyNoGmIn
功能：排除全局内存输入的拷贝策略，支持Ascend910_95环境。

使用限制：
- CopyOut的TPosition，Format和是否量化组合支持：GM/UB，ND/ND_ALIGN/NZ，量化/非量化。

## CopyInAndCopyOutSplitMWithParams
功能：沿M维度分割输入和输出的拷贝策略，包含参数，支持Ascend910_95环境。

使用限制：
- CopyIn的TPosition和Format组合支持：GM，ND/NZ。
- CopyOut的TPosition，Format和是否量化组合支持：UB，ND/ND_ALIGN/NZ，非量化。

