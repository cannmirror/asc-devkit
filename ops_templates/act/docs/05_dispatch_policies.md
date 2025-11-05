# Block Dispatch Policies说明
DispatchPolicy是BlockMmad的一个重要模板参数，各个DispatchPolicy定义在[`dispatch_policy.h`](../include/matmul/policy/dispatch_policy.h)中。本文档对下列DispatchPolicy的调度策略和使用限制进行简单介绍。
- MatmulNaivePipelineWithLayout
- MatmulMultiBlockWithLayout
- MatmulMultiBlockBiasWithLayout
- MatmulMultiBlockOnKAxisWithLayout
- SparseMatmulMultiBlockOnKAxisWithLayout
- MatmulMultiBlock
- MatmulMultiBlockBias
- MatmulL1Input
- MatmulL1InputBias
- MatmulL1InputWithLayout
- MatmulL1InputBiasWithLayout
- MatmulL0COutputWithLayout
- QuantMatmulMultiBlock
- MatmulMultiBlockWithStreamK
- MatmulIterBatch
- MatmulMultiBlockWithOutQue
- GMMPerTile
## MatmulNaivePipelineWithLayout
功能：简单流水，未做优化。

使用限制：
- 输入/输出类型：bf16输入，bf16/fp32输出；fp16输入，fp16/fp32输出；fp32输入，fp32输出。
- 输入/输出Format：ND/ND_ALIGN输入，ND/ND_ALIGN输出。
- 不支持bias。
- Ascend910B环境支持的TileCopy：CopyWithLayout或者不指定（不指定时默认会选择CopyWithLayout）。
- Ascend910_95环境支持的TileCopy：CopyWithLayout。

当前使用该DispatchPolicy的examples有`07_naive_matmul`。

## MatmulMultiBlockWithLayout
功能：多块搬运流水，Layout输入，转换后调用高阶API。

使用限制：
- 输入/输出类型：bf16输入，bf16/fp32输出；fp16输入，fp16/fp32输出；fp32输入，fp32输出；int8输入，int32输出。
- 输入/输出Format：A矩阵ND/ND_ALIGN，B矩阵ND/ND_ALIGN/NZ，ND/ND_ALIGN输出。
- 不支持bias。
- Ascend910B环境不支持自定义的TileCopy。
- Ascend910_95环境支持的TileCopy：CopyOutSplitMWithParams，CopyOutSplitNWithParams。

当前使用该DispatchPolicy的examples有`00_basic_matmul`、`01_misplace_core_matmul`、`02_batch_matmul`、`03_quant_matmul`、`04_l2_misplace_core_matmul`、`05_l2_misplace_core_batchmatmul`、`06_l2_misplace_core_quant_matmul`。

## MatmulMultiBlockBiasWithLayout
功能：多块搬运流水，Layout输入，转换后调用高阶API。

使用限制：
- 输入/输出类型：bf16输入，bf16/fp32输出；fp16输入，fp16/fp32输出；fp32输入，fp32输出；int8输入，int32输出。
- 输入/输出Format：A矩阵ND/ND_ALIGN，B矩阵ND/ND_ALIGN/NZ，ND/ND_ALIGN输出。
- Ascend910B环境不支持自定义的TileCopy。
- Ascend910_95环境支持的TileCopy：CopyOutSplitMWithParams，CopyOutSplitNWithParams。

## MatmulMultiBlockOnKAxisWithLayout
功能：使能MDL特性，在k轴实现缓存。

使用限制：
- 输入/输出类型：bf16输入，bf16输出；fp16输入，fp16输出。
- 输入/输出Format：ND/ND_ALIGN输入，ND/ND_ALIGN输出。
- 不支持bias。
- 仅支持Ascend910B环境，支持的TileCopy：CopyEnUnitFlagWithLayout或者不指定（不指定时默认会选择CopyEnUnitFlagWithLayout）。

## SparseMatmulMultiBlockOnKAxisWithLayout
功能：使能MDL特性，在k轴实现缓存，4:2稀疏矩阵专用。

使用限制：
- 输入/输出类型：int8输入，int32输出；稀疏索引矩阵uint8。
- 输入/输出Format：A/B矩阵ND/ND_ALIGN，稀疏索引矩阵NZ，C矩阵ND/ND_ALIGN。
- 输入Shape：<code>N % 16 = 0</code>，<code>K % 64 = 0</code>。
- B矩阵只支持转置输入。
- 不支持bias。
- 仅支持Ascend910B环境，支持的TileCopy：CopySparseWithLayout或者不指定（不指定时默认会选择CopySparseWithLayout）。

当前使用该DispatchPolicy的examples有`08_sparse_matmul`。

## MatmulMultiBlock
功能：多块搬运流水，基于高阶API接口，其他约束由TileCopy决定。

使用限制：
- 输入/输出类型：bf16输入，bf16/fp32输出；fp16输入，fp16/fp32输出；fp32输入，fp32输出；int8输入，int32输出。
- 输入/输出Format：A矩阵ND/ND_ALIGN，B矩阵ND/ND_ALIGN/NZ，ND/ND_ALIGN输出。
- 不支持bias。
- Ascend910B环境支持的TileCopy：CopyWithParams或者不指定（不指定时默认会选择CopyWithParams）。
- Ascend910_95环境支持的TileCopy：CopyWithParams。

## MatmulMultiBlockBias
功能：多块搬运流水，基于高阶API接口，其他约束由TileCopy决定。

使用限制：
- 输入/输出类型：bf16输入，bf16/fp32输出；fp16输入，fp16/fp32输出；fp32输入，fp32输出；int8输入，int32输出。
- 输入/输出Format：A矩阵ND/ND_ALIGN，B矩阵ND/ND_ALIGN/NZ，ND/ND_ALIGN输出。
- Ascend910B环境支持的TileCopy：CopyWithParams或者不指定（不指定时默认会选择CopyWithParams）。
- Ascend910_95环境支持的TileCopy：CopyWithParams。

## MatmulL1Input
功能：L1输入流水，基于高阶API接口。

使用限制：
- 输入/输出类型：bf16输入，bf16/fp32输出；fp16输入，fp16/fp32输出；fp32输入，fp32输出；int8输入，int32输出。
- 输入/输出Format：A矩阵NZ，B矩阵NZ，ND/ND_ALIGN输出。
- 不支持bias。
- 仅支持Ascend910_95环境，支持的TileCopy：CopyNoGmIn。

## MatmulL1InputBias
功能：L1输入流水，基于高阶API接口。

使用限制：
- 输入/输出类型：bf16输入，bf16/fp32输出；fp16输入，fp16/fp32输出；fp32输入，fp32输出；int8输入，int32输出。
- 输入/输出Format：A矩阵NZ，B矩阵NZ，ND/ND_ALIGN输出。
- 仅支持Ascend910_95环境，支持的TileCopy：CopyNoGmIn。

## MatmulL1InputWithLayout
功能：L1输入流水，Layout输入，转换后调用高阶API。

使用限制：
- 输入/输出类型：bf16输入，bf16/fp32输出；fp16输入，fp16/fp32输出；fp32输入，fp32输出；int8输入，int32输出。
- 输入/输出Format：A矩阵NZ，B矩阵NZ，ND/ND_ALIGN输出。
- 不支持bias。
- 仅支持Ascend910_95环境，支持的TileCopy：CopyNoGmIn。

## MatmulL1InputBiasWithLayout
功能：L1输入流水，Layout输入，转换后调用高阶API。

使用限制：
- 输入/输出类型：bf16输入，bf16/fp32输出；fp16输入，fp16/fp32输出；fp32输入，fp32输出；int8输入，int32输出。
- 输入/输出Format：A矩阵NZ，B矩阵NZ，ND/ND_ALIGN输出。
- 仅支持Ascend910_95环境，支持的TileCopy：CopyNoGmIn。

## MatmulL0COutputWithLayout
功能：L0输出流水，基于Layout实现。

使用限制：
- 输入/输出类型：bf16输入，fp32输出；fp16输入，fp32输出。
- 输入/输出Format：A矩阵ND/ND_ALIGN，B矩阵ND/ND_ALIGN/NZ，ND/ND_ALIGN输出。
- 不支持bias。
- 仅支持Ascend910_95环境，支持的TileCopy：CopyWithLayout或者不指定（不指定时默认会选择CopyWithLayout）。

## QuantMatmulMultiBlock
功能：多块矩阵乘法策略。

使用限制：
- 输入/输出类型：fp8_e5m2_t/fp8_e4m3fn_t输入，fp32输出。
- 输入/输出Format：A矩阵ND/ND_ALIGN，B矩阵ND/ND_ALIGN/NZ，ND/ND_ALIGN输出。
- 不支持bias。
- 仅支持Ascend910_95环境，不支持自定义的TileCopy。

## MatmulMultiBlockWithStreamK
功能：矩阵乘法拆分为k轴处理结构，基于Layout实现。

使用限制：
- 输入/输出类型：bf16输入，bf16/fp32输出；fp16输入，fp16/fp32输出；fp32输入，fp32输出。
- 输入/输出Format：ND/ND_ALIGN输入，ND/ND_ALIGN输出。
- 仅支持Ascend910_95环境，不支持自定义的TileCopy。

## MatmulIterBatch
功能：矩阵乘法迭代批量处理结构，基于Layout实现。

使用限制：
- 输入/输出类型：bf16输入，bf16/fp32输出；fp16输入，fp16/fp32输出；fp32输入，fp32输出。
- 输入/输出Format：ND/ND_ALIGN输入，ND/ND_ALIGN输出。
- 不支持bias。
- 仅支持Ascend910_95环境，不支持自定义的TileCopy。

## MatmulMultiBlockWithOutQue
功能：计算一个基本块的结果。

使用限制：
- 输入/输出类型：bf16输入，bf16输出；fp16输入，fp16输出；fp32输入，fp32输出。
- 输入/输出Format：ND/ND_ALIGN输入，ND/ND_ALIGN输出。
- 仅支持Ascend910_95环境，不支持自定义的TileCopy。

## GMMPerTile
功能：每baseK计算即输出到UB。

使用限制：
- 输入/输出类型：fp8_e5m2_t/fp8_e4m3fn_t/hifloat8输入，fp32输出。
- 输入/输出Format：ND_ALIGN输入，ND输出。
- 不支持bias。
- 仅支持Ascend910_95环境，不支持自定义的TileCopy。
