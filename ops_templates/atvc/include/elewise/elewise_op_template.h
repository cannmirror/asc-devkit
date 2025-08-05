/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_ELEWISE_OP_TEMPLATE_H
#define ATVC_ELEWISE_OP_TEMPLATE_H

#include <type_traits>
#include "kernel_operator.h"
#include "common/tensor_info.h"
#include "common/type_list.h"
#include "common/index_seq.h"
#include "common/atvc_opdef.h"
#include "common/const_def.h"
#include "elewise/common/elewise_common.h"

namespace ATVC {
namespace Kernel {
template <class EleWiseCompute>
class EleWiseOpTemplate {
    using EleWiseOpTraits = typename GetFunctionTraits<EleWiseCompute>::ComputeTraits;
    using Inputs = typename EleWiseOpTraits::In::types;
    using Outputs = typename EleWiseOpTraits::Out::types;
    using Temps = typename EleWiseOpTraits::Temp::types;
    // xxCount表示TensorList里面有几个Tensor
    static constexpr size_t InputCount = ATVC::TypeListSize<Inputs>::value;
    static constexpr size_t OutputCount = ATVC::TypeListSize<Outputs>::value;
    static constexpr size_t TempCount = ATVC::TypeListSize<Temps>::value;
    // xxTensroSumBytes表示TensorList里面所有数据类型长度的累加值， xxTensroSumBytes = sum(sizeof(Tensor_i::type))
    static constexpr size_t inTensorSumBytes = ATVC::TypeListReduce<Inputs, SizeValue<0>, SumSizes>::Type::value;
    static constexpr size_t outTensorSumBytes = ATVC::TypeListReduce<Outputs, SizeValue<0>, SumSizes>::Type::value;
    static constexpr size_t tempTensorSumBytes = ATVC::TypeListReduce<Temps, SizeValue<0>, SumSizes>::Type::value;

public:
    __aicore__ inline EleWiseOpTemplate() {}
    template<typename ...Args>
    __aicore__ inline void Run(Args&&... args) {
        g_pipe.Reset();
        constexpr std::size_t GM_ARGS_COUNT = InputCount + OutputCount;
        GM_ADDR argsArr[InputCount + OutputCount];
        InitHelper<0>(argsArr, ATVC::Forward<Args>(args)...);
    }
private:
    template<size_t N>
    __aicore__ inline void InitHelper(GM_ADDR tensorsGMArr[]) {
    }

    template <size_t N, class T0, class... Ts>
    __aicore__ inline void InitHelper(GM_ADDR tensorsGMArr[], T0 t0, Ts... ts)
    {
        if constexpr (N < (InputCount + OutputCount)) {
            tensorsGMArr[N] = t0;
            InitHelper<N + 1>(tensorsGMArr, ts...);
        } else if constexpr (N == (InputCount + OutputCount)) {
            FillAddrs(tensorsGMArr);
            FillOffsets<Inputs>(inOffsets_);
            FillOffsets<Outputs>(outOffsets_);
            FillOffsets<Temps>(tempOffsets_);

            this->param_ = reinterpret_cast<__gm__ ATVC::EleWiseParam*>(t0);
            uint32_t curBlockId = AscendC::GetBlockIdx();

            if (curBlockId < param_->tilingData.tailBlockCnt) {
                this->curCoreCnt_ = (param_->tilingData.numPerBlock + 1) * param_->tilingData.tiledCnt;
                this->curCoreStartCnt_ = (param_->tilingData.numPerBlock + 1) *
                                          curBlockId * param_->tilingData.tiledCnt;
            } else {
                this->curCoreCnt_ = param_->tilingData.numPerBlock * param_->tilingData.tiledCnt;
                this->curCoreStartCnt_ = (curBlockId * param_->tilingData.numPerBlock +
                                          param_->tilingData.tailBlockCnt) * param_->tilingData.tiledCnt;
            }
            if (curBlockId + 1 == param_->tilingData.blockNum) {
                this->curCoreCnt_ += param_->tilingData.tailElemCnt;
            }
            if (this->curCoreCnt_ == 0) {
                return;
            }
            Init();
            Process(ATVC::Forward<Ts>(ts)...);
        }
    }

private:

    // 申请LocalTensor等资源，初始化本核计算的GlobalTensor
    __aicore__ inline void Init()
    {
        // in/out/temp各自使用一个pipe进行管理，每个pipe里面管理的是ub地址连续的多个tensor
        if constexpr (InputCount > 0) {
            g_pipe.InitBuffer(inQueue, param_->nBufferNum, param_->tilingData.tiledCnt * inTensorSumBytes);
        }
        if constexpr (OutputCount > 0) {
            g_pipe.InitBuffer(outQueue, param_->nBufferNum, param_->tilingData.tiledCnt * outTensorSumBytes);
        }
        if constexpr(TempCount > 0) {
            g_pipe.InitBuffer(tempQueue, param_->tilingData.tiledCnt * tempTensorSumBytes);
        }
    }
    // 根据tiling循环调用CopyIn/CopyOut，以及外部传入的Compute计算
    // 如果有尾块，则处理尾块的CopyIn/Compute/CopyOut
    template<typename ...Args>
    __aicore__ inline void Process(Args&&... args) {
        typename TensorTuple<Inputs>::Type inTensors;
        typename TensorTuple<Outputs>::Type outTensors;
        typename TensorTuple<Temps>::Type tempTensors;

        InitInputTensors(inTensors, param_->tilingData.tiledCnt, ATVC::MakeIndexSequence<InputCount>{});
        InitOutputTensors(outTensors, param_->tilingData.tiledCnt, ATVC::MakeIndexSequence<OutputCount>{});
        InitTempTensors(tempTensors, param_->tilingData.tiledCnt, ATVC::MakeIndexSequence<TempCount>{});
        
        uint32_t repeat = curCoreCnt_ / param_->tilingData.tiledCnt;
        uint32_t tailCnt = curCoreCnt_ % param_->tilingData.tiledCnt;
        offsetCnt_ = 0;
        caclCnt_ = param_->tilingData.tiledCnt;
        // 循环处理主块数据
        for (uint32_t i = 0; i < repeat; i++)
        {
            CopyIn(inTensors, ATVC::MakeIndexSequence<InputCount>{});
            Compute(inTensors, outTensors, tempTensors, 
                ATVC::MakeIndexSequence<InputCount>{}, 
                ATVC::MakeIndexSequence<OutputCount>{}, 
                ATVC::MakeIndexSequence<TempCount>{}, 
                ATVC::Forward<Args>(args)...);
            CopyOut(outTensors, ATVC::MakeIndexSequence<OutputCount>{});
            offsetCnt_ += caclCnt_;
        }
        // 如果有尾块，则处理尾块
        if (tailCnt > 0)
        {
            caclCnt_ = tailCnt;
            CopyIn(inTensors, ATVC::MakeIndexSequence<InputCount>{});
            Compute(inTensors, outTensors, tempTensors,
                ATVC::MakeIndexSequence<InputCount>{},
                ATVC::MakeIndexSequence<OutputCount>{},
                ATVC::MakeIndexSequence<TempCount>{},
                ATVC::Forward<Args>(args)...);
            CopyOut(outTensors, ATVC::MakeIndexSequence<OutputCount>{});
        }
    }
    // 模拟单个 Tensor 的处理逻辑：入参为类型 T 对应的 Tensor 变量
    template<typename T>
    __aicore__ inline void CopyInAllTensors(AscendC::LocalTensor<uint8_t>& inLocal, int32_t i, T& tensorInfo) {
        // 单个 Tensor 的处理逻辑
        auto inLocal_i = inLocal[tensorInfo.local_offset].template ReinterpretCast<typename T::Dtype>();

        using DataType = typename T::Dtype;
        constexpr uint32_t typeAlignCnt = UB_ALIGN_32 / sizeof(DataType);
        uint32_t alignMainCnt = caclCnt_/typeAlignCnt*typeAlignCnt;
        uint32_t alignTailCnt = caclCnt_ - alignMainCnt;
        if (alignMainCnt > 0) {
            AscendC::DataCopy(inLocal_i, tensorInfo.gmTensor[curCoreStartCnt_ + offsetCnt_], alignMainCnt);
        }
        if (alignTailCnt > 0) {
            struct AscendC::DataCopyExtParams repeatParams = {1, (uint16_t)(alignTailCnt*sizeof(DataType)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<DataType> padParams;
            AscendC::DataCopyPad(inLocal_i[alignMainCnt],
                tensorInfo.gmTensor[curCoreStartCnt_ + offsetCnt_ + alignMainCnt],
                repeatParams, padParams);
        }
    }

    // 对应于没有 Tensor 时候的空处理逻辑
    __aicore__ inline void CopyInAllTensors(AscendC::LocalTensor<uint8_t>& inLocal, int32_t i) {
    }

    // 所有 Tensor 的处理入口逻辑：递归完成对每个 Tensor 的处理
    template<typename T, typename... Ts>
    __aicore__ inline void CopyInAllTensors(AscendC::LocalTensor<uint8_t>& inLocal, int32_t i, T& first, Ts&... rest) {
        CopyInAllTensors(inLocal, i, first);
        CopyInAllTensors(inLocal, ++i, rest...);
    }

    // 将所有输入tensor从gm拷贝到local
    template<typename InTuple, std::size_t... Is>
    __aicore__ inline void CopyIn(InTuple& inTensors, ATVC::IndexSequence<Is...>) {
        if constexpr (InputCount == 0) {
            return;
        }
        AscendC::LocalTensor<uint8_t> inLocal = inQueue.template AllocTensor<uint8_t>();
        // TypeListGetOffset是从TypeList里面获得当前Tensor以Bytes为单位的偏移
        // 例如TypeList<float, half, u8>，tensor_0的偏移为0， tensor_1的偏移为sizeof(float),
        //       tensor_2的偏移为sizeof(float)+sizeof(half)
        CopyInAllTensors(inLocal, 0, ATVC::TupleElemGet<Is>(inTensors)...);
        inQueue.EnQue(inLocal);
    }

    // 模拟单个 Tensor 的处理逻辑：入参为类型 T 对应的 Tensor 变量
    template<typename T>
    __aicore__ inline void CopyOutAllTensors(AscendC::LocalTensor<uint8_t>& outLocal, int32_t i, T& tensorInfo) {
        // 单个 Tensor 的处理逻辑
        auto outLocal_i = outLocal[tensorInfo.local_offset].template ReinterpretCast<typename T::Dtype>();
        using DataType = typename T::Dtype;
        constexpr uint32_t typeAlignCnt = 32 / sizeof(DataType);
        uint32_t alignMainCnt = caclCnt_/typeAlignCnt*typeAlignCnt;
        uint32_t alignTailCnt = caclCnt_ - alignMainCnt;
        if (alignMainCnt > 0) {
            AscendC::DataCopy(tensorInfo.gmTensor[curCoreStartCnt_ + offsetCnt_], outLocal_i, alignMainCnt);
        }
        if (alignTailCnt > 0) {
            struct AscendC::DataCopyParams repeatParams = {1, (uint16_t)(alignTailCnt*sizeof(DataType)), 0, 0};
            AscendC::DataCopyPad(tensorInfo.gmTensor[curCoreStartCnt_ + offsetCnt_ + alignMainCnt],
                outLocal_i[alignMainCnt], repeatParams);
        }
    }

    // 对应于没有 Tensor 时候的空处理逻辑
    __aicore__ inline void CopyOutAllTensors(AscendC::LocalTensor<uint8_t>& outLocal, int32_t i) {
    }

    // 所有 Tensor 的处理入口逻辑：递归完成对每个 Tensor 的处理
    template <typename T, typename... Ts>
    __aicore__ inline void
    CopyOutAllTensors(AscendC::LocalTensor<uint8_t> &outLocal, int32_t i,
                      T &first, Ts &...rest)
    {
      CopyOutAllTensors(outLocal, i, first);
      CopyOutAllTensors(outLocal, ++i, rest...);
    }
    // 将所有输出tensor拷贝到gm
    template<typename OutTuple, std::size_t... Is>
    __aicore__ inline void CopyOut(OutTuple& outTensors, ATVC::IndexSequence<Is...>) {
        AscendC::LocalTensor<uint8_t> outLocal = outQueue.template DeQue<uint8_t>();
        CopyOutAllTensors(outLocal, 0, ATVC::TupleElemGet<Is>(outTensors)...);
        outQueue.FreeTensor(outLocal);
    }

template<typename InTuple, typename OutTuple, typename TmpTuple,
    std::size_t... I1, std::size_t... I2,  std::size_t... I3, typename ...Args>
__aicore__ inline void Compute(InTuple& inTensors, OutTuple& outTensors, TmpTuple& tempTensors,
    ATVC::IndexSequence<I1...>, ATVC::IndexSequence<I2...>, ATVC::IndexSequence<I3...>, Args&&... args)
{
    AscendC::LocalTensor<uint8_t> inLocal = inQueue.template DeQue<uint8_t>();
    AscendC::LocalTensor<uint8_t> outLocal = outQueue.template AllocTensor<uint8_t>();
    AscendC::LocalTensor<uint8_t> tempLocal;
    if constexpr(TempCount > 0) {
        tempLocal = tempQueue.Get<uint8_t>();
    }
    compute_(TupleElemGetLocalTensor<I1>(inLocal, inTensors, this->caclCnt_)...,
        TupleElemGetLocalTensor<I2>(outLocal, outTensors, this->caclCnt_)...,
        TupleElemGetLocalTensor<I3>(tempLocal, tempTensors, this->caclCnt_)...,
        ATVC::Forward<Args>(args)...);
    inQueue.FreeTensor(inLocal);
    outQueue.template EnQue<uint8_t>(outLocal);
}

private:
    template <typename InTuple, std::size_t... Is>
    __aicore__ inline void InitInputTensors(InTuple& tuple, std::size_t cnt, ATVC::IndexSequence<Is...>) {
        // 初始化每个 Tensor
        int32_t dummy[] = { 0, (InitInputTensor(ATVC::TupleElemGet<Is>(tuple), cnt, Is), 0)... };
        (void)dummy; // 避免未使用变量警告
    }

    template <typename OutTuple, std::size_t... Is>
    __aicore__ inline void InitOutputTensors(OutTuple& tuple, std::size_t cnt, ATVC::IndexSequence<Is...>) {
        int32_t dummy[] = { 0, (InitOutputTensor(ATVC::TupleElemGet<Is>(tuple), cnt, Is), 0)... };
        (void)dummy;
    }

    template <typename TmpTuple, std::size_t... Is>
    __aicore__ inline void InitTempTensors(TmpTuple& tuple, std::size_t cnt, ATVC::IndexSequence<Is...>) {
        int32_t dummy[] = { 0, (InitTempTensor(ATVC::TupleElemGet<Is>(tuple), cnt, Is), 0)... };
        (void)dummy;
    }

    template <typename T>
    __aicore__ inline TensorInfo<T>& InitInputTensor(TensorInfo<T>& tensor, std::size_t cnt, std::size_t index) {
        tensor.gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inGMAddrs_[index].GetPhyAddr(0)));
        tensor.local_offset = inOffsets_[index] * cnt;
        return tensor;
    }

    template <typename T>
    __aicore__ inline TensorInfo<T>& InitOutputTensor(TensorInfo<T>& tensor, std::size_t cnt, std::size_t index) {
        tensor.gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outGMAddrs_[index].GetPhyAddr(0)));
        tensor.local_offset = outOffsets_[index] * cnt;
        return tensor;
    }

    template <typename T>
    __aicore__ inline TensorInfo<T>& InitTempTensor(TensorInfo<T>& tensor, std::size_t cnt, std::size_t index) {
        tensor.local_offset = tempOffsets_[index] * cnt;
        return tensor;
    }

private:
    // 填充 addr 到数组
    __aicore__ inline void FillAddrs(GM_ADDR argsArr[])
    {
        for (std::size_t i = 0; i < InputCount; ++i) {
            inGMAddrs_[i].SetGlobalBuffer(argsArr[i]);
        }
        for (std::size_t i = 0; i < OutputCount; ++i) {
            outGMAddrs_[i].SetGlobalBuffer(argsArr[InputCount + i]);
        }
    }

    // 填充 offset 到数组
    template <typename List, std::size_t... Is>
    __aicore__ inline constexpr void FillOffsetsImpl(std::size_t* offsets, ATVC::IndexSequence<Is...>) {
        ((offsets[Is] = ATVC::TypeListByteOffset<List, Is>::value), ...);
    }

    template <typename List>
    __aicore__ inline constexpr void FillOffsets(std::size_t* offsets) {
        constexpr std::size_t count = ATVC::TypeListSize<List>::value;
        FillOffsetsImpl<List>(offsets, ATVC::MakeIndexSequence<count>{});
    }

private:
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tempQueue;
    
    // 全局变量
    AscendC::GlobalTensor<uint8_t> inGMAddrs_[InputCount];
    AscendC::GlobalTensor<uint8_t> outGMAddrs_[OutputCount];

    std::size_t inOffsets_[InputCount];
    std::size_t outOffsets_[OutputCount];
    std::size_t tempOffsets_[TempCount];

    // 计算得到的tiling数据
    __gm__ EleWiseParam* param_;

    uint32_t curCoreCnt_;
    uint32_t curCoreStartCnt_;
    int32_t offsetCnt_;
    int32_t caclCnt_;

    // 算子开发者传入的计算对象
    EleWiseCompute compute_;
};
}
}
#endif