/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mul_fusion_op.h
 * \brief
 */

#ifndef EPILOGUE_FUSION_MUL_FUSION_OP_H
#define EPILOGUE_FUSION_MUL_FUSION_OP_H
#include "kernel_operator.h"
#include "../../utils/common_utils.h"
#include "../../utils/device_utils.h"

namespace Act {
namespace Gemm {
namespace Block {
template <typename DataTypeOut_, typename DataTypeIn_>
class MulFusion {
public:
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    __aicore__ inline MulFusion(){};

    struct Arguments {
        GM_ADDR inputGmAddr{nullptr};
    };

    struct Params {
        GM_ADDR inputGmAddr{nullptr};
    };

    AscendC::GlobalTensor<DataTypeIn> inputGlobal_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_COUNT> vecQueInput_;

    int64_t ubCalcM_{0};
    int64_t ubCalcN_{0};
    int64_t strideN_{0};

    __aicore__ inline void InitBuffers()
    {
        GetTPipePtr()->InitBuffer(vecQueInput_, DOUBLE_BUFFER_COUNT, ubCalcM_ * ubCalcN_ * sizeof(DataTypeIn));
    }

    __aicore__ inline void Init(Params const& params, int64_t ubCalcM, int64_t ubCalcN, int64_t n)
    {
        inputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeIn*>(params.inputGmAddr));
        ubCalcM_ = ubCalcM;
        ubCalcN_ = ubCalcN;
        strideN_ = n;
        InitBuffers();
    }

    __aicore__ inline int64_t GetUbSizeOneM(int64_t ubCalcN)
    {
        return sizeof(DataTypeIn) * DOUBLE_BUFFER_COUNT * ubCalcN;
    }

    __aicore__ inline int64_t GetInputOffset(int64_t mIdx, int64_t nIdx)
    {
        return mIdx * strideN_ + nIdx;
    }

    __aicore__ inline void Run(LocalTensor<DataTypeOut>& dstLocal, LocalTensor<DataTypeIn>& srcLocal, int64_t curAivM,
                               int64_t curAivN, int64_t mIdx, int64_t nIdx)
    {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopyParams gm2UbParams{static_cast<uint16_t>(curAivM),
                                            static_cast<uint16_t>(curAivN * sizeof(DataTypeIn)),
                                            (strideN_ - curAivN) * sizeof(DataTypeIn), 0};
        DataCopyPadParams padParams;
        LocalTensor<DataTypeIn> inputLocal = vecQueInput_.AllocTensor<DataTypeIn>();
        DataCopyPad(inputLocal, inputGlobal_[GetInputOffset(mIdx, nIdx)], gm2UbParams, padParams);
        TPipeSetWaitFlag<HardEvent::MTE2_V>();
        int64_t computedAivN = CeilAlign(curAivN, UB_FLOAT_ALIGN_NUM);
        Mul(dstLocal, srcLocal, inputLocal, static_cast<int32_t>(computedAivN * curAivM));
        return;
    }

    __aicore__ inline void operator()(LocalTensor<DataTypeOut>& dstLocal, LocalTensor<DataTypeIn>& srcLocal,
                                      int64_t curAivM, int64_t curAivN, int64_t mIdx, int64_t nIdx)
    {
        Run(dstLocal, srcLocal, curAivM, curAivN, mIdx, nIdx);
    }
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif // EPILOGUE_FUSION_MUL_FUSION_OP_H
