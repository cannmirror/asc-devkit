/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file cube_in_buffer_n_buffer.h
 * \brief
 */

#ifndef AICORE_ADV_API_DETAIL_MATMUL_RESOURCE_CUBE_IN_BUFFER_CUBE_IN_BUFFER_N_BUFFER_H
#define AICORE_ADV_API_DETAIL_MATMUL_RESOURCE_CUBE_IN_BUFFER_CUBE_IN_BUFFER_N_BUFFER_H

#include "cube_in_buffer_intf.h"
#include "../../param/matmul_shape_tiling.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    CubeInBuffer is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    CubeInBuffer is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class INPUT_TYPE, const auto& MM_CFG, PolicyType POLICY_TYPE>
class CubeInBuffer<IMPL, INPUT_TYPE, MM_CFG, POLICY_TYPE,
    enable_if_t<!MatmulFeatureTrait<MM_CFG>::IsNeedUB()
                && GetCubeInBufferType<INPUT_TYPE, MM_CFG>() == CubeInBufferType::DOUBLE_BUFFER
                && POLICY_TYPE == PolicyType::MATMUL_NBUFFER_33 && INPUT_TYPE::TAG == InputTypeTag::A>> {
    MATMUL_USE_MODULE(MatmulShapeTiling);
    using TransT = typename INPUT_TYPE::TRANS_T;

public:
    __aicore__ inline CubeInBuffer() {}
    __aicore__ inline ~CubeInBuffer() {}
    __aicore__ inline void Init(int32_t baseBlockSize, int32_t cacheNum)
    {
        int32_t matrixByteSize = baseBlockSize * AscendC::GetBitSize<TransT>() / ONE_BYTE_BIT_SIZE;
        const auto& tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        GetTPipePtr()->InitBuffer(qid_, Ceil(tiling.GetSingleCoreM(), tiling.GetBaseM()),
            matrixByteSize * Ceil(tiling.GetSingleCoreK(), tiling.GetBaseK()));
    }

    __aicore__ inline void Destroy()
    {
        isCachingPing_ = false;
        isCachingPong_ = false;
        isCachingThirdBuffer_ = false;
        qid_.FreeAllEvent();
    }

    __aicore__ inline LocalTensor<TransT> AllocTensor(int32_t bufferPos = -1)
    {
        ASCENDC_ASSERT(bufferPos != -1,
            { KERNEL_LOG(KERNEL_ERROR, "bufferPos in AllocTensor for only db version should not be -1."); });
        LocalTensor<TransT> tensor = qid_.template AllocTensor<TransT>();
        SetCache(bufferPos, tensor);
        SetBufferCaching(bufferPos, true);
        return tensor;
    }

    __aicore__ inline void FreeTensor(int32_t bufferPos = -1, const LocalTensor<TransT>& tensor = NULL_TENSOR<TransT>)
    {
        ASCENDC_ASSERT(bufferPos != -1,
            { KERNEL_LOG(KERNEL_ERROR, "bufferPos in FreeTensor for only db version should not be -1."); });
        if (IsBufferCaching(bufferPos)) {
            qid_.FreeBuffer(GetCache(bufferPos));
            SetBufferCaching(bufferPos, false);
        }
    }

    __aicore__ inline void Reset()
    {
        for (int32_t pos = 0; pos < N_BUFFER_33_FACTOR; pos++) {
            if (IsBufferCaching(pos)) {
                qid_.FreeBuffer(GetCache(pos));
                SetBufferCaching(pos, false);
            }
        }
    }

    __aicore__ inline bool Hit(int32_t iterIndex, int32_t bufferPos = -1)
    {
        ASCENDC_ASSERT(
            bufferPos != -1, { KERNEL_LOG(KERNEL_ERROR, "bufferPos in Hit for only db version should not be -1."); });
        return IsBufferCaching(bufferPos);
    }

    __aicore__ inline LocalTensor<TransT> GetBuffer(int32_t iterIndex, int32_t bufferPos = -1)
    {
        (void)iterIndex;
        ASCENDC_ASSERT(
            bufferPos != -1, { KERNEL_LOG(KERNEL_ERROR, "bufferPos in Hit for only db version should not be -1."); });
        LocalTensor<TransT> tensor;
        tensor.SetAddr(qid_.GetBufferAddr(GetCache(bufferPos)));
        return tensor;
    }

    __aicore__ inline void EnQue(LocalTensor<TransT>& tensor)
    {
        qid_.EnQue(tensor);
    }

    __aicore__ inline void DeQue()
    {
        (void)qid_.DeQue();
    }

private:
    __aicore__ inline auto& GetCache(int32_t pos)
    {
        if (pos == 0) {
            return cachePing_;
        } else if (pos == 1) {
            return cachePong_;
        } else {
            return cacheThirdBuffer_;
        }
    }

    __aicore__ inline bool IsBufferCaching(int32_t pos)
    {
        if (pos == 0) {
            return isCachingPing_;
        } else if (pos == 1) {
            return isCachingPong_;
        } else {
            return isCachingThirdBuffer_;
        }
    }

    __aicore__ inline void SetCache(int32_t pos, const LocalTensor<TransT>& cacheTensor)
    {
        if (pos == 0) {
            cachePing_ = cacheTensor.GetBufferHandle();
        } else if (pos == 1) {
            Barrier();
            cachePong_ = cacheTensor.GetBufferHandle();
        } else {
            Barrier();
            cacheThirdBuffer_ = cacheTensor.GetBufferHandle();
        }
    }

    __aicore__ inline void SetBufferCaching(int32_t pos, bool isCaching)
    {
        if (pos == 0) {
            isCachingPing_ = isCaching;
        } else if (pos == 1) {
            Barrier();
            isCachingPong_ = isCaching;
        } else {
            Barrier();
            isCachingThirdBuffer_ = isCaching;
        }
    }

private:
    typename CubeInQueType<INPUT_TYPE>::QUE qid_;
    TBufHandle cachePing_;
    TBufHandle cachePong_;
    TBufHandle cacheThirdBuffer_;
    bool isCachingPing_{false};
    bool isCachingPong_{false};
    bool isCachingThirdBuffer_{false};
};

} // namespace Detail
} // namespace Impl
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_MATMUL_RESOURCE_CUBE_IN_BUFFER_CUBE_IN_BUFFER_N_BUFFER_H
