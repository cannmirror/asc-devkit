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
 * \file k_loop_mdl_mx.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_MATMUL_SCHEDULER_ITERATOR_K_LOOP_K_LOOP_MDL_MX_H
#define AICORE_ADV_API_DETAIL_MATMUL_SCHEDULER_ITERATOR_K_LOOP_K_LOOP_MDL_MX_H

#include "k_loop_intf.h"
#include "k_loop_mdl_base.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    KLoop for Scale A/B is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    KLoop is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, typename TRANS_T, class A_TYPE, const auto& MM_CFG>
class KLoop<IMPL, TRANS_T, A_TYPE, MM_CFG,
    enable_if_t<(DoMatmulMDL(MM_CFG) && !IsKdimReorderLoad<MM_CFG>)&&HasScalePosition<A_TYPE>::value>>
    : public KLoopMDLBase<IMPL, TRANS_T, A_TYPE, MM_CFG> {
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(CopyCubeOut);
    MATMUL_USE_MODULE(CubeOutBuffer);
    MATMUL_USE_MODULE(CopyCubeInA);
    MATMUL_USE_MODULE(CopyCubeInB);
    MATMUL_USE_MODULE(CopyCubeInScaleA);
    MATMUL_USE_MODULE(CopyCubeInScaleB);
    MATMUL_USE_MODULE(BiasScheduler);
    MATMUL_USE_MODULE(TBufPoolL0);
    MATMUL_USE_MODULE(MatmulSubBlockInfo);
    MATMUL_USE_MODULE(MatmulQuantProcessor);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE(MatmulCrossCoreSync);

public:
    using BASE_MODULE = AscendC::Impl::Detail::KLoopMDLBase<IMPL, TRANS_T, A_TYPE, MM_CFG>;
    __aicore__ inline KLoop() = default;
    __aicore__ inline ~KLoop() = default;

    __aicore__ inline void Init(int32_t singleShape)
    {
        SetSingleShape(singleShape);
    }

    __aicore__ inline void SetSingleShape(int32_t singleShape)
    {
        BASE_MODULE::SetSingleShape(singleShape);
        const auto& tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        int32_t stepKa = tiling.GetStepKa();
        int32_t stepKb = tiling.GetStepKb();
        int32_t baseK = tiling.GetBaseK();
        scaleFactorA_ = tiling.GetScaleFactorA();
        scaleFactorB_ = tiling.GetScaleFactorB();
        // originTailScaleStepKa_ represents the origin size of the GM -> L1 tail blok
        tailScaleStepKa_ = singleShape % (baseK * stepKa * scaleFactorA_);
        tailScaleStepKb_ = singleShape % (baseK * stepKb * scaleFactorB_);
        if (tailScaleStepKa_ == 0) {
            tailScaleStepKa_ = baseK * stepKa * scaleFactorA_;
        }
        if (tailScaleStepKb_ == 0) {
            tailScaleStepKb_ = baseK * stepKb * scaleFactorB_;
        }

        tailScaleStepKa_ = Ceil(tailScaleStepKa_, MX_BASEK_FACTOR) * MX_EVEN_FACTOR;
        tailScaleStepKb_ = Ceil(tailScaleStepKb_, MX_BASEK_FACTOR) * MX_EVEN_FACTOR;

        isScaleA1KFullLoad_ = (stepKa * scaleFactorA_) >= BASE_MODULE::kIter_;
        isScaleB1KFullLoad_ = (stepKb * scaleFactorB_) >= BASE_MODULE::kIter_;
    }

    __aicore__ inline void OuterStart()
    {
        BASE_MODULE::outerIdx_ = 0;
        UpdateOuterParams();
    }

    __aicore__ inline bool OuterNext()
    {
        BASE_MODULE::outerIdx_++;
        if (BASE_MODULE::OuterEnd()) {
            return false;
        } else {
            UpdateOuterParams();
            return true;
        }
    }

    __aicore__ inline bool IsScaleAKL1FullLoad() const
    {
        return isScaleA1KFullLoad_;
    }

    __aicore__ inline bool IsScaleBKL1FullLoad() const
    {
        return isScaleB1KFullLoad_;
    }

    /**
     * @description: Get current scaleKa outer loop index, used for GetBufferPos in CopyCubeIn
     * @param: void
     * @return: return current scaleKa outerIdx
     */
    __aicore__ inline int32_t GetOuterScaleKaIdx() const
    {
        return BASE_MODULE::outerIdx_ / (BASE_MODULE::kaStepFactor_ * scaleFactorA_);
    }

    /**
     * @description: Get current scaleKb outer loop index, used for GetBufferPos in CopyCubeIn
     * @param: void
     * @return: return current scaleKb outerIdx
     */
    __aicore__ inline int32_t GetOuterScaleKbIdx() const
    {
        return BASE_MODULE::outerIdx_ / (BASE_MODULE::kbStepFactor_ * scaleFactorB_);
    }

    /**
     * @description: Get next scaleKa outer loop index, used for ClearL1BufferCache in SchedulerMDL
     * @param: void
     * @return: return next scaleKa outerIdx
     */
    __aicore__ inline int32_t GetNextOuterScaleKaIdx() const
    {
        return (BASE_MODULE::outerIdx_ + 1) / (BASE_MODULE::kaStepFactor_ * scaleFactorA_);
    }

    /**
     * @description: Get next scaleKb outer loop index, used for ClearL1BufferCache in SchedulerMDL
     * @param: void
     * @return: return next scaleKb outerIdx
     */
    __aicore__ inline int32_t GetNextOuterScaleKbIdx() const
    {
        return (BASE_MODULE::outerIdx_ + 1) / (BASE_MODULE::kbStepFactor_ * scaleFactorB_);
    }

    __aicore__ inline int32_t GetTileShapeScaleA() const
    {
        return tileShapeScaleA_;
    }

    __aicore__ inline int32_t GetTileShapeScaleB() const
    {
        return tileShapeScaleB_;
    }

    __aicore__ inline int32_t GetScaleFactorA() const
    {
        return scaleFactorA_;
    }

    __aicore__ inline int32_t GetScaleFactorB() const
    {
        return scaleFactorB_;
    }

protected:
    __aicore__ inline void UpdateOuterParams()
    {
        BASE_MODULE::UpdateOuterParams();
        auto tilingStepKa = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKa();
        auto tilingStepKb = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKb();
        auto tilingBaseK = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
        // for scaleA/scaleB
        scaleFactorA_ = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetScaleFactorA();
        scaleFactorB_ = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetScaleFactorB();

        int32_t curScaleKaOuterIdx = BASE_MODULE::innerStartIdx_ / (tilingStepKa * scaleFactorA_);
        int32_t curScaleKbOuterIdx = BASE_MODULE::innerStartIdx_ / (tilingStepKb * scaleFactorB_);

        if (curScaleKaOuterIdx + 1 >= Ceil(BASE_MODULE::outerKaIter_, scaleFactorA_)) {
            tileShapeScaleA_ = tailScaleStepKa_;
        } else {
            tileShapeScaleA_ = Ceil(tilingStepKa * tilingBaseK * scaleFactorA_, MX_K_FACTOR);
        }

        if (curScaleKbOuterIdx + 1 >= Ceil(BASE_MODULE::outerKbIter_, scaleFactorB_)) {
            tileShapeScaleB_ = tailScaleStepKb_;
        } else {
            tileShapeScaleB_ = Ceil(tilingStepKb * tilingBaseK * scaleFactorB_, MX_K_FACTOR);
        }
    }

private:
    int32_t tileShapeScaleA_; // scaleKaL1 length
    int32_t tileShapeScaleB_; // scaleKbL1 length
    uint8_t scaleFactorA_;
    uint8_t scaleFactorB_;

    int32_t tailScaleStepKa_; // align scaleKaL1 length for GM->L1
    int32_t tailScaleStepKb_; // align scaleKbL1 length for GM->L1
    bool isScaleA1KFullLoad_, isScaleB1KFullLoad_;
};
} // namespace Detail
} // namespace Impl
} // namespace AscendC
#endif // _K_LOOP_MDL_MX_H_