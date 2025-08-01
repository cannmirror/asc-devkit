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
 * \file scheduler_mdl_mx_base.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_MATMUL_SCHEDULER_BASE_SCHEDULER_MDL_MX_BASE_H
#define AICORE_ADV_API_DETAIL_MATMUL_SCHEDULER_BASE_SCHEDULER_MDL_MX_BASE_H

#include "scheduler_intf.h"
#include "scheduler_mdl_mx_base.h"

namespace AscendC {
namespace Impl {
namespace Detail {

/*
    MatmulScheduler for A/B/ScaleA/ScaleB is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    MatmulScheduler is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG,
    PolicyType POLICY_TYPE = PolicyType::MATMUL_DEFAULT, typename = void>
class MxMatmulSchedulerBase
    : public MatmulMDLSchedulerCommon<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, POLICY_TYPE> {
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(CopyCubeInScaleA);
    MATMUL_USE_MODULE(CopyCubeInScaleB);
    MATMUL_USE_MODULE(CopyCubeInA);
    MATMUL_USE_MODULE(CopyCubeInB);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);

    using SrcAT = typename A_TYPE::T;
    using SrcBT = typename B_TYPE::T;
    using ScaleT = fp8_e8m0_t;
    using BiasT = typename BIAS_TYPE::T;
    using SrcA2T = typename GetL0DataType<typename A_TYPE::T, true>::Type;
    using SrcB2T = typename GetL0DataType<typename B_TYPE::T, true>::Type;
    using L0cT = typename GetMmDstType<typename A_TYPE::T>::Type;

public:
    using BASE_MODULE =
        AscendC::Impl::Detail::MatmulMDLSchedulerCommon<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, POLICY_TYPE>;

    __aicore__ inline bool MoveNext()
    {
        if (unlikely(BASE_MODULE::isFirstIter_)) {
            return BASE_MODULE::MoveOnFirstIterate();
        } else {
            if constexpr (ToMatmulConfig(MM_CFG).iterateOrder == IterateOrder::UNDEF) {
                if (likely(MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetIterateOrder()
                           == static_cast<int>(IterateOrder::ORDER_M))) {
                    return MoveOnIterateOrderM();
                } else {
                    ASCENDC_ASSERT((MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetIterateOrder()
                                       == static_cast<int>(IterateOrder::ORDER_N)),
                        {
                            KERNEL_LOG(KERNEL_ERROR, "iterateOrder is %d , which should be ORDER_N",
                                MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetIterateOrder());
                        });
                    return MoveOnIterateOrderN();
                }
            } else if constexpr (ToMatmulConfig(MM_CFG).iterateOrder == IterateOrder::ORDER_N) {
                return MoveOnIterateOrderN();
            } else {
                return MoveOnIterateOrderM();
            }
        }
    }

    __aicore__ inline bool MoveOnIterateOrderNInner()
    {
        if (!MATMUL_MODULE(NLoop)->OuterNext()) {
            ClearLoadDataA();
            MATMUL_MODULE(NLoop)->OuterStart();
            MATMUL_MODULE(NLoop)->InnerStart();
            if (!MATMUL_MODULE(MLoop)->OuterNext()) {
                return false;
            }
            MATMUL_MODULE(MLoop)->InnerStart();
        }
        return true;
    }

    __aicore__ inline bool MoveOnIterateOrderN()
    {
        // when M inner loop is finished, and restart M inner loop
        if (!MATMUL_MODULE(MLoop)->InnerNext()) {
            MATMUL_MODULE(MLoop)->InnerStart();
            // when N inner loop is finished, clear right matrix's data in L1 buffer
            if (!MATMUL_MODULE(NLoop)->InnerNext()) {
                ClearLoadDataB();
                if (!MoveOnIterateOrderNInner()) {
                    return false;
                }
            }
        }
        return true;
    }

    __aicore__ inline bool MoveOnIterateOrderMInner()
    {
        if (!MATMUL_MODULE(MLoop)->OuterNext()) {
            ClearLoadDataB();
            MATMUL_MODULE(MLoop)->OuterStart();
            MATMUL_MODULE(MLoop)->InnerStart();
            if (!MATMUL_MODULE(NLoop)->OuterNext()) {
                return false;
            }
            MATMUL_MODULE(NLoop)->InnerStart();
        }
        return true;
    }

    __aicore__ inline bool MoveOnIterateOrderM()
    {
        // when N inner loop is finished, and restart N inner loop
        if (!MATMUL_MODULE(NLoop)->InnerNext()) {
            MATMUL_MODULE(NLoop)->InnerStart();
            // when M inner loop is finished, clear left matrix's data in L1 buffer
            if (!MATMUL_MODULE(MLoop)->InnerNext()) {
                ClearLoadDataA();
                // when M outer loop is finished, clear right matrix's data in L1 buffer,
                // and restart M outer and inner loop
                if (!MoveOnIterateOrderMInner()) {
                    return false;
                }
            }
        }
        return true;
    }

    __aicore__ inline void ClearLoadDataA()
    {
        if constexpr (!PhyPosIsL1(A_TYPE::pos)) {
            if (MATMUL_MODULE(KLoop)->IsAKL1FullLoad()) {
                MATMUL_MODULE(CopyCubeInA)->ClearLoadData();
            }
        }
        if constexpr (!PhyMxScalePosIsL1<A_TYPE>()) {
            if (MATMUL_MODULE(KLoop)->IsScaleAKL1FullLoad()) {
                MATMUL_MODULE(CopyCubeInScaleA)->ClearLoadData();
            }
        }
    }

    __aicore__ inline void ClearLoadDataB()
    {
        if constexpr (!PhyPosIsL1(B_TYPE::pos)) {
            if (MATMUL_MODULE(KLoop)->IsBKL1FullLoad()) {
                MATMUL_MODULE(CopyCubeInB)->ClearLoadData();
            }
        }
        if constexpr (!PhyMxScalePosIsL1<B_TYPE>()) {
            if (MATMUL_MODULE(KLoop)->IsScaleBKL1FullLoad()) {
                MATMUL_MODULE(CopyCubeInScaleB)->ClearLoadData();
            }
        }
    }

    __aicore__ inline int32_t GetBL1OffsetFromGM()
    {
        int32_t tilingBaseN = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
        int32_t stepN = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepN();
        if (!MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB()) {
            if (tilingBaseN % BASE_MODULE::c0Size_ == 0 || stepN == 1) {
                return (MATMUL_MODULE(NLoop)->GetInnerIdx() - MATMUL_MODULE(NLoop)->GetOuterIdx() * stepN)
                       * tilingBaseN;
            } else {
                return (MATMUL_MODULE(NLoop)->GetInnerIdx() - MATMUL_MODULE(NLoop)->GetOuterIdx() * stepN)
                       * CeilAlign(tilingBaseN, BASE_MODULE::c0Size_);
            }
        }
        return (MATMUL_MODULE(NLoop)->GetInnerIdx() - MATMUL_MODULE(NLoop)->GetOuterIdx() * stepN) * tilingBaseN;
    }
};
} // namespace Detail
} // namespace Impl
} // namespace AscendC
#endif
