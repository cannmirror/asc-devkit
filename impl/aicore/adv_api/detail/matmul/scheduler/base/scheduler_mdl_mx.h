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
 * \file scheduler_mdl_mx.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_MATMUL_SCHEDULER_BASE_SCHEDULER_MDL_MX_H
#define AICORE_ADV_API_DETAIL_MATMUL_SCHEDULER_BASE_SCHEDULER_MDL_MX_H

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
    PolicyType POLICY_TYPE>
class MatmulScheduler<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, POLICY_TYPE,
    enable_if_t<DoMatmulMDL(MM_CFG) && isMxMatmul<A_TYPE, B_TYPE>>>
    : public MxMatmulSchedulerBase<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, POLICY_TYPE> {
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(CopyCubeInA);
    MATMUL_USE_MODULE(CopyCubeInB);
    MATMUL_USE_MODULE(CopyCubeInScaleA);
    MATMUL_USE_MODULE(CopyCubeInScaleB);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE(LoadToA2);
    MATMUL_USE_MODULE(LoadToB2);
    MATMUL_USE_MODULE(TBufPoolL0);
    MATMUL_USE_MODULE(BiasScheduler);
    MATMUL_USE_MODULE(CubeOutBuffer);
    MATMUL_USE_MODULE(CopyCubeOut);
    MATMUL_USE_MODULE(Context);
    MATMUL_USE_MODULE(MatmulUnitFlag);
    MATMUL_USE_MODULE(MmadCompute);
    MATMUL_USE_MODULE(MatmulCrossCoreSync);

    using SrcAT = typename A_TYPE::T;
    using SrcBT = typename B_TYPE::T;
    using ScaleT = fp8_e8m0_t;
    using BiasT = typename BIAS_TYPE::T;
    using SrcA2T = typename GetL0DataType<typename A_TYPE::T, true>::Type;
    using SrcB2T = typename GetL0DataType<typename B_TYPE::T, true>::Type;
    using L0cT = typename GetMmDstType<typename A_TYPE::T>::Type;

public:
    using BASE_MODULE =
        AscendC::Impl::Detail::MxMatmulSchedulerBase<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, POLICY_TYPE>;
    __aicore__ inline void Init(const TCubeTiling* __restrict cubeTiling, TPipe* tpipe)
    {
        BASE_MODULE::Init(cubeTiling, tpipe);
        MATMUL_MODULE(CopyCubeInScaleA)->Init();
        MATMUL_MODULE(CopyCubeInScaleB)->Init();
        InitShareBufEnd(tpipe);
    }

    __aicore__ inline bool ScheduleOnce(bool enPartialSum)
    {
        MATMUL_MODULE(BiasScheduler)->SetBias(!enPartialSum && MATMUL_MODULE(BiasScheduler)->IsBias());
        if (!BASE_MODULE::MoveNext()) {
            return false;
        }
        if (!enPartialSum) {
            MATMUL_MODULE(CubeOutBuffer)->AllocTensor();
        }
        ReduceK(enPartialSum);
        return true;
    }

private:
    __aicore__ inline MxSplitParams InitSplitAParams()
    {
        MxSplitParams aL0Params;
        const auto& tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        int32_t tilingBaseM = tiling.GetBaseM();
        if constexpr (PhyPosIsL1OrUB<MM_CFG>(A_TYPE::pos)) {
            aL0Params.axisL1Offset = MATMUL_MODULE(MLoop)->GetInnerIdx() * tilingBaseM;
            // ds && 82 mdl support multi singleshape in l1
            if constexpr (IsFullStaticTiling(MM_CFG) || MatmulFeatureTrait<MM_CFG>::IsSupportUBToL1Singleshape()) {
                aL0Params.axisL1Len = MATMUL_MODULE(MatmulShapeInfo)->GetOrgM() != -1 ?
                                          MATMUL_MODULE(MatmulShapeInfo)->GetOrgM() :
                                          MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreM();
                aL0Params.kAxisL1Len = MATMUL_MODULE(MatmulShapeInfo)->GetOrgKa() != -1 ?
                                           MATMUL_MODULE(MatmulShapeInfo)->GetOrgKa() :
                                           MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK();
            } else {
                aL0Params.kAxisL1Len = MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK();
                aL0Params.axisL1Len = MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreM();
            }
        } else if constexpr (IsStaticPaddingEnable(MM_CFG)) {
            aL0Params.axisL1Len = tiling.GetStepM() * tilingBaseM;
            aL0Params.kAxisL1Len = tiling.GetStepKa() * tiling.GetBaseK();
            aL0Params.axisL1Offset =
                (MATMUL_MODULE(MLoop)->GetInnerIdx() - MATMUL_MODULE(MLoop)->GetOuterIdx() * tiling.GetStepM())
                * tilingBaseM;
        } else {
            aL0Params.axisL1Len = MATMUL_MODULE(MLoop)->GetTileBlockShape() * BLOCK_CUBE;
            aL0Params.axisL1Offset =
                (MATMUL_MODULE(MLoop)->GetInnerIdx() - MATMUL_MODULE(MLoop)->GetOuterIdx() * tiling.GetStepM())
                * tilingBaseM;
        }

        if constexpr (PhyPosIsL1OrUB<MM_CFG>(A_TYPE::scalePosition)) {
            aL0Params.auxMatrixL1Offset = MATMUL_MODULE(MLoop)->GetInnerIdx() * tilingBaseM;
            aL0Params.kAuxMatrixL1Len = Ceil(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK(), MX_K_FACTOR);
        } else {
            aL0Params.auxMatrixL1Offset =
                (MATMUL_MODULE(MLoop)->GetInnerIdx() - MATMUL_MODULE(MLoop)->GetOuterIdx() * tiling.GetStepM())
                * tilingBaseM;
        }
        if constexpr (IsStaticPaddingEnable(MM_CFG)) {
            aL0Params.axisL0Len = tilingBaseM;
        } else {
            aL0Params.axisL0Len = MATMUL_MODULE(MLoop)->GetBaseShape();
        }
        if constexpr ((A_TYPE::format == CubeFormat::VECTOR) || (A_TYPE::format == CubeFormat::SCALAR)) {
            aL0Params.axisL0Len = 1;
        } else {
            if (aL0Params.axisL0Len == 1) {
                aL0Params.axisL0Len = BLOCK_CUBE;
            }
        }
        return aL0Params;
    }

    __aicore__ inline MxSplitParams InitSplitBParams()
    {
        MxSplitParams bL0Params;
        int32_t tilingBaseN = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
        bL0Params.axisL0Len = MATMUL_MODULE(NLoop)->GetBaseShape();
        if constexpr (IsStaticPaddingEnable(MM_CFG)) {
            bL0Params.axisL1Len = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepN() * tilingBaseN;
            bL0Params.kAxisL1Len = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKb()
                                   * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
            bL0Params.axisL0Len = tilingBaseN;
            bL0Params.axisL1Offset = BASE_MODULE::GetBL1OffsetFromGM();
        } else {
            if constexpr (PhyPosIsL1OrUB<MM_CFG>(B_TYPE::pos)) {
                bL0Params.axisL1Offset = MATMUL_MODULE(NLoop)->GetInnerIdx() * tilingBaseN;
                // ds && 82 mdl support multi singleshape in l1
                if constexpr (IsFullStaticTiling(MM_CFG) || MatmulFeatureTrait<MM_CFG>::IsSupportUBToL1Singleshape()) {
                    bL0Params.axisL1Len = MATMUL_MODULE(MatmulShapeInfo)->GetOrgN() != -1 ?
                                              MATMUL_MODULE(MatmulShapeInfo)->GetOrgN() :
                                              MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreN();
                    bL0Params.kAxisL1Len = MATMUL_MODULE(MatmulShapeInfo)->GetOrgKb() != -1 ?
                                               MATMUL_MODULE(MatmulShapeInfo)->GetOrgKb() :
                                               MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK();
                } else {
                    bL0Params.kAxisL1Len = MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK();
                    bL0Params.axisL1Len = MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreN();
                }
            } else {
                bL0Params.axisL1Len = MATMUL_MODULE(NLoop)->GetTileBlockShape() * BLOCK_CUBE;
                bL0Params.axisL1Offset = BASE_MODULE::GetBL1OffsetFromGM();
            }
        }

        if constexpr (PhyPosIsL1OrUB<MM_CFG>(B_TYPE::scalePosition)) {
            bL0Params.auxMatrixL1Offset = MATMUL_MODULE(NLoop)->GetInnerIdx() * tilingBaseN;
            bL0Params.kAuxMatrixL1Len = Ceil(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK(), MX_K_FACTOR);
        } else {
            bL0Params.auxMatrixL1Offset = BASE_MODULE::GetBL1OffsetFromGM();
        }
        return bL0Params;
    }

    __aicore__ inline void SplitPrepare(const bool enPartialSum, const bool isATranspose, const bool isBTranspose,
        MxSplitParams& aL0Params, MxSplitParams& bL0Params, bool& sL0CInit, bool& sL0CLast)
    {
        UpdateSplitParams(aL0Params, bL0Params);
        BASE_MODULE::UpdateComputeParams(enPartialSum, sL0CInit, sL0CLast);
        if (isATranspose) {
            aL0Params.axisL1Len = CeilAlign(MATMUL_MODULE(MLoop)->GetTileShape(), BASE_MODULE::c0Size_);
            aL0Params.axisL0Len = CeilAlign(MATMUL_MODULE(MLoop)->GetBaseShape(), BASE_MODULE::c0Size_);
            if constexpr (PhyPosIsL1OrUB<MM_CFG>(A_TYPE::pos)) {
                aL0Params.kAxisL1Len = CeilAlign(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK(), MX_BASEK_FACTOR);
            } else {
                aL0Params.kAxisL1Len = CeilAlign(MATMUL_MODULE(KLoop)->GetTileShapeA(), MX_BASEK_FACTOR);
            }
            if constexpr (IsScaleTransWithInlv<A_TYPE>) {
                aL0Params.kAuxMatrixL1Len =
                    Ceil(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK(), MX_BASEK_FACTOR) * MX_EVEN_FACTOR;
            }
        }
        if (!isBTranspose) {
            bL0Params.axisL1Len = CeilAlign(MATMUL_MODULE(NLoop)->GetTileShape(), BASE_MODULE::c0Size_);
            bL0Params.axisL0Len = CeilAlign(MATMUL_MODULE(NLoop)->GetBaseShape(), BASE_MODULE::c0Size_);
            if constexpr (PhyPosIsL1OrUB<MM_CFG>(B_TYPE::pos)) {
                bL0Params.kAxisL1Len = CeilAlign(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK(), MX_BASEK_FACTOR);
            } else {
                bL0Params.kAxisL1Len = CeilAlign(MATMUL_MODULE(KLoop)->GetTileShapeB(), MX_BASEK_FACTOR);
            }
            if constexpr (IsScaleTransWithInlv<B_TYPE>) {
                bL0Params.kAuxMatrixL1Len =
                    Ceil(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK(), MX_BASEK_FACTOR) * MX_EVEN_FACTOR;
            }
        }
        MATMUL_MODULE(LoadToA2)->Prepare(isATranspose, aL0Params.kAxisL1Len, aL0Params.axisL1Len);
        MATMUL_MODULE(LoadToB2)->Prepare(isBTranspose, bL0Params.kAxisL1Len);
    }

    __aicore__ inline void UpdateSplitParams(MxSplitParams& aL0Params, MxSplitParams& bL0Params)
    {
        int32_t kInnerIdx = MATMUL_MODULE(KLoop)->GetInnerStartIdx();
        int32_t tilingBaseK = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
        if constexpr (PhyPosIsL1OrUB<MM_CFG>(A_TYPE::pos)) {
            aL0Params.kAxisL1Offset = kInnerIdx * tilingBaseK;
        } else {
            // Mx scene K direction should be aligned to 64, baseK is already guaranteed to be aligned to 64
            aL0Params.kAxisL1Len = CeilAlign(MATMUL_MODULE(KLoop)->GetTileShapeA(), MX_BASEK_FACTOR);
            int32_t tilingStepKa = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKa();
            aL0Params.kAxisL1Offset = (kInnerIdx - kInnerIdx / tilingStepKa * tilingStepKa) * tilingBaseK;
        }

        if constexpr (PhyPosIsL1OrUB<MM_CFG>(A_TYPE::scalePosition)) {
            if constexpr (PhyPosIsL1(A_TYPE::scalePosition)) {
                aL0Params.kAuxMatrixL1Len =
                    Ceil(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK() * MATMUL_MODULE(KLoop)->GetScaleFactorA(),
                        MX_K_FACTOR);
            }
            aL0Params.kAuxMatrixL1Offset = Ceil(kInnerIdx * tilingBaseK, MX_K_FACTOR);
        } else {
            int32_t tilingStepKa = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKa();
            aL0Params.kAuxMatrixL1Len = MATMUL_MODULE(KLoop)->GetTileShapeScaleA();
            aL0Params.kAuxMatrixL1Offset = (kInnerIdx
                                               - kInnerIdx / (tilingStepKa * MATMUL_MODULE(KLoop)->GetScaleFactorA())
                                                     * (tilingStepKa * MATMUL_MODULE(KLoop)->GetScaleFactorA()))
                                           * Ceil(tilingBaseK, MX_K_FACTOR);
        }

        if constexpr (PhyPosIsL1OrUB<MM_CFG>(B_TYPE::pos)) {
            bL0Params.kAxisL1Offset = kInnerIdx * tilingBaseK;
        } else {
            // Mx scene K direction should be aligned to 64, baseK is already guaranteed to be aligned to 64
            bL0Params.kAxisL1Len = CeilAlign(MATMUL_MODULE(KLoop)->GetTileShapeB(), MX_BASEK_FACTOR);
            int32_t tilingStepKb = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKb();
            bL0Params.kAxisL1Offset = (kInnerIdx - kInnerIdx / tilingStepKb * tilingStepKb) * tilingBaseK;
        }

        if constexpr (PhyPosIsL1OrUB<MM_CFG>(B_TYPE::scalePosition)) {
            if constexpr (PhyPosIsL1(B_TYPE::scalePosition)) {
                bL0Params.kAuxMatrixL1Len =
                    Ceil(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK() * MATMUL_MODULE(KLoop)->GetScaleFactorB(),
                        MX_K_FACTOR);
            }
            bL0Params.kAuxMatrixL1Offset = Ceil(kInnerIdx * tilingBaseK, MX_K_FACTOR);
        } else {
            int32_t tilingStepKb = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKb();
            bL0Params.kAuxMatrixL1Len = MATMUL_MODULE(KLoop)->GetTileShapeScaleB();
            bL0Params.kAuxMatrixL1Offset = (kInnerIdx
                                               - kInnerIdx / (tilingStepKb * MATMUL_MODULE(KLoop)->GetScaleFactorB())
                                                     * (tilingStepKb * MATMUL_MODULE(KLoop)->GetScaleFactorB()))
                                           * Ceil(tilingBaseK, MX_K_FACTOR);
        }
    }

    __aicore__ inline void CopyIn(
        LocalTensor<SrcAT>& a1, LocalTensor<SrcBT>& b1, LocalTensor<ScaleT>& scaleA1, LocalTensor<ScaleT>& scaleB1)
    {
        a1 = MATMUL_MODULE(CopyCubeInA)
                 ->LoadData(MATMUL_MODULE(MLoop)->GetInnerIdx(), MATMUL_MODULE(KLoop)->GetInnerStartIdx(),
                     MATMUL_MODULE(MLoop)->GetTileShape(), MATMUL_MODULE(KLoop)->GetTileShapeA());
        b1 = MATMUL_MODULE(CopyCubeInB)
                 ->LoadData(MATMUL_MODULE(KLoop)->GetInnerStartIdx(), MATMUL_MODULE(NLoop)->GetInnerIdx(),
                     MATMUL_MODULE(KLoop)->GetTileShapeB(), MATMUL_MODULE(NLoop)->GetTileShape());
        scaleA1 = MATMUL_MODULE(CopyCubeInScaleA)
                      ->LoadData(MATMUL_MODULE(MLoop)->GetInnerIdx(), MATMUL_MODULE(KLoop)->GetInnerStartIdx(),
                          MATMUL_MODULE(MLoop)->GetTileShape(), MATMUL_MODULE(KLoop)->GetTileShapeScaleA());
        scaleB1 = MATMUL_MODULE(CopyCubeInScaleB)
                      ->LoadData(MATMUL_MODULE(KLoop)->GetInnerStartIdx(), MATMUL_MODULE(NLoop)->GetInnerIdx(),
                          MATMUL_MODULE(KLoop)->GetTileShapeScaleB(), MATMUL_MODULE(NLoop)->GetTileShape());
        if constexpr (MatmulFeatureTrait<MM_CFG>::IsSupportUBToL1Singleshape()) {
            MATMUL_MODULE(MatmulCrossCoreSync)->WaitL1Ready();
        }
    }

    __aicore__ inline void ReduceKOneIter(bool enPartialSum)
    {
        // init split params for left and right matrix
        MxSplitParams aL0Params = InitSplitAParams();
        MxSplitParams bL0Params = InitSplitBParams();
        MATMUL_MODULE(KLoop)->OuterStart();
        // CopyIn
        LocalTensor<SrcAT> a1;
        LocalTensor<SrcBT> b1;
        LocalTensor<ScaleT> scaleA1;
        LocalTensor<ScaleT> scaleB1;
        CopyIn(a1, b1, scaleA1, scaleB1);
        bool isATranspose = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA();
        bool isBTranspose = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB();
        bool isScaleATranspose = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeScaleA();
        bool isScaleBTranspose = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeScaleB();
        SplitPrepareOneIter(isATranspose, isBTranspose, aL0Params, bL0Params);
        LocalTensor<BiasT> bias = BASE_MODULE::SplitBias(bL0Params.axisL0Len);
        int32_t kL1Stride;
        int32_t kAuxL1Stride;
        if constexpr (IsStaticPaddingEnable(MM_CFG)) {
            kL1Stride = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
        } else {
            kL1Stride = MATMUL_MODULE(KLoop)->GetBaseBlockShape() * BASE_MODULE::c0Size_;
        }
        kAuxL1Stride = Ceil(kL1Stride, MX_K_FACTOR);
        // start k inner loop
        MATMUL_MODULE(KLoop)->InnerStart();
        do {
            ComputeKDB(a1, b1, scaleA1, scaleB1, aL0Params, bL0Params, isATranspose, isBTranspose, !enPartialSum, true);
            aL0Params.kAxisL1Offset += kL1Stride;
            bL0Params.kAxisL1Offset += kL1Stride;
            aL0Params.kAuxMatrixL1Offset += kAuxL1Stride;
            bL0Params.kAuxMatrixL1Offset += kAuxL1Stride;
        } while (MATMUL_MODULE(KLoop)->InnerNext());
        if constexpr (MatmulFeatureTrait<MM_CFG>().IsSupportMNL0DB()) {
            MATMUL_MODULE(BiasScheduler)->Free(bias);
        }
    }

    __aicore__ inline void ReduceK(bool enPartialSum)
    {
        if (BASE_MODULE::IsMDLKFullLoad()) {
            ReduceKOneIter(enPartialSum);
        } else {
            ReduceKMultiIter(enPartialSum);
        }
    }

    __aicore__ inline void ReduceKMultiIter(bool enPartialSum)
    {
        // init split params for left and right matrix
        MxSplitParams aL0Params = InitSplitAParams();
        MxSplitParams bL0Params = InitSplitBParams();
        bool isATranspose = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA();
        bool isBTranspose = MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB();
        // start K outer loop
        MATMUL_MODULE(KLoop)->OuterStart();
        // curKaOuterIdx and curKbOuterIdx are used to decide if left or right matrix need to clear its l1 buffer
        int32_t curKaOuterIdx = MATMUL_MODULE(KLoop)->GetOuterKaIdx();
        int32_t curKbOuterIdx = MATMUL_MODULE(KLoop)->GetOuterKbIdx();
        int32_t curScaleKaOuterIdx = MATMUL_MODULE(KLoop)->GetOuterScaleKaIdx();
        int32_t curScaleKbOuterIdx = MATMUL_MODULE(KLoop)->GetOuterScaleKbIdx();
        do {
            // CopyIn
            LocalTensor<SrcAT> a1;
            LocalTensor<SrcBT> b1;
            LocalTensor<ScaleT> scaleA1;
            LocalTensor<ScaleT> scaleB1;
            CopyIn(a1, b1, scaleA1, scaleB1);
            LocalTensor<BiasT> bias = BASE_MODULE::SplitBias(bL0Params.axisL0Len);
            Compute(a1, b1, scaleA1, scaleB1, bias, enPartialSum, isATranspose, isBTranspose, aL0Params, bL0Params);
            if constexpr (MatmulFeatureTrait<MM_CFG>().IsSupportMNL0DB()) {
                MATMUL_MODULE(BiasScheduler)->Free(bias);
            }
            ClearL1BufferCache(curKaOuterIdx, curKbOuterIdx, curScaleKaOuterIdx, curScaleKbOuterIdx);
        } while (MATMUL_MODULE(KLoop)->OuterNext());
        ResetCopyInBuffer();
    }

    __aicore__ inline void ClearL1BufferCache(
        int32_t& curKaOuterIdx, int32_t& curKbOuterIdx, int32_t& curScaleKaOuterIdx, int32_t& curScaleKbOuterIdx)
    {
        if constexpr (!PhyPosIsL1OrUB<MM_CFG>(A_TYPE::pos)) {
            int32_t curKaIdx = MATMUL_MODULE(KLoop)->GetNextOuterKaIdx();
            // if next outerKaIdx is not equal to curKaOuterIdx, clear left matrix's data in L1 buffer
            if (curKaIdx != curKaOuterIdx && !MATMUL_MODULE(KLoop)->IsAKL1FullLoad()) {
                MATMUL_MODULE(CopyCubeInA)->ClearLoadData();
                curKaOuterIdx = curKaIdx;
            }
        }

        if constexpr (!PhyPosIsL1OrUB<MM_CFG>(A_TYPE::scalePosition)) {
            int32_t curKaIdx = MATMUL_MODULE(KLoop)->GetNextOuterScaleKaIdx();
            // if next outerKaIdx is not equal to curScaleKaOuterIdx, clear leftScale matrix's data in L1 buffer
            if (curKaIdx != curScaleKaOuterIdx && !MATMUL_MODULE(KLoop)->IsScaleAKL1FullLoad()) {
                MATMUL_MODULE(CopyCubeInScaleA)->ClearLoadData();
                curScaleKaOuterIdx = curKaIdx;
            }
        }

        if constexpr (!PhyPosIsL1OrUB<MM_CFG>(B_TYPE::pos)) {
            // if next outerKbIdx is not equal to curKbOuterIdx, clear right matrix's data in L1 buffer
            int32_t curKbIdx = MATMUL_MODULE(KLoop)->GetNextOuterKbIdx();
            if (curKbIdx != curKbOuterIdx && !MATMUL_MODULE(KLoop)->IsBKL1FullLoad()) {
                MATMUL_MODULE(CopyCubeInB)->ClearLoadData();
                curKbOuterIdx = curKbIdx;
            }
        }

        if constexpr (!PhyPosIsL1OrUB<MM_CFG>(B_TYPE::scalePosition)) {
            int32_t curKbIdx = MATMUL_MODULE(KLoop)->GetNextOuterScaleKbIdx();
            // if next outerKaIdx is not equal to curScaleKbOuterIdx, clear leftScale matrix's data in L1 buffer
            if (curKbIdx != curScaleKbOuterIdx && !MATMUL_MODULE(KLoop)->IsScaleBKL1FullLoad()) {
                MATMUL_MODULE(CopyCubeInScaleB)->ClearLoadData();
                curScaleKbOuterIdx = curKbIdx;
            }
        }
    }

    __aicore__ inline LocalTensor<SrcA2T> SplitA(const LocalTensor<SrcAT>& a1, LocalTensor<ScaleT>& scaleA1,
        const MxSplitParams& aL0Params, const bool isATranspose)
    {
        auto posA = MATMUL_MODULE(MLoop)->GetInnerIdx() * MATMUL_MODULE(KLoop)->GetTotalIter()
                    + MATMUL_MODULE(KLoop)->GetInnerIdx();
        int32_t kL0Len = CeilAlign(MATMUL_MODULE(KLoop)->GetBaseShape(), MX_BASEK_FACTOR);
        if (!(MATMUL_MODULE(TBufPoolL0)->template Hit<TPosition::A2>(posA))) {
            LocalTensor<SrcA2T> a2 = MATMUL_MODULE(TBufPoolL0)->template GetBuffer<TPosition::A2, SrcA2T>();
            MATMUL_MODULE(LoadToA2)->Load(a2, a1, aL0Params.axisL1Len, aL0Params.kAxisL1Len, aL0Params.axisL0Len,
                kL0Len, aL0Params.axisL1Offset, aL0Params.kAxisL1Offset, isATranspose, scaleA1,
                aL0Params.kAuxMatrixL1Len, aL0Params.kAuxMatrixL1Offset, aL0Params.auxMatrixL1Offset);
            return a2;
        } else {
            return MATMUL_MODULE(TBufPoolL0)->template GetBuffer<TPosition::A2, SrcA2T>();
        }
    }

    __aicore__ inline LocalTensor<SrcB2T> SplitB(const LocalTensor<SrcBT>& b1, LocalTensor<ScaleT>& scaleB1,
        const MxSplitParams& bL0Params, const bool isBTranspose)
    {
        auto posB = MATMUL_MODULE(NLoop)->GetInnerIdx() * MATMUL_MODULE(KLoop)->GetTotalIter()
                    + MATMUL_MODULE(KLoop)->GetInnerIdx();
        int32_t kL0Len = CeilAlign(MATMUL_MODULE(KLoop)->GetBaseShape(), MX_BASEK_FACTOR);
        if (!(MATMUL_MODULE(TBufPoolL0)->template Hit<TPosition::B2>(posB))) {
            LocalTensor<SrcB2T> b2 = MATMUL_MODULE(TBufPoolL0)->template GetBuffer<TPosition::B2, SrcB2T>();
            MATMUL_MODULE(LoadToB2)->Load(b2, b1, bL0Params.axisL1Len, bL0Params.kAxisL1Len, bL0Params.axisL0Len,
                kL0Len, bL0Params.axisL1Offset, bL0Params.kAxisL1Offset, isBTranspose, scaleB1,
                bL0Params.kAuxMatrixL1Len, bL0Params.kAuxMatrixL1Offset, bL0Params.auxMatrixL1Offset);
            return b2;
        } else {
            return MATMUL_MODULE(TBufPoolL0)->template GetBuffer<TPosition::B2, SrcB2T>();
        }
    }

    __aicore__ inline void ResetCopyInBuffer()
    {
        // clear L1 buffers
        if constexpr (!PhyPosIsL1(A_TYPE::pos)) {
            if (!MATMUL_MODULE(KLoop)->IsAKL1FullLoad()) {
                MATMUL_MODULE(CopyCubeInA)->Reset();
            }
        }
        if constexpr (!PhyPosIsL1(A_TYPE::scalePosition)) {
            if (!MATMUL_MODULE(KLoop)->IsScaleAKL1FullLoad()) {
                MATMUL_MODULE(CopyCubeInScaleA)->Reset();
            }
        }
        if constexpr (!PhyPosIsL1(B_TYPE::pos)) {
            if (!MATMUL_MODULE(KLoop)->IsBKL1FullLoad()) {
                MATMUL_MODULE(CopyCubeInB)->Reset();
            }
        }
        if constexpr (!PhyPosIsL1(B_TYPE::scalePosition)) {
            if (!MATMUL_MODULE(KLoop)->IsScaleBKL1FullLoad()) {
                MATMUL_MODULE(CopyCubeInB)->Reset();
            }
        }
    }

    __aicore__ inline void CubeCompute(const LocalTensor<L0cT>& cMatrix, const LocalTensor<SrcA2T>& a2,
        const LocalTensor<SrcB2T>& b2, const uint16_t madM, const uint16_t madN, const uint16_t madK,
        const bool isATranspose, const bool isBTranspose, const bool sL0CInit, const bool sL0CLast)
    {
        int32_t kInnerStartIdx;
        if (BASE_MODULE::IsMDLKFullLoad()) {
            kInnerStartIdx = 0;
        } else {
            kInnerStartIdx = MATMUL_MODULE(KLoop)->GetInnerStartIdx();
        }
        auto unitFlag = MATMUL_MODULE(MatmulUnitFlag)
                            ->GetUnitFlag(sL0CLast
                                          && (MATMUL_MODULE(KLoop)->GetInnerIdx()
                                              == kInnerStartIdx + MATMUL_MODULE(KLoop)->GetInnerIter() - 1));
        if constexpr (EnUnitFlag(MM_CFG) && IsMxEnableUnitFlag<A_TYPE, B_TYPE>) {
            unitFlag = 0;
        }
        bool cmatrixSource;
        bool cmatrixInitVal;
        BASE_MODULE::UpdateMmadComputeParams(kInnerStartIdx, sL0CInit, cmatrixSource, cmatrixInitVal);
        MATMUL_MODULE(MmadCompute)
            ->Compute(
                cMatrix, a2, b2, madM, madK, madN, isATranspose, isBTranspose, unitFlag, cmatrixSource, cmatrixInitVal);
    }

    __aicore__ inline void Compute(const LocalTensor<SrcAT>& a1, const LocalTensor<SrcBT>& b1,
        LocalTensor<ScaleT>& scaleA1, LocalTensor<ScaleT>& scaleB1, LocalTensor<BiasT>& bias, const bool enPartialSum,
        const bool isATranspose, const bool isBTranspose, MxSplitParams& aL0Params, MxSplitParams& bL0Params)
    {
        bool sL0CInit = false;
        bool sL0CLast = false;
        SplitPrepare(enPartialSum, isATranspose, isBTranspose, aL0Params, bL0Params, sL0CInit, sL0CLast);
        int32_t kL1Stride = MATMUL_MODULE(KLoop)->GetBaseBlockShape() * BASE_MODULE::c0Size_;
        int32_t kAuxL1Stride = Ceil(kL1Stride, MX_K_FACTOR);
        MATMUL_MODULE(KLoop)->InnerStart();
        do {
            ComputeKDB(a1, b1, scaleA1, scaleB1, aL0Params, bL0Params, isATranspose, isBTranspose, sL0CInit, sL0CLast);
            aL0Params.kAxisL1Offset += kL1Stride;
            bL0Params.kAxisL1Offset += kL1Stride;
            aL0Params.kAuxMatrixL1Offset += kAuxL1Stride;
            bL0Params.kAuxMatrixL1Offset += kAuxL1Stride;
        } while (MATMUL_MODULE(KLoop)->InnerNext());
    }

    __aicore__ inline void ComputeKDB(const LocalTensor<SrcAT>& a1, const LocalTensor<SrcBT>& b1,
        LocalTensor<ScaleT>& scaleA1, LocalTensor<ScaleT>& scaleB1, const MxSplitParams& aL0Params,
        const MxSplitParams& bL0Params, const bool isATranspose, const bool isBTranspose, const bool sL0CInit,
        const bool sL0CLast)
    {
        MATMUL_MODULE(TBufPoolL0)->Allocate();
        LocalTensor<SrcA2T> a2 = SplitA(a1, scaleA1, aL0Params, isATranspose);
        LocalTensor<SrcB2T> b2 = SplitB(b1, scaleB1, bL0Params, isBTranspose);
        MATMUL_MODULE(TBufPoolL0)->EnQue();
        MATMUL_MODULE(TBufPoolL0)->DeQue();
        CubeCompute(MATMUL_MODULE(CubeOutBuffer)->GetTensor(), a2, b2, aL0Params.axisL0Len, bL0Params.axisL0Len,
            CeilAlign(MATMUL_MODULE(KLoop)->GetBaseShape(), MX_BASEK_FACTOR), isATranspose, isBTranspose, sL0CInit,
            sL0CLast);
        MATMUL_MODULE(TBufPoolL0)->Free();
        MATMUL_MODULE(BiasScheduler)->Free();
    }

    __aicore__ inline void SplitPrepareOneIter(
        const bool isATranspose, const bool isBTranspose, MxSplitParams& aL0Params, MxSplitParams& bL0Params)
    {
        aL0Params.kAxisL1Offset = 0;
        bL0Params.kAxisL1Offset = 0;
        aL0Params.kAuxMatrixL1Offset = 0;
        bL0Params.kAuxMatrixL1Offset = 0;

        if constexpr (PhyPosIsL1OrUB<MM_CFG>((A_TYPE::pos)) && IsStaticPaddingEnable(MM_CFG)) {
            aL0Params.kAxisL1Len = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK();
        } else if constexpr (PhyPosIsL1OrUB<MM_CFG>((A_TYPE::pos))
                             && (IsFullStaticTiling(MM_CFG)
                                 || MatmulFeatureTrait<MM_CFG>::IsSupportUBToL1Singleshape())) {
            aL0Params.kAxisL1Len = MATMUL_MODULE(MatmulShapeInfo)->GetOrgKa() != -1 ?
                                       MATMUL_MODULE(MatmulShapeInfo)->GetOrgKa() :
                                       MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK();
        } else {
            aL0Params.kAxisL1Len = MATMUL_MODULE(KLoop)->GetTileBlockShapeA() * BASE_MODULE::c0Size_;
        }

        if constexpr (PhyPosIsL1OrUB<MM_CFG>((A_TYPE::scalePosition)) && IsStaticPaddingEnable(MM_CFG)) {
            aL0Params.kAuxMatrixL1Len =
                Ceil(MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK(), MX_K_FACTOR);
        } else if constexpr (PhyPosIsL1OrUB<MM_CFG>((A_TYPE::scalePosition))
                             && (IsFullStaticTiling(MM_CFG)
                                 || MatmulFeatureTrait<MM_CFG>::IsSupportUBToL1Singleshape())) {
            aL0Params.kAuxMatrixL1Len = Ceil(MATMUL_MODULE(MatmulShapeInfo)->GetOrgKa() != -1 ?
                                                 MATMUL_MODULE(MatmulShapeInfo)->GetOrgKa() :
                                                 MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK(),
                MX_K_FACTOR);
        } else if constexpr (IsStaticPaddingEnable(MM_CFG)) {
            aL0Params.kAuxMatrixL1Len =
                Ceil(MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK(), MX_K_FACTOR);
        } else {
            aL0Params.kAuxMatrixL1Len = MATMUL_MODULE(KLoop)->GetTileShapeScaleA();
        }

        if constexpr (PhyPosIsL1OrUB<MM_CFG>((B_TYPE::pos)) && IsStaticPaddingEnable(MM_CFG)) {
            bL0Params.kAxisL1Len = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK();
        } else if constexpr (PhyPosIsL1OrUB<MM_CFG>((B_TYPE::pos))
                             && (IsFullStaticTiling(MM_CFG)
                                 || MatmulFeatureTrait<MM_CFG>::IsSupportUBToL1Singleshape())) {
            bL0Params.kAxisL1Len = MATMUL_MODULE(MatmulShapeInfo)->GetOrgKb() != -1 ?
                                       MATMUL_MODULE(MatmulShapeInfo)->GetOrgKb() :
                                       MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK();
        } else {
            bL0Params.kAxisL1Len = MATMUL_MODULE(KLoop)->GetTileBlockShapeB() * BASE_MODULE::c0Size_;
        }

        if constexpr (PhyPosIsL1OrUB<MM_CFG>((B_TYPE::scalePosition)) && IsStaticPaddingEnable(MM_CFG)) {
            bL0Params.kAuxMatrixL1Len =
                Ceil(MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK(), MX_K_FACTOR);
        } else if constexpr (PhyPosIsL1OrUB<MM_CFG>((B_TYPE::scalePosition))
                             && (IsFullStaticTiling(MM_CFG)
                                 || MatmulFeatureTrait<MM_CFG>::IsSupportUBToL1Singleshape())) {
            bL0Params.kAuxMatrixL1Len = Ceil(MATMUL_MODULE(MatmulShapeInfo)->GetOrgKb() != -1 ?
                                                 MATMUL_MODULE(MatmulShapeInfo)->GetOrgKb() :
                                                 MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK(),
                MX_K_FACTOR);
        } else {
            bL0Params.kAuxMatrixL1Len = MATMUL_MODULE(KLoop)->GetTileShapeScaleB();
        }
    }
};
} // namespace Detail
} // namespace Impl
} // namespace AscendC
#endif
