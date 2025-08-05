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

/*!
 * \file broadcast_tiling.h
 * \brief tiling for broadcast
 */

#ifndef ATVC_BROADCAST_TILING_H
#define ATVC_BROADCAST_TILING_H
#include <cstring>
#include <algorithm>
#include <vector>
#include <string>
#include "securec.h"
#include "graph/types.h"
#include "common/const_def.h"
#include "common/compile_info.h"
#include "common/ops_utils_host.h"


namespace ATVC {
struct BroadcastTilingInputParam {
    std::vector<int64_t> shapeIn;
    std::vector<int64_t> shapeOut;
    ge::DataType inputDtype = ge::DataType::DT_UNDEFINED;
};
}

namespace OpTiling {
constexpr static int32_t BRC_BASIC_NUM = 4;     // broadcast输入输出内存基本块分配个数
class BroadcastOpTiling {
public:
    BroadcastOpTiling(ATVC::BroadcastTilingInputParam& inputParam,
        ATVC::BroadcastPolicy* policy, ATVC::BroadcastParam* param)
        : opInput_(inputParam), param_(param), policy_(policy)
    {
        compileInfo_ = ATVC::GetOpCompileInfo();
    }

    bool Run()
    {
        if (!IsAxesValid(opInput_.shapeIn, opInput_.shapeOut)) {
            printf("[ERROR]Shape checkout failed!\n");
            return false;
        }
        std::vector<uint64_t> newShapeIn;
        std::vector<uint64_t> newShapeOut;
        if (!EliminateOne(opInput_.shapeIn, opInput_.shapeOut, newShapeIn, newShapeOut)) {
            printf("[ERROR]Failed to  eliminate shape!\n");
            return false;
        }

        if (!DoTiling(newShapeIn, newShapeOut)) {
            printf("[ERROR]Failed to Calculate Tiling param!\n");
            return false;
        }
        CalcWorkSpace();
        return true;
    }

private:
    template <class Pattern>
    void ComputeStride(std::vector<uint64_t>& shapeIn, std::vector<uint64_t>& shapeOut)
    {
        // shape
        if (shapeIn[0] == 1 && shapeOut[ATVC::DIM0] == 1) {
            for (size_t i = 1; i < shapeIn.size(); i ++) {
                param_->tilingData.shape[i-1] = shapeIn[i];
                param_->tilingData.dstShape[i-1] = shapeOut[i];
            }
        } else {
            for (size_t i = 0; i < shapeIn.size(); i ++) {
                param_->tilingData.shape[i] = shapeIn[i];
                param_->tilingData.dstShape[i] = shapeOut[i];
            }
        }
        
        // stride
        uint64_t dimA = param_->tilingData.A2 * param_->tilingData.A11 * param_->tilingData.A12;
        uint64_t dimB = param_->tilingData.B2 * param_->tilingData.B1;
        if (Pattern::TailA) {
            param_->tilingData.stride[ATVC::DIM0] = dimA;
            param_->tilingData.stride[ATVC::DIM1] = dimA;
            param_->tilingData.dstStride[ATVC::DIM0] = dimB * dimA;
            param_->tilingData.dstStride[ATVC::DIM1] = dimB;
        } else {
            param_->tilingData.stride[ATVC::DIM0] = dimA * 1;
            param_->tilingData.stride[ATVC::DIM1] = dimA;
            param_->tilingData.stride[ATVC::DIM2] = 1;
            param_->tilingData.dstStride[ATVC::DIM0] = dimA * dimB * 1;
            param_->tilingData.dstStride[ATVC::DIM1] = dimB;
            param_->tilingData.dstStride[ATVC::DIM2] = 1;
        }
    }

    template <class Pattern>
    void SetTilingKey()
    {
        uint32_t innerID = Pattern::TailA ? 0 : 1;
        policy_->patternID = Pattern::ID * ATVC::CONST10 + innerID;
        policy_->loopABCount = 1 * ATVC::CONST10 + 0;
        policy_->loopInnerABCount = 0 * ATVC::CONST10 + 1;
    }

    void CalcWorkSpace()
    {
        param_->workspaceSize = ATVC::WORKSPACE_SIZE;
    }

    bool IsAxesValid(const std::vector<int64_t>& shapeIn, const std::vector<int64_t>& shapeOut)
    {
        size_t sizeIn = shapeIn.size();
        size_t sizeOut = shapeOut.size();
        if (sizeOut != sizeIn) {
            printf("input dim in is not equel to output dim! \n");
            return false;
        };

        for (size_t i = 0; i < sizeIn; i++) {
            if (shapeOut[i] != shapeIn[i] && shapeIn[i] != 1) {
                printf("Input shape in broadcast dim should be 1\n");
                return false;
            } else if (shapeIn[i] <= 0) {
                printf("Input and output shape should be more than 0\n");
                return false; 
            }
        }
        return true;
    }

    bool EliminateOne(std::vector<int64_t> &oriShapeIn,
                      std::vector<int64_t> &oriShapeOut,
                      std::vector<uint64_t> &shapeIn,
                      std::vector<uint64_t> &shapeOut)
    {
        bool isCurB = false;
        bool haveA = false;
        bool haveB = false;

        for (size_t i = 0; i < oriShapeIn.size(); i++) {
            if (oriShapeIn[i] == 1 && oriShapeOut[i] != oriShapeIn[i]) { // B轴
                if (!isCurB && haveB) {
                    printf("[ERROR]Only support AB/BA!\n");
                    return false;
                }
                if (!haveB) {
                    shapeIn.emplace_back(oriShapeIn[i]);
                    shapeOut.emplace_back(oriShapeOut[i]);
                } else { // 连续B轴
                    shapeIn.back() = 1;
                    shapeOut.back() *= oriShapeOut[i];
                }
                isCurB = true;
                haveB = true;
            } else { // A轴
                if (isCurB && haveA) {
                    printf("[ERROR]Only support AB/BA!\n");
                    return false;
                }
                if (!haveA) {
                    shapeIn.emplace_back(oriShapeIn[i]);
                    shapeOut.emplace_back(oriShapeOut[i]);
                } else { // 连续A轴
                    shapeIn.back() *= oriShapeIn[i];
                    shapeOut.back() *= oriShapeOut[i];
                }
                isCurB = false;
                haveA = true;
            }
        }
        if (shapeIn.size() !=2U && shapeOut.size() != 2U) {
            printf("[ERROR] Shape after eliminate is not 2 dim!\n");
            return false;
        }
        if (shapeIn[0] != shapeOut[0]) {
            shapeIn.emplace(shapeIn.begin(), 1);
            shapeOut.emplace(shapeOut.begin(), 1);
        }
        return true;
    }

    bool DoTiling(std::vector<uint64_t>& shapeIn, std::vector<uint64_t>& shapeOut)
    {
        int32_t shapeSize = shapeIn.size();
        for (int32_t i = 0; i < shapeSize; i++) {
            printf("DoTiling shapeSize[%d]: shape[%d] %lu\n", shapeSize, i, shapeIn[i]);
        }
        switch (shapeSize) {
            case ATVC::CONST1:
                return ComputeTiling<ATVC::BroadcastPattern::A>(shapeIn, shapeOut);
            case ATVC::CONST2:
                return ComputeTiling<ATVC::BroadcastPattern::AB>(shapeIn, shapeOut);
            case ATVC::CONST3:
                return ComputeTiling<ATVC::BroadcastPattern::ABA>(shapeIn, shapeOut);
            default:
                printf("[ERROR] Compute tiling error because of invalid input shape size[%d]\n", shapeSize);
                return false;
        }
        return false;
    }

    template <class Pattern>
    bool ComputeTiling(std::vector<uint64_t>& shapeIn, std::vector<uint64_t>& shapeOut)
    {
        if (!CalcSplitParam<Pattern>(shapeOut)) {
            printf("[ERROR] Calculate tiling param failed!\n");
            return false;
        }
        ComputeStride<Pattern>(shapeIn, shapeOut);
        SetTilingKey<Pattern>();
        return true;
    }

    uint64_t CalcBasicBlock()
    {
        uint64_t basicBlock = OpsUtils::FloorAlign(compileInfo_.ubSize / BRC_BASIC_NUM, ATVC::UB_ALIGN_32);
        if (basicBlock > ATVC::BLOCK_SIZE_64K) {
            basicBlock = ATVC::BLOCK_SIZE_64K;
        } else if (basicBlock > ATVC::BLOCK_SIZE_48K) {
            basicBlock = ATVC::BLOCK_SIZE_48K;
        } else if (basicBlock > ATVC::BLOCK_SIZE_32K) {
            basicBlock = ATVC::BLOCK_SIZE_32K;
        }
        return basicBlock;
    }

    void ExpandTilingParam(uint64_t basicBlock)
    {
        ATVC::BroadcastOpTilingData& tilingData = param_->tilingData;
        tilingData.coreNum = tilingData.A0 * tilingData.B0;
        tilingData.basicBlock = basicBlock;
        tilingData.factorACntPerCore = tilingData.A11 * tilingData.A12* tilingData.A2;
        tilingData.factorATotalCnt = tilingData.A0;
        tilingData.factorBCntPerCore = tilingData.B1 * tilingData.B2;
        tilingData.factorBTotalCnt = tilingData.B0;
    }

    bool CheckTilingParam(uint32_t dimA, uint32_t dimB)
    {
        ATVC::BroadcastOpTilingData& tilingData = param_->tilingData;
        uint64_t dSize = ge::GetSizeByDataType(opInput_.inputDtype);
        if (dSize == 0) {
            printf("[ERROR] Data size is invalid, please check input data type!\n");
            return false;
        }

        if (tilingData.coreNum > compileInfo_.vectorCoreNum) {
            printf("[ERROR] Check tiling failed, coreNum(%u) > vector Real Core count(%lu)\n",
                tilingData.coreNum, compileInfo_.vectorCoreNum);
            return false;
        }
        if (tilingData.A2 * tilingData.A12 * tilingData.A11 * tilingData.A0 < dimA) {
            printf("[ERROR] Check tiling failed, A2 * A12 * A11 * A0 < dimA(%u)\n", dimA);
            return false;
        }
        if (tilingData.B2 * tilingData.B1 * tilingData.B0 < dimB) {
            printf("[ERROR] Check tiling failed, B2 * B1 * B0 < dimB(%u)\n", dimB);
            return false;
        }
        if (tilingData.B2 * dSize % ATVC::UB_ALIGN_32 != 0) {
            printf("[ERROR] Check tiling failed, B2(%lu) is not aligined with 32B\n", tilingData.B2);
            return false;
        }
        if (tilingData.A2 * dSize % ATVC::UB_ALIGN_32 != 0) {
            printf("[ERROR] Check tiling failed, A2(%lu) is not aligined with 32B\n", tilingData.A2);
            return false;
        }
        return true;
    }

    template <class Pattern>
    bool CalcSplitParam(const std::vector<uint64_t>& shape)
    {
        /*
            BASIC_BLOCK = UB_TOTAL / 4  根据UB总大小动态分配,输入2份输出2份
            A2 B2 32B对齐
            A2 * B2 * sizeof(T) <= BASIC_BLOCK
            A2 * A12 * size(T) <= BASIC_BLOCK
            AB场景：B2尽量大
            BA场景：A2尽量大
        */
        uint64_t basicBlock = CalcBasicBlock();
        uint64_t dSize = ge::GetSizeByDataType(opInput_.inputDtype);
        if (dSize == 0) {
            printf("[ERROR] Data size is invalid, please check input data type!\n");
            return false;
        }
        uint64_t dUint = ATVC::UB_ALIGN_32 / dSize;
        uint64_t cacheSize = OpsUtils::FloorDiv(basicBlock, dSize);
        uint32_t dimA = Pattern::TailA ? Pattern::Dim - 1 :  Pattern::Dim - 2; // A
        uint32_t dimB = Pattern::TailA ? Pattern::Dim - 2 :  Pattern::Dim - 1; // B
        uint64_t i = OpsUtils::FloorAlign(shape[dimA], dUint); // 32B对齐
        uint64_t j = OpsUtils::FloorAlign(shape[dimB], dUint); // 32B对齐
        ATVC::BroadcastOpTilingData& tilingData = param_->tilingData;

        if (Pattern::TailA) {// 优先A轴打满
            tilingData.B2 = dUint; // B2最小值
            tilingData.A2 = i > OpsUtils::FloorDiv(cacheSize, dUint) ? OpsUtils::FloorDiv(cacheSize, dUint) : i;
            tilingData.B2 = OpsUtils::FloorAlign(OpsUtils::FloorDiv(cacheSize, tilingData.A2), dUint);
            if (tilingData.B2 > j) {
                tilingData.B2 = j;
            }
        } else { // 优先B轴打满
            tilingData.A2 = dUint; // A2最小值
            tilingData.B2 = j > OpsUtils::FloorDiv(cacheSize, dUint) ? OpsUtils::FloorDiv(cacheSize, dUint) : j;
            tilingData.A2 = OpsUtils::FloorAlign(OpsUtils::FloorDiv(cacheSize, tilingData.B2), dUint);
            if (tilingData.A2 > i) {
                tilingData.A2 = i;
            }
        }

        // 1.优先多核 A0 B0打满核后再计算核内循环
        tilingData.A0 =  OpsUtils::CeilDiv(shape[dimA], tilingData.A2);
        tilingData.B0 =  OpsUtils::CeilDiv(shape[dimB], tilingData.B2);
        // A0*B0为实际的block num 必须小于vectorCoreNum
        while (tilingData.A0 * tilingData.B0 > compileInfo_.vectorCoreNum) {
            if (tilingData.B0 > 1) { // 优先A0切轴
                --tilingData.B0;
            } else {
                --tilingData.A0;
            }
        }

        // 2.核内循环优先A12,因为A12只需要copyIn 1次
        tilingData.A12 = OpsUtils::CeilDiv(shape[dimA], tilingData.A2 * tilingData.A0);
        if (tilingData.A12 * tilingData.A2 > cacheSize) {
            tilingData.A12 = OpsUtils::FloorDiv(cacheSize , tilingData.A2);
        }
        tilingData.A11 = OpsUtils::CeilDiv(shape[dimA], (tilingData.A0 * tilingData.A2 * tilingData.A12)); // 计算精确A11
        tilingData.B1= OpsUtils::CeilDiv(shape[dimB], (tilingData.B0 * tilingData.B2));

        // 3.最后重新计算A0  B0避免空核
        tilingData.A0 =  OpsUtils::CeilDiv(shape[dimA], tilingData.A2 * tilingData.A11 * tilingData.A12);
        tilingData.B0 =  OpsUtils::CeilDiv(shape[dimB], tilingData.B2 * tilingData.B1);

        // 4.写Tiling结果
        ExpandTilingParam(basicBlock);
        return CheckTilingParam(shape[dimA], shape[dimB]);
    }

private:
    ATVC::BroadcastTilingInputParam opInput_;
    ATVC::BroadcastParam* param_ {nullptr};
    ATVC::BroadcastPolicy* policy_ {nullptr};
    ATVC::OpCompileInfo compileInfo_;
};
}  // namespace OpTiling
#endif // ATVC_BROADCAST_TILING_H