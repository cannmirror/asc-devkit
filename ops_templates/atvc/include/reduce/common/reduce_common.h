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

#ifndef ATVC_REDUCE_COMMON_H
#define ATVC_REDUCE_COMMON_H

#include "patterns.h"
namespace ATVC {
enum ShapeDim {
    DIM_0,
    DIM_1,
    DIM_2,
    DIM_3,
    DIM_4,
    DIM_5,
    DIM_6,
    DIM_7,
    DIM_8,
    DIM_9,
    DIM_REDUCE,     // Reduce轴
    DIM_BROADCAST   // Broadcast轴
};

namespace AR_PATTERN {
    static constexpr uint32_t A = 100;
    static constexpr uint32_t AR = 11;
    static constexpr uint32_t ARA = 20;
    static constexpr uint32_t ARAR = 31;
    static constexpr uint32_t ARARA = 40;
};

namespace BASIC_CNT {
    static constexpr uint32_t BASIC_128 = 128;
    static constexpr uint32_t BASIC_256 = 256;
    static constexpr uint32_t BASIC_512 = 512;
    static constexpr uint32_t BASIC_1024 = 1024;
    static constexpr uint32_t BASIC_2048 = 2048;
    static constexpr uint32_t BASIC_4096 = 4096;
};

namespace AR_COUNT {
    static constexpr uint32_t A0R1 = 1;
    static constexpr uint32_t A0R2 = 2;
    static constexpr uint32_t A1R0 = 10;
    static constexpr uint32_t A2R0 = 20;
    static constexpr uint32_t A3R0 = 30;
    static constexpr uint32_t A4R0 = 40;
    static constexpr uint32_t A5R0 = 50;
    static constexpr uint32_t A1R2 = 12;
    static constexpr uint32_t A1R3 = 13;
    static constexpr uint32_t A1R4 = 14;
    static constexpr uint32_t A1R5 = 15;
    static constexpr uint32_t A2R3 = 23;
    static constexpr uint32_t A2R4 = 24;
    static constexpr uint32_t A2R5 = 25;
    static constexpr uint32_t A2R6 = 26;
    static constexpr uint32_t A3R4 = 34;
    static constexpr uint32_t A3R5 = 35;
    static constexpr uint32_t A3R6 = 36;
    static constexpr uint32_t A3R7 = 37;
    static constexpr uint32_t A4R5 = 45;
    static constexpr uint32_t A4R6 = 46;
    static constexpr uint32_t A4R7 = 47;
    static constexpr uint32_t A4R8 = 48;
    static constexpr uint32_t A5R6 = 56;
    static constexpr uint32_t A5R7 = 57;
    static constexpr uint32_t A5R8 = 58;
    static constexpr uint32_t A5R9 = 59;
};

struct ReducePolicy {
public:
    int32_t patternID = -1;
    int32_t loopARCount = -1;
    int32_t loopInnerARCount = -1;
    bool operator==(const ReducePolicy& rhs) const
    {
        return this->patternID == rhs.patternID && this->loopARCount == rhs.loopARCount &&\
        this->loopInnerARCount == rhs.loopInnerARCount;
    }
};

static constexpr ReducePolicy REDUCE_POLICY0 { AR_PATTERN::A, AR_COUNT::A1R0, 0 };
static constexpr ReducePolicy REDUCE_POLICY1 { AR_PATTERN::AR, AR_COUNT::A0R1, 10 };
static constexpr ReducePolicy REDUCE_POLICY2 { AR_PATTERN::AR, AR_COUNT::A1R0, 0 };
static constexpr ReducePolicy REDUCE_POLICY3 { AR_PATTERN::AR, AR_COUNT::A1R0, 1 };
static constexpr ReducePolicy REDUCE_POLICY4 { AR_PATTERN::AR, AR_COUNT::A1R2, 0 };
static constexpr ReducePolicy REDUCE_POLICY5 { AR_PATTERN::ARA, AR_COUNT::A0R1, 10 };
static constexpr ReducePolicy REDUCE_POLICY6 { AR_PATTERN::ARA, AR_COUNT::A1R0, 0 };
static constexpr ReducePolicy REDUCE_POLICY7 { AR_PATTERN::ARA, AR_COUNT::A1R0, 1 };
static constexpr ReducePolicy REDUCE_POLICY8 { AR_PATTERN::ARA, AR_COUNT::A1R2, 0 };
static constexpr ReducePolicy REDUCE_POLICY9 { AR_PATTERN::ARA, AR_COUNT::A2R0, 0 };
static constexpr ReducePolicy REDUCE_POLICY10 { AR_PATTERN::ARA, AR_COUNT::A2R0, 1 };
static constexpr ReducePolicy REDUCE_POLICY11 { AR_PATTERN::ARA, AR_COUNT::A2R3, 0 };
static constexpr ReducePolicy REDUCE_POLICY12 { AR_PATTERN::ARAR, AR_COUNT::A0R2, 10 };
static constexpr ReducePolicy REDUCE_POLICY13 { AR_PATTERN::ARAR, AR_COUNT::A1R0, 0 };
static constexpr ReducePolicy REDUCE_POLICY14 { AR_PATTERN::ARAR, AR_COUNT::A1R0, 2 };
static constexpr ReducePolicy REDUCE_POLICY15 { AR_PATTERN::ARAR, AR_COUNT::A2R0, 1 };
static constexpr ReducePolicy REDUCE_POLICY16 { AR_PATTERN::ARAR, AR_COUNT::A2R0, 2 };
static constexpr ReducePolicy REDUCE_POLICY17 { AR_PATTERN::ARAR, AR_COUNT::A2R4, 0 };
static constexpr ReducePolicy REDUCE_POLICY18 { AR_PATTERN::ARARA, AR_COUNT::A1R0, 0 };
static constexpr ReducePolicy REDUCE_POLICY19 { AR_PATTERN::ARARA, AR_COUNT::A2R0, 2 };
static constexpr ReducePolicy REDUCE_POLICY20 { AR_PATTERN::ARARA, AR_COUNT::A3R0, 0 };
static constexpr ReducePolicy REDUCE_POLICY21 { AR_PATTERN::ARARA, AR_COUNT::A3R0, 2 };
static constexpr ReducePolicy REDUCE_POLICY22 { AR_PATTERN::ARARA, AR_COUNT::A2R0, 0 };

struct ReduceTilingData {
    uint64_t factorACntPerCore;
    uint64_t factorATotalCnt;
    uint64_t ubFactorA;
    uint64_t factorRCntPerCore;
    uint64_t factorRTotalCnt;
    uint64_t ubFactorR;
    uint64_t groupR;
    uint64_t outSize;
    uint64_t basicBlock;
    int32_t coreNum;
    float meanVar;
    uint64_t shape[MAX_DIM];
    uint64_t stride[MAX_DIM];
    uint64_t dstStride[MAX_DIM];
};

struct ReduceParam {
    uint64_t workspaceAddr;         // 申请空间在device上的地址
    uint32_t workspaceSize = 0;     // 申请空间大小
    ReduceTilingData tilingData;    // Reduce类算子的tiling数据
    int32_t nBufferNum = 2;         // 每个Queue中的Tensor数量
};

struct ReduceSchLoopInfo {
    int32_t patternID;
    int32_t reduceDichotomy;
    int32_t loopACount;
    int32_t loopAAxis[ATVC::ReducePattern::MAX_LOOP_DIM];
    int32_t loopRCount;
    int32_t loopRAxis[ATVC::MAX_DIM];

    int32_t loopInnerACount;
    int32_t loopInnerAAxis[ATVC::MAX_DIM];
    int32_t loopInnerRCount;
    int32_t loopInnerRAxis[ATVC::MAX_DIM];
    int32_t innerPatternID;
};
};

#endif // ATVC_REDUCE_COMMON_H