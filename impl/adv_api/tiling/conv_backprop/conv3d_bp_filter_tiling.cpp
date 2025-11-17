/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file conv3d_bp_filter_tiling.cpp
 * \brief
 */
#include "../../detail/host_log.h"
#include "conv3d_bp_tiling_util.h"
#include "include/adv_api/conv_backprop/conv3d_bp_filter_tiling.h"

namespace {
    constexpr int32_t BLOCK_CUBE = 16;
    constexpr uint64_t BLOCK_CUBE_U64 = 16;
} // namespace


namespace ConvBackpropApi {

int64_t Conv3dBpFilterTiling::GetTiling(optiling::Conv3DBackpropFilterTilingData& tiling)
{
    int64_t ret = Compute();
    if (ret == -1) {
        TILING_LOG_ERROR("can not gen conv3d api tiling");
        return -1;
    }

    SetFinalTiling(tiling);
    PrintTilingData();
    return ret;
}

int64_t Conv3dBpFilterTiling::Compute()
{
    if (!CheckInputParam()) {
        return -1;
    }

    Conv3dBpFilterTilingBase::ShapeInitCalc();
    SetBasicBlockAttrsTiling();

    InitTilingValue();
    uint64_t kernelHW = static_cast<uint64_t>(shapeInfo.orgkH) * shapeInfo.orgkW;
    uint64_t strideHW = static_cast<uint64_t>(attrInfo.strideH) * attrInfo.strideW;
    // MTE1压力较大且N方向满足搬运对齐，切MN
    if (strideHW > 1 && (kernelHW % BLOCK_CUBE == 0 || BLOCK_CUBE % kernelHW == 0)) {
        MultiCoreSplitMN();
    } else {
        MultiCoreSplitK();
    }
    return 0;
}

void Conv3dBpFilterTiling::SetBasicBlockAttrsTiling()
{
    uint64_t fractalSize0 = shapeCalc.channelSize;
    mmInfo_.mValue = shapeCalc.cout1_g * fractalSize0;
    mmInfo_.nValue = shapeInfo.orgkH * shapeInfo.orgkW *
        shapeCalc.cin1_g * fractalSize0;
    // 不拆dk时将dk合并到n轴
    if (!seperateDk_) {
        mmInfo_.nValue *= shapeInfo.orgkD;
    }
    mmInfo_.kValue = CeilAlign(static_cast<uint64_t>(shapeInfo.orgHo) * shapeInfo.orgWo, fractalSize0);
    blockTiling_.usedCoreNum = coreNum_;
}

void Conv3dBpFilterTiling::InitTilingValue()
{
    // 默认开启double buffer
    tilingParams.al0Pbuffer = DB_ON;
    tilingParams.bl0Pbuffer = DB_ON;
    tilingParams.singleCoreBatch = 1;
    tilingParams.singleCoreGroup = 1;
    tilingParams.singleCoreDk = 1;
}

/*
 *
 * 基本块
 *
*/
void Conv3dBpFilterTiling::MultiCoreSplitMN()
{
    blockTiling_.iterateOrder = mmInfo_.mValue > mmInfo_.nValue ? 1 : 0;
    blockTiling_.coreBindDirection = SPLIT_M_N;
    InitBaseMNK();

    // 默认策略，一侧pingpong全载后驻留，另一侧pingpong交替载入
    blockTiling_.coreBindOrder = ROW_FIRST;
    blockTiling_.dbL1A = DB_ON;
    blockTiling_.dbL1B = DB_ON;

    // L1配比算法，按照16个块往下进行对称阶梯衰减
    uint32_t depthA1 = L1_DEPTH_16;
    uint32_t depthB1 = L1_DEPTH_16;
    while (depthA1 >= 1 && depthB1 >= 1) {
        blockTiling_.depthA1 = depthA1;
        blockTiling_.depthB1 = depthB1;
        blockTiling_.stepM = 1;
        blockTiling_.stepN = 1;
        UpdateStepMNK();
        if (!IsCurBlockL1L0Invalid()) {
            break;
        }
        depthA1 = depthA1 > STEP_2 ? (depthA1 - STEP_2) : (depthA1 - 1);
        depthB1 = depthB1 > STEP_2 ? (depthB1 - STEP_2) : (depthB1 - 1);
    }

    // 合法性兜底，防止w一次要搬运的过大，直接超L1
    if (IsCurBlockL1L0Invalid()) {
        ShrinkBaseBlock();
        UpdateStepMNK();
    }

    UpdateSingleCoreInfo();
    if (blockTiling_.totalCnt < blockTiling_.usedCoreNum) {
        MultiCoreSplitK();
    }
}

void Conv3dBpFilterTiling::MultiCoreSplitK()
{
    blockTiling_.iterateOrder = mmInfo_.mValue > mmInfo_.nValue ? 1 : 0;
    blockTiling_.coreBindDirection = mmInfo_.mValue > mmInfo_.nValue ? SPLIT_N_K : SPLIT_M_K;
    InitBaseMNK();

    // 默认24算法，depthA1/B1都为4*1*DB_ON, L1占用128*128*2*16=512KB
    // 默认策略，一侧pingpong全载后驻留，另一侧pingpong交替载入
    blockTiling_.coreBindOrder = ROW_FIRST;
    blockTiling_.dbL1A = DB_ON;
    blockTiling_.dbL1B = DB_ON;

    // L1配比算法，按照8+8,8+4,4+4,4+2,2+2,2+1,1+1进行阶梯匹配
    uint32_t depthA1 = L1_DEPTH_8;
    uint32_t depthB1 = L1_DEPTH_8;
    while (depthA1 >= 1 && depthB1 >= 1) {
        blockTiling_.depthA1 = depthA1;
        blockTiling_.depthB1 = depthB1;
        blockTiling_.stepM = 1;
        blockTiling_.stepN = 1;
        UpdateStepMNK();
        UpdateSingleCoreInfo();
        if (blockTiling_.totalCnt >= blockTiling_.usedCoreNum && !IsCurBlockL1L0Invalid()) {
            break;
        }
        // 大的一侧L1先减半，相等时先减少非驻留侧的L1占用
        if (depthA1 == 1 && depthB1 == 1) {
            break;
        } else if (depthA1 > depthB1) {
            depthA1 /= NUM_HALF;
        } else if (depthB1 > depthA1) {
            depthB1 /= NUM_HALF;
        } else if (blockTiling_.coreBindDirection == SPLIT_M_K) {
            depthB1 /= NUM_HALF;
        } else {
            depthA1 /= NUM_HALF;
        }
    }

    // 合法性兜底，防止w一次要搬运的过大，直接超L1
    if (IsCurBlockL1L0Invalid()) {
        ShrinkBaseBlock();
        UpdateStepMNK();
        UpdateSingleCoreInfo();
    }
}

void Conv3dBpFilterTiling::InitBaseMNK()
{
    if (blockTiling_.coreBindDirection == SPLIT_M_N) {
        // 不切K算法主要是MTE1 Bound场景，L0A搬运效率是L0B两倍以上，优先让L0A填满
        if (mmInfo_.mValue > BASIC_BLOCK_SIZE_128) {
            blockTiling_.blockBaseM = BASIC_BLOCK_SIZE_256;
            blockTiling_.blockBaseN = BASIC_BLOCK_SIZE_128;
        } else {
            blockTiling_.blockBaseM = BASIC_BLOCK_SIZE_128;
            blockTiling_.blockBaseN = BASIC_BLOCK_SIZE_256;
        }
        blockTiling_.blockBaseK = BASIC_BLOCK_SIZE_64;
        blockTiling_.dbL0C = DB_OFF;
    } else {
        // 切K算法, 默认128基本块保证L0C能开PingPong, 否则会断流
        blockTiling_.blockBaseM = BASIC_BLOCK_SIZE_128;
        blockTiling_.blockBaseN = BASIC_BLOCK_SIZE_128;
        blockTiling_.blockBaseK = BASIC_BLOCK_SIZE_128;
        blockTiling_.dbL0C = DB_ON;
    }

    if (dtypeByte_ == g_dtypeSizeTab.at(ConvCommonApi::ConvDtype::FLOAT32)) {
        blockTiling_.blockBaseK = blockTiling_.blockBaseK / NUM_HALF;
    }

    uint64_t fractalSize0 = shapeCalc.channelSize;
    uint64_t aL0Max = blockTiling_.blockBaseK * blockTiling_.blockBaseM;
    uint64_t bL0Max = blockTiling_.blockBaseK * blockTiling_.blockBaseN;

    // M或N方向不够一个基本块，适应性调小BaseM或者StepM
    blockTiling_.blockBaseM = std::min(static_cast<uint64_t>(blockTiling_.blockBaseM), CeilAlign(mmInfo_.mValue, BLOCK_CUBE_U64));
    blockTiling_.blockBaseN = std::min(static_cast<uint64_t>(blockTiling_.blockBaseN), CeilAlign(mmInfo_.nValue, BLOCK_CUBE_U64));

    // K方向不够一个基本块，适应性调小BaseK，否则根据BaseM和BaseN的情况调大BaseK并进行搬运对齐
    uint64_t alignedKValue = CeilAlign(mmInfo_.kValue, fractalSize0);
    if (alignedKValue < static_cast<uint64_t>(blockTiling_.blockBaseK)) {
        blockTiling_.blockBaseK = alignedKValue;
    } else {
        if (blockTiling_.blockBaseM == 0 || blockTiling_.blockBaseN == 0) {
            return;
        }
        // 根据调小后的BaseM和BaseN调大BaseK
        uint64_t newBaseKa = std::max(aL0Max / blockTiling_.blockBaseM / fractalSize0,
            static_cast<uint64_t>(1)) * fractalSize0;
        uint64_t newBaseKb = std::max(bL0Max / blockTiling_.blockBaseN / BLOCK_CUBE,
            static_cast<uint64_t>(1)) * BLOCK_CUBE;
        uint64_t newBaseK = std::min(std::min(newBaseKa, newBaseKb), alignedKValue);
        blockTiling_.blockBaseK = newBaseK;

        // K在不超过L0约束情况下，优先满足搬运对齐
        if (static_cast<uint64_t>(shapeInfo.orgWo) < newBaseK && shapeInfo.orgWo % fractalSize0 == 0) {
            blockTiling_.blockBaseK = newBaseK / shapeInfo.orgWo * shapeInfo.orgWo;
        }
    }
}

void Conv3dBpFilterTiling::UpdateStepMNK()
{
    if (blockTiling_.depthA1 < L1_DEPTH_2) {
        blockTiling_.dbL1A = DB_OFF;
    }
    if (blockTiling_.depthB1 < L1_DEPTH_2) {
        blockTiling_.dbL1B = DB_OFF;
    }

    uint64_t aL1Max = blockTiling_.depthA1 / blockTiling_.dbL1A;
    uint64_t bL1Max = blockTiling_.depthB1 / blockTiling_.dbL1B;

    uint64_t maxMIter = CeilDiv(mmInfo_.mValue, static_cast<uint64_t>(blockTiling_.blockBaseM));
    uint64_t maxNIter = CeilDiv(mmInfo_.nValue, static_cast<uint64_t>(blockTiling_.blockBaseN));
    uint64_t maxKIter = CeilDiv(mmInfo_.kValue, static_cast<uint64_t>(blockTiling_.blockBaseK));
    uint64_t minIter = 1;

    // 根据预置的StepM/StepN初始化StepKa和StepKb, 不超过K方向最大循环次数
    blockTiling_.stepKa = std::max(std::min(aL1Max / blockTiling_.stepM, maxKIter), minIter);
    blockTiling_.stepKb = std::max(std::min(bL1Max / blockTiling_.stepN, maxKIter), minIter);

    // 驻留的一侧允许适应性调整非K方向载入量，不超过最大循环次数
    if (blockTiling_.coreBindDirection == SPLIT_M_K) {
        blockTiling_.stepM = std::max(std::min(aL1Max / blockTiling_.stepKa, maxMIter), minIter);
    } else if (blockTiling_.coreBindDirection == SPLIT_N_K) {
        blockTiling_.stepN = std::max(std::min(bL1Max / blockTiling_.stepKb, maxNIter), minIter);
    }

    // 根据调整后的StepM和StepN调整StepKa和StepKb, 不超过K方向最大循环次数
    blockTiling_.stepKa = std::max(std::min(aL1Max / blockTiling_.stepM, maxKIter), minIter);
    blockTiling_.stepKb = std::max(std::min(bL1Max / blockTiling_.stepN, maxKIter), minIter);

    if (blockTiling_.coreBindDirection == SPLIT_M_K) {
        blockTiling_.stepKa = std::max(blockTiling_.stepKa / blockTiling_.stepKb, static_cast<int64_t>(1)) * blockTiling_.stepKb;
    } else if (blockTiling_.coreBindDirection == SPLIT_N_K) {
        blockTiling_.stepKb = std::max(blockTiling_.stepKb / blockTiling_.stepKa, static_cast<int64_t>(1)) * blockTiling_.stepKa;
    } else {
        if (blockTiling_.stepKa > blockTiling_.stepKb) {
            blockTiling_.stepKa = std::max(blockTiling_.stepKa / blockTiling_.stepKb, static_cast<int64_t>(1)) * blockTiling_.stepKb;
        }
        if (blockTiling_.stepKb > blockTiling_.stepKa) {
            blockTiling_.stepKb = std::max(blockTiling_.stepKb / blockTiling_.stepKa, static_cast<int64_t>(1)) * blockTiling_.stepKa;
        }
    }
}

bool Conv3dBpFilterTiling::IsCurBlockL1L0Invalid()
{
    // 先判断L0合法性，由于LoadData mExtention=Bask会强制16对齐，所以需要对齐到16再判断大小是否会超限制
    uint64_t alignedBaseK = CeilAlign(static_cast<uint64_t>(blockTiling_.blockBaseK), BLOCK_CUBE_U64);
    uint64_t al0LoadSize = alignedBaseK * blockTiling_.blockBaseM * dtypeByte_ * DB_ON;
    if (al0LoadSize > platformInfo.l0ASize) {
        return true;
    }
    uint64_t bl0LoadSize = alignedBaseK * blockTiling_.blockBaseN * dtypeByte_ * DB_ON;
    if (bl0LoadSize > platformInfo.l0BSize) {
        return true;
    }

    // 再判断L1合法性
    uint64_t al1LoadSize = blockTiling_.stepKa * blockTiling_.blockBaseK * blockTiling_.stepM *
                           blockTiling_.blockBaseM * dtypeByte_ * blockTiling_.dbL1A;
    uint64_t bl1LoadSize = CalBL1Bound() * dtypeByte_ * blockTiling_.dbL1B;
    if (al1LoadSize + bl1LoadSize > platformInfo.l1Size) {
        return true;
    }
    return false;
}

void Conv3dBpFilterTiling::UpdateSingleCoreInfo()
{
    // 搬运对齐时默认向下取整，避免越过基本块运算导致重新触发L1载入
    blockTiling_.singleCoreM = blockTiling_.stepM * blockTiling_.blockBaseM;

    uint64_t maxStepKL1 = std::max(blockTiling_.stepKa, blockTiling_.stepKb) * blockTiling_.blockBaseK;
    blockTiling_.singleCoreK = std::max(maxStepKL1 / shapeInfo.orgWo, static_cast<uint64_t>(1)) * shapeInfo.orgWo;
    uint64_t orgKernelHW =  static_cast<uint64_t>(shapeInfo.orgkH * shapeInfo.orgkW * shapeCalc.channelSize);
    if (orgKernelHW == 0) {
        return;
    }
    uint64_t l1Cin1 = std::max(blockTiling_.stepN * blockTiling_.blockBaseN / orgKernelHW,
        static_cast<uint64_t>(1));
    blockTiling_.singleCoreN = l1Cin1 * shapeInfo.orgkH * shapeInfo.orgkW * shapeCalc.channelSize;

    if (blockTiling_.coreBindDirection == SPLIT_M_K) {
        blockTiling_.singleCoreN = mmInfo_.nValue;
    } else if (blockTiling_.coreBindDirection == SPLIT_N_K) {
        blockTiling_.singleCoreM = mmInfo_.mValue;
    } else {
        blockTiling_.singleCoreK = mmInfo_.kValue;
    }

    uint64_t mCnt = CeilDiv(mmInfo_.mValue, static_cast<uint64_t>(blockTiling_.singleCoreM));
    uint64_t kCnt = CeilDiv(mmInfo_.kValue, static_cast<uint64_t>(blockTiling_.singleCoreK));
    uint64_t nCnt = CeilDiv(mmInfo_.nValue, static_cast<uint64_t>(blockTiling_.singleCoreN));
    blockTiling_.totalCnt = static_cast<uint64_t>(shapeInfo.orgN) * shapeInfo.orgDo * mCnt * kCnt * nCnt;
    if (seperateDk_) {
        blockTiling_.totalCnt *= shapeInfo.orgkD;
    }
}

uint64_t Conv3dBpFilterTiling::CalBL1Bound() const
{
    int64_t hoCal = 0;
    int64_t kBl1Size = blockTiling_.blockBaseK * blockTiling_.stepKb;
    if (kBl1Size == 0) {
        return 0;
    }
    if (kBl1Size % shapeInfo.orgWo == 0 || shapeInfo.orgWo % kBl1Size == 0) {
        hoCal = CeilDiv(kBl1Size, shapeInfo.orgWo);
    } else if (kBl1Size > shapeInfo.orgWo) {
        hoCal = kBl1Size / shapeInfo.orgWo + NUM_HALF;
    } else {
        hoCal = NUM_HALF;
    }
    int32_t hiCal = (hoCal - 1) * attrInfo.strideH + (shapeInfo.orgkH - 1) * attrInfo.dilationH + 1;
    uint32_t kernelHW = static_cast<uint32_t>(shapeInfo.orgkH * shapeInfo.orgkW);
    uint32_t bL1N = CeilDiv(blockTiling_.stepN * blockTiling_.blockBaseN, static_cast<int64_t>(BLOCK_CUBE));
    uint32_t bL1Cin1CopyLen = CeilDiv(bL1N, kernelHW); // 向上取整，拖尾时默认多搬一行
    if (bL1N == 0) {
        return 0;
    }
    if (kernelHW > bL1N && kernelHW % bL1N != 0) {
        ++bL1Cin1CopyLen; // 此时bL1Cin1CopyLen为1, 每个基本块不足一行，考虑拖尾最多搬两行
    } else if (NUM_HALF * bL1N % kernelHW != 0) {
        ++bL1Cin1CopyLen; // 除了尾块是0.5，其他场景都要搬2行
    }
    uint64_t bL1Size = static_cast<uint64_t>(hiCal) * shapeInfo.orgWi * bL1Cin1CopyLen * BLOCK_CUBE;
    return bL1Size;
}

uint64_t Conv3dBpFilterTiling::CalculateL1SizeGap()
{
    uint32_t al1LoadSize = blockTiling_.blockBaseK * blockTiling_.blockBaseM * dtypeByte_;
    uint32_t bl1LoadSize = CalBL1Bound() * dtypeByte_;
    uint64_t deltaL1LoadSize = (al1LoadSize + bl1LoadSize > platformInfo.l1Size) ?
                                al1LoadSize + bl1LoadSize - platformInfo.l1Size : 0;
    return deltaL1LoadSize;
}

uint32_t Conv3dBpFilterTiling::CalculateBl1Cin1CopyLen(uint32_t newBaseN)
{
    uint32_t kernelHW = static_cast<uint32_t>(shapeInfo.orgkH * shapeInfo.orgkW);
    if (newBaseN == 0) {
        return 0; // newBaseN 不可能为0
    }
    uint32_t bL1N = CeilDiv(static_cast<int64_t>(newBaseN), shapeCalc.channelSize);
    uint32_t bL1Cin1CopyLen = CeilDiv(bL1N, kernelHW); // 向上取整，拖尾时默认多搬一行
    if (kernelHW > bL1N && kernelHW % bL1N != 0) {
        ++bL1Cin1CopyLen; // 此时bL1Cin1CopyLen为1, 每个基本块不足一行，考虑拖尾最多搬两行
    } else if (NUM_HALF * bL1N % kernelHW != 0) {
        ++bL1Cin1CopyLen; // 除了尾块是0.5，其他场景都要搬2行
    }
    return bL1Cin1CopyLen;
}

bool Conv3dBpFilterTiling::ShrinkBlockBaseK()
{
    // k方向减小
    uint64_t fractalSize0 = shapeCalc.channelSize;
    uint64_t deltaL1LoadSize = CalculateL1SizeGap();
    // 基本块K方向每减小C0, L1A装载大小减小deltaAl1PerC0
    uint64_t deltaAl1PerC0 = blockTiling_.blockBaseM * fractalSize0 * dtypeByte_;
    
    uint32_t bL1Cin1CopyLen = CalculateBl1Cin1CopyLen(blockTiling_.blockBaseN);
    // 基本块K方向每减小C0, L1B装载大小减小deltaAl1PerC0, 本身这个过程是阶跃的, 此处做线性处理
    uint64_t deltaBl1PerC0 = CeilDiv(bL1Cin1CopyLen * fractalSize0 * shapeInfo.orgWi * attrInfo.strideH
                                            * fractalSize0 * dtypeByte_, static_cast<uint64_t>(shapeInfo.orgWo));
    
    // 线性处理后, deltaBl1PerC0一定不小于实际每C0减小, 所以c0ShrinkCount不会大于实际需减小C0数量
    uint64_t c0ShrinkCount = CeilDiv(deltaL1LoadSize, deltaAl1PerC0 + deltaBl1PerC0);
    uint64_t newBaseK = 0;
    if (blockTiling_.blockBaseK > static_cast<int64_t>(c0ShrinkCount * fractalSize0)) {
        newBaseK = blockTiling_.blockBaseK - c0ShrinkCount * fractalSize0;
    }
    if (newBaseK >= fractalSize0) {
        blockTiling_.blockBaseK = newBaseK;
        while(blockTiling_.blockBaseK > static_cast<int64_t>(fractalSize0) && IsCurBlockL1L0Invalid()) {
            blockTiling_.blockBaseK -= fractalSize0;
            if (static_cast<int32_t>(blockTiling_.blockBaseK) <= shapeInfo.orgWo
                && (shapeInfo.orgWo % blockTiling_.blockBaseK == 0 || shapeInfo.orgWo % fractalSize0 != 0)) {
                break;
            }
        }
        if (!IsCurBlockL1L0Invalid()) {
            return true;
        }
    } else {
        blockTiling_.blockBaseK = fractalSize0;
    }
    return false;
}

void Conv3dBpFilterTiling::ShrinkBlockBaseMN()
{
    uint64_t kernelHW = static_cast<uint64_t>(shapeInfo.orgkH * shapeInfo.orgkW);
    // M和N方向减小, 首先让M和N大小平齐
    while (blockTiling_.blockBaseM > static_cast<int64_t>(BLOCK_CUBE) && blockTiling_.blockBaseM > blockTiling_.blockBaseN 
            && IsCurBlockL1L0Invalid()) {
        blockTiling_.blockBaseM -= BLOCK_CUBE;
    }
    while (blockTiling_.blockBaseN > static_cast<int64_t>(BLOCK_CUBE) && blockTiling_.blockBaseN > blockTiling_.blockBaseM
            && IsCurBlockL1L0Invalid()) {
        blockTiling_.blockBaseN -= BLOCK_CUBE;
    }
    if (!IsCurBlockL1L0Invalid()) {
        return;
    }
    uint64_t deltaAl1PerC16 = blockTiling_.blockBaseK * BLOCK_CUBE * dtypeByte_;
    int32_t hoCal = 0;
    int32_t kBl1Size = blockTiling_.blockBaseK * blockTiling_.stepKb;
    if (kBl1Size % shapeInfo.orgWo == 0 || shapeInfo.orgWo % kBl1Size == 0) {
        hoCal = CeilDiv(static_cast<int64_t>(kBl1Size), shapeInfo.orgWo);
    } else if (kBl1Size > shapeInfo.orgWo) {
        hoCal = kBl1Size / shapeInfo.orgWo + NUM_HALF;
    } else {
        hoCal = NUM_HALF;
    }
    int32_t hiCal = (hoCal - 1) * attrInfo.strideH + (shapeInfo.orgkH - 1) * attrInfo.dilationH + 1;
    // 与K方向减小采用同样思路, 做线性化处理
    uint64_t deltaBl1PerC16 = CeilDiv(static_cast<uint64_t>(hiCal) * shapeInfo.orgWi * BLOCK_CUBE * dtypeByte_, kernelHW);
    uint64_t deltaL1LoadSize = CalculateL1SizeGap();
    uint32_t c0ShrinkCount = CeilDiv(deltaL1LoadSize, deltaAl1PerC16 + deltaBl1PerC16);
    if (static_cast<uint64_t>(blockTiling_.blockBaseM) < (c0ShrinkCount + 1) * BLOCK_CUBE) {
        blockTiling_.blockBaseM = BLOCK_CUBE;
        blockTiling_.blockBaseN = BLOCK_CUBE;
        return;
    }
    blockTiling_.blockBaseM -= (c0ShrinkCount * BLOCK_CUBE);
    blockTiling_.blockBaseN = blockTiling_.blockBaseM;
    uint32_t bL1Cin1CopyLen = CalculateBl1Cin1CopyLen(blockTiling_.blockBaseN);
    
    while (blockTiling_.blockBaseM > static_cast<int64_t>(BLOCK_CUBE) && IsCurBlockL1L0Invalid()) {
        uint32_t newBl1Cin1CopyLen = CalculateBl1Cin1CopyLen(blockTiling_.blockBaseM);// 向上取整，拖尾时默认多搬一行
        if (newBl1Cin1CopyLen < bL1Cin1CopyLen) {
            blockTiling_.blockBaseN = blockTiling_.blockBaseM;
            bL1Cin1CopyLen = newBl1Cin1CopyLen;
        } else {
            blockTiling_.blockBaseM -= BLOCK_CUBE;
        }
    }
}

void Conv3dBpFilterTiling::ShrinkBaseBlock()
{
    if (ShrinkBlockBaseK()) {
        return;
    }
    ShrinkBlockBaseMN();
    
    // M方向回调
    uint64_t fractalSize0 = shapeCalc.channelSize;
    uint32_t al1LoadSize = blockTiling_.stepKa * blockTiling_.blockBaseK * blockTiling_.stepM *
                           blockTiling_.blockBaseM * dtypeByte_ * blockTiling_.dbL1A;
    uint32_t bl1LoadSize = CalBL1Bound() * dtypeByte_ * blockTiling_.dbL1B;
    uint64_t deltaL1LoadSize = platformInfo.l1Size - al1LoadSize - bl1LoadSize;
    uint64_t deltaAl1PerC16M = blockTiling_.blockBaseK * BLOCK_CUBE * dtypeByte_;
    uint64_t c0CompensateCountM = deltaL1LoadSize / deltaAl1PerC16M;
    uint64_t cL0Max = platformInfo.l0CSize / dtypeByte_ / DB_ON;
    uint64_t newBaseMc = std::max(cL0Max / blockTiling_.blockBaseM / BLOCK_CUBE,
            static_cast<uint64_t>(1)) * BLOCK_CUBE;
    blockTiling_.blockBaseM = std::min(blockTiling_.blockBaseM + c0CompensateCountM * BLOCK_CUBE, CeilAlign(mmInfo_.mValue, BLOCK_CUBE_U64));
    blockTiling_.blockBaseM = std::min(newBaseMc, static_cast<uint64_t>(blockTiling_.blockBaseM));
    // K方向回调
    uint32_t validBaseK = blockTiling_.blockBaseK;
    while (!IsCurBlockL1L0Invalid()) {
        validBaseK = blockTiling_.blockBaseK;
        blockTiling_.blockBaseK += fractalSize0;
    }
    blockTiling_.blockBaseK = validBaseK;

    uint64_t aL0Max = platformInfo.l0ASize / dtypeByte_ / DB_ON;
    uint64_t bL0Max = platformInfo.l0BSize / dtypeByte_ / DB_ON;

    uint64_t alignedKValue = CeilAlign(mmInfo_.kValue, fractalSize0);
    if (static_cast<int64_t>(alignedKValue) < blockTiling_.blockBaseK) {
        blockTiling_.blockBaseK = alignedKValue;
    } else {
        if (blockTiling_.blockBaseM == 0 || blockTiling_.blockBaseN == 0) {
            return;
        }
        // 根据调小后的BaseM和BaseN调大BaseK，这里是兜底，由于LoadData的对齐限制，BaseK这里要16对齐
        uint64_t newBaseKa = std::max(aL0Max / blockTiling_.blockBaseM / BLOCK_CUBE,
            static_cast<uint64_t>(1)) * BLOCK_CUBE;
        uint64_t newBaseKb = std::max(bL0Max / blockTiling_.blockBaseN / BLOCK_CUBE,
            static_cast<uint64_t>(1)) * BLOCK_CUBE;
        uint64_t newBaseK = std::min(std::min(newBaseKa, newBaseKb), alignedKValue);
        blockTiling_.blockBaseK = std::min(newBaseK, static_cast<uint64_t>(blockTiling_.blockBaseK));

        // K在不超过L0约束情况下，优先满足搬运对齐
        if (shapeInfo.orgWo < static_cast<int32_t>(blockTiling_.blockBaseK) && shapeInfo.orgWo % fractalSize0 == 0) {
            blockTiling_.blockBaseK = blockTiling_.blockBaseK / shapeInfo.orgWo * shapeInfo.orgWo;
        }
    }
}

void Conv3dBpFilterTiling::SetFinalTiling(optiling::Conv3DBackpropFilterTilingData& tiling)
{
    Conv3dBpFilterTilingBase::SetFinalTiling(tiling);

    tiling.params.set_totalL1Size(static_cast<uint32_t>(platformInfo.l1Size));

    tiling.dwTiling.set_al0Pbuffer(static_cast<uint32_t>(tilingParams.al0Pbuffer));
    tiling.dwTiling.set_bl0Pbuffer(static_cast<uint32_t>(tilingParams.bl0Pbuffer));
    tiling.dwTiling.set_m0(static_cast<uint32_t>(BLOCK_CUBE));
    tiling.dwTiling.set_k0(static_cast<uint32_t>(shapeCalc.channelSize));
    tiling.dwTiling.set_n0(static_cast<uint32_t>(BLOCK_CUBE));
    
    // singleCore
    tiling.dwTiling.set_singleCoreDk(static_cast<uint32_t>(tilingParams.singleCoreDk));
    tiling.dwTiling.set_singleCoreGroup(static_cast<uint32_t>(tilingParams.singleCoreGroup));
    tiling.dwTiling.set_singleCoreBatch(static_cast<uint64_t>(tilingParams.singleCoreBatch));

    // 设置基本块
    SetFinalBasickBlockTiling(tiling);
}

void Conv3dBpFilterTiling::SetFinalBasickBlockTiling(optiling::Conv3DBackpropFilterTilingData& tiling)
{
    tiling.dwTiling.set_singleCoreHo(static_cast<uint32_t>(blockTiling_.singleCoreK / shapeInfo.orgWo));
    tiling.dwTiling.set_baseM(static_cast<uint32_t>(blockTiling_.blockBaseM));
    tiling.dwTiling.set_baseN(static_cast<uint32_t>(blockTiling_.blockBaseN));
    tiling.dwTiling.set_baseK(static_cast<uint32_t>(blockTiling_.blockBaseK));
    tiling.dwTiling.set_stepM(static_cast<uint32_t>(blockTiling_.stepM));
    tiling.dwTiling.set_stepN(static_cast<uint32_t>(blockTiling_.stepN));
    tiling.dwTiling.set_stepKa(static_cast<uint32_t>(blockTiling_.stepKa));
    tiling.dwTiling.set_stepKb(static_cast<uint32_t>(blockTiling_.stepKb));
    tiling.dwTiling.set_iterateOrder(static_cast<uint32_t>(blockTiling_.iterateOrder));
    tiling.dwTiling.set_al1Pbuffer(static_cast<uint32_t>(blockTiling_.dbL1A));
    tiling.dwTiling.set_bl1Pbuffer(static_cast<uint32_t>(blockTiling_.dbL1B));
    tiling.dwTiling.set_cl0Pbuffer(static_cast<uint32_t>(blockTiling_.dbL0C));
    tiling.dwTiling.set_bl1Bound(static_cast<uint32_t>(CalBL1Bound()));
    tiling.dwTiling.set_singleCoreCout(static_cast<uint32_t>(blockTiling_.singleCoreM));
    tiling.dwTiling.set_hf32Flag(static_cast<uint32_t>(attrInfo.hf32Enable));

    uint64_t l1Cin1 = std::max(blockTiling_.singleCoreN /
        static_cast<uint64_t>(shapeInfo.orgkH * shapeInfo.orgkW * shapeCalc.channelSize),
        static_cast<uint64_t>(1));
    tiling.dwTiling.set_singleCoreCin(static_cast<uint32_t>(l1Cin1 * shapeCalc.channelSize));

    tiling.basicBlockTiling.set_singleCoreM(static_cast<uint32_t>(blockTiling_.singleCoreM));
    tiling.basicBlockTiling.set_singleCoreN(static_cast<uint32_t>(blockTiling_.singleCoreN));
    tiling.basicBlockTiling.set_singleCoreK(static_cast<uint32_t>(blockTiling_.singleCoreK));
}


void Conv3dBpFilterTiling::PrintTilingData() const
{
    Conv3dBpFilterTilingBase::PrintTilingData();

    TILING_LOG_DEBUG("Total L1 Cache Size:%lu", platformInfo.l1Size);
    TILING_LOG_DEBUG("Al0 Pingpong Buffer Flag:%ld", tilingParams.al0Pbuffer);
    TILING_LOG_DEBUG("Bl0 Pingpong Buffer Flag:%ld", tilingParams.bl0Pbuffer);
    TILING_LOG_DEBUG("Block Size M:%d", BLOCK_CUBE);
    TILING_LOG_DEBUG("Channel Size K:%ld", shapeCalc.channelSize);
    TILING_LOG_DEBUG("Block Size N:%d", BLOCK_CUBE);
    TILING_LOG_DEBUG("Single Core Dk:%ld", tilingParams.singleCoreDk);
    TILING_LOG_DEBUG("Single Core Group:%ld", tilingParams.singleCoreGroup);
    TILING_LOG_DEBUG("Single Core Batch:%ld", tilingParams.singleCoreBatch);
    
    // basicblock tiling info
    TILING_LOG_DEBUG("Base M:%ld", blockTiling_.blockBaseM);
    TILING_LOG_DEBUG("Base N:%ld", blockTiling_.blockBaseN);
    TILING_LOG_DEBUG("Base K:%ld", blockTiling_.blockBaseK);
    TILING_LOG_DEBUG("Step M:%ld", blockTiling_.stepM);
    TILING_LOG_DEBUG("Step N:%ld", blockTiling_.stepN);
    TILING_LOG_DEBUG("Step Ka:%ld", blockTiling_.stepKa);
    TILING_LOG_DEBUG("Step Kb:%ld", blockTiling_.stepKb);
    TILING_LOG_DEBUG("Iterate Order:%ld", blockTiling_.iterateOrder);
    TILING_LOG_DEBUG("Al1 Pbuffer:%ld", blockTiling_.dbL1A);
    TILING_LOG_DEBUG("Bl1 Pbuffer:%ld", blockTiling_.dbL1B);
    TILING_LOG_DEBUG("Cl0 Pbuffer:%ld", blockTiling_.dbL0C);
    TILING_LOG_DEBUG("Bl1 Bound:%lu", CalBL1Bound());
    TILING_LOG_DEBUG("Single Core Cout:%ld", blockTiling_.singleCoreM);
    uint64_t l1Cin1 = std::max(blockTiling_.singleCoreN /
        static_cast<uint64_t>(shapeInfo.orgkH * shapeInfo.orgkW * shapeCalc.channelSize),
        static_cast<uint64_t>(1));
    TILING_LOG_DEBUG("Single Core Cin:%lu", l1Cin1 * shapeCalc.channelSize);
    TILING_LOG_DEBUG("Single Core M:%ld", blockTiling_.singleCoreM);
    TILING_LOG_DEBUG("Single Core N:%ld", blockTiling_.singleCoreN);
    TILING_LOG_DEBUG("Single Core K:%ld", blockTiling_.singleCoreK);
}

} // ConvBackpropApi
