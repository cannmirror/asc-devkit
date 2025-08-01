/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_bp_input_tiling.cpp
 * \brief
 */
#include "detail/host_log.h"
#include "conv3d_bp_tiling_util.h"
#include "conv_backprop/conv3d_bp_input_tiling.h"

namespace {
constexpr uint32_t BLOCK_CUBE = 16;
constexpr uint32_t DB_ON = 2;
constexpr uint32_t DB_OFF = 1;
constexpr int64_t L0_AB_SIZE = 65536;
constexpr int32_t L0C_SIZE = 128 * 1024;
constexpr int32_t L1_SIZE = 512 * 1024;
constexpr uint64_t TWO = 2;
constexpr uint64_t ONE_U64 = 1;
constexpr uint32_t ONE_U32 = 1;
constexpr int64_t ONE_S64 = 1;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint32_t BASIC_BLOCK_SIZE_64 = 64;
constexpr int32_t W_MERGE_THRESHHOLD = 64; // 256基本块场景，w小于这个值拖尾影响不会大于1/8
constexpr uint32_t L2_CACHE_SIZE_THRESHHOLD = 150994944; // 144 * 1024 * 1024, B2经验值, 越过144M后L2效率开始下降
} // namespace

namespace ConvBackpropApi {
static inline bool CheckRange(int32_t value, int32_t valueLow, int32_t valueUp)
{
    if (value < valueLow || value > valueUp) {
        return false;
    }
    return true;
}

static inline bool IsOverflowInt32(int64_t value)
{
    if (value > INT32_MAX || value < INT32_MIN) {
        return true;
    }
    return false;
}

static inline uint32_t GetMaxDivisor(uint32_t a, uint32_t b, uint32_t step)
{
    while (b >= step) {
        if (a % b == 0) {
            return b;
        }
        b -= step;
    }
    return 0;
}

bool Conv3DBpInputTiling::CheckCalPads()
{
    int64_t filterDDilation = (shapeInfo.orgkD - 1) * attrInfo.dilationD + 1;
    int64_t filterHDilation = (shapeInfo.orgkH - 1) * attrInfo.dilationH + 1;
    int64_t filterWDilation = (shapeInfo.orgkW - 1) * attrInfo.dilationW + 1;

    if (opType_ == OpType::kConv3DTranspose) {
        filterDDilation += attrInfo.outputPadD;
        filterHDilation += attrInfo.outputPadH;
        filterWDilation += attrInfo.outputPadW;
    }

    int64_t doExpect =
        (shapeInfo.orgDi + attrInfo.padFront + attrInfo.padBack - filterDDilation) / attrInfo.strideD + 1;
    int64_t hoExpect = (shapeInfo.orgHi + attrInfo.padUp + attrInfo.padDown - filterHDilation) / attrInfo.strideH + 1;
    int64_t woExpect =
        (shapeInfo.orgWi + attrInfo.padLeft + attrInfo.padRight - filterWDilation) / attrInfo.strideW + 1;
    OP_TILING_CHECK(doExpect != shapeInfo.orgDo || hoExpect != shapeInfo.orgHo || woExpect != shapeInfo.orgWo,
        TILING_LOG_ERROR(
            "out_backprop's shape[%ld,%ld,%ld,%ld,%ld] is not equal with inferred shape[%ld,%ld,%ld,%ld,%ld]",
            shapeInfo.orgN, shapeInfo.orgCo, shapeInfo.orgDo, shapeInfo.orgHo, shapeInfo.orgWo, shapeInfo.orgN,
            shapeInfo.orgCo, doExpect, hoExpect, woExpect),
        return false);
    return true;
}

bool Conv3DBpInputTiling::CheckAttrs()
{
    OP_TILING_CHECK(attrInfo.strideD > shapeInfo.orgkD,
        TILING_LOG_ERROR("cannot support stride_d: %ld > kernel_d: %ld", attrInfo.strideD, shapeInfo.orgkD),
        return false);

    OP_TILING_CHECK(CheckRange(attrInfo.strideH, DIM_LOW, STRIDES_DIM_HW_UP) == false,
        TILING_LOG_ERROR(
            "stride_h: %ld is invaild, support range [%d, %d]", attrInfo.strideH, DIM_LOW, STRIDES_DIM_HW_UP),
        return false);

    OP_TILING_CHECK(CheckRange(attrInfo.strideW, DIM_LOW, STRIDES_DIM_HW_UP) == false,
        TILING_LOG_ERROR(
            "stride_w: %ld is invaild, support range [%d, %d]", attrInfo.strideW, DIM_LOW, STRIDES_DIM_HW_UP),
        return false);

    OP_TILING_CHECK(CheckRange(attrInfo.strideD, DIM_LOW, STRIDES_DIM_DEPTH_UP) == false,
        TILING_LOG_ERROR(
            "stride_d: %ld is invaild, support range [%d, %d]", attrInfo.strideD, DIM_LOW, STRIDES_DIM_DEPTH_UP),
        return false);
    uint64_t curL0CDstStride = static_cast<uint64_t>(shapeInfo.orgHi) * shapeInfo.orgWi;
    OP_TILING_CHECK(curL0CDstStride > UINT32_MAX,
        TILING_LOG_ERROR("cannot support hi * wi=%lu over %u", curL0CDstStride, UINT32_MAX), return false);

    OP_TILING_CHECK(CheckRange(attrInfo.groups, GROUPS_LOW, GROUPS_UP) == false,
        TILING_LOG_ERROR("only support groups(%ld) in range [%d, %d]", attrInfo.groups, GROUPS_LOW, GROUPS_UP),
        return false);
    return true;
}

bool Conv3DBpInputTiling::CheckPadRange()
{
    int32_t padHDimUp = std::min(PAD_DIM_UP, static_cast<int32_t>(shapeInfo.orgkD - 1));
    int32_t padUDimUp = std::min(PAD_DIM_UP, static_cast<int32_t>(shapeInfo.orgkH - 1));
    int32_t padLDimUp = std::min(PAD_DIM_UP, static_cast<int32_t>(shapeInfo.orgkW - 1));
    OP_TILING_CHECK(CheckRange(attrInfo.padFront, PAD_DIM_LOW, padHDimUp) == false,
        TILING_LOG_ERROR("pad head: %ld invaild, it should be in [%d, %d]", attrInfo.padFront, PAD_DIM_LOW, padHDimUp),
        return false);
    OP_TILING_CHECK(CheckRange(attrInfo.padBack, PAD_DIM_LOW, padHDimUp) == false,
        TILING_LOG_ERROR("pad tail: %ld invaild, it should be in [%d, %d]", attrInfo.padBack, PAD_DIM_LOW, padHDimUp),
        return false);
    OP_TILING_CHECK(CheckRange(attrInfo.padUp, PAD_DIM_LOW, padUDimUp) == false,
        TILING_LOG_ERROR("pad up: %ld invaild, it should be in [%d, %d]", attrInfo.padUp, PAD_DIM_LOW, padUDimUp),
        return false);
    OP_TILING_CHECK(CheckRange(attrInfo.padDown, PAD_DIM_LOW, padUDimUp) == false,
        TILING_LOG_ERROR("pad down: %ld invaild, it should be in [%d, %d]", attrInfo.padDown, PAD_DIM_LOW, padUDimUp),
        return false);
    OP_TILING_CHECK(CheckRange(attrInfo.padLeft, PAD_DIM_LOW, padLDimUp) == false,
        TILING_LOG_ERROR("pad left: %ld invaild, it should be in [%d, %d]", attrInfo.padLeft, PAD_DIM_LOW, padLDimUp),
        return false);
    OP_TILING_CHECK(CheckRange(attrInfo.padRight, PAD_DIM_LOW, padLDimUp) == false,
        TILING_LOG_ERROR("pad right: %ld invaild, it should be in [%d, %d]", attrInfo.padRight, PAD_DIM_LOW, padLDimUp),
        return false);
    return true;
}

bool Conv3DBpInputTiling::CheckOutputHeight()
{
    int64_t fmapHPadding = shapeInfo.orgHi + attrInfo.padUp + attrInfo.padDown;
    int64_t filterHDilation = shapeInfo.orgHi + attrInfo.padUp + attrInfo.padDown;

    int64_t hoModulo = (fmapHPadding - filterHDilation) % attrInfo.strideH;
    OP_TILING_CHECK(hoModulo > attrInfo.padDown,
        TILING_LOG_ERROR(
            "mod((fmapHPadding - filterHDilation), stride_h)=%ld is invalid, it should be less than or equal pad_down(%ld)",
            hoModulo, attrInfo.padDown),
        return false);

    return true;
}

bool Conv3DBpInputTiling::CheckTransposeOutputdingRange()
{
    // outputPadding值需要小于同维度dilation或stride
    OP_TILING_CHECK((attrInfo.outputPadD >= attrInfo.strideD && attrInfo.outputPadD >= attrInfo.dilationD),
        TILING_LOG_ERROR("outputPadD value[%ld] should smaller than dilationD[%ld] or strideD[%ld]",
            attrInfo.outputPadD, attrInfo.dilationD, attrInfo.strideD),
        return false);
    OP_TILING_CHECK((attrInfo.outputPadH >= attrInfo.strideH && attrInfo.outputPadH >= attrInfo.dilationH),
        TILING_LOG_ERROR("outputPadH value[%ld] should smaller than dilationH[%ld] or strideH[%ld]",
            attrInfo.outputPadH, attrInfo.dilationH, attrInfo.strideH),
        return false);
    OP_TILING_CHECK((attrInfo.outputPadW >= attrInfo.strideW && attrInfo.outputPadW >= attrInfo.dilationW),
        TILING_LOG_ERROR("outputPadW value[%ld] should smaller than dilationW[%ld] or strideW[%ld]",
            attrInfo.outputPadW, attrInfo.dilationW, attrInfo.strideW),
        return false);

    return true;
}

bool Conv3DBpInputTiling::InferShape()
{
    int64_t di = (shapeInfo.orgDo - 1) * attrInfo.strideD - attrInfo.padFront - attrInfo.padBack
                 + attrInfo.dilationD * (shapeInfo.orgkD - 1) + attrInfo.outputPadD + 1;
    int64_t hi = (shapeInfo.orgHo - 1) * attrInfo.strideH - attrInfo.padUp - attrInfo.padDown
                 + attrInfo.dilationH * (shapeInfo.orgkH - 1) + attrInfo.outputPadH + 1;
    int64_t wi = (shapeInfo.orgWo - 1) * attrInfo.strideW - attrInfo.padLeft - attrInfo.padRight
                 + attrInfo.dilationW * (shapeInfo.orgkW - 1) + attrInfo.outputPadW + 1;
    OP_TILING_CHECK((di <= 0 || hi <= 0 || wi <= 0),
        TILING_LOG_ERROR(
            "Infer shape failed, transpose output shape size should > 0, [dout, hout, wout] = [%ld, %ld, %ld]", di, hi,
            wi),
        return false);
    shapeInfo.orgDi = di;
    shapeInfo.orgHi = hi;
    shapeInfo.orgWi = wi;
    return true;
}

bool Conv3DBpInputTiling::CheckInputParam()
{
    if (!Conv3DBpInputTilingBase::CheckInputParam()) {
        return false;
    }
    if (opType_ == OpType::kConv3DTranspose) {
        CheckTransposeOutputdingRange();
        if (!InferShape()) {
            return false;
        }
    }

    OP_TILING_CHECK(!CheckCalPads() || !CheckAttrs() || !CheckPadRange() || !CheckOutputHeight(),
        TILING_LOG_ERROR("params is invalid."), return false);
    return true;
}

int64_t Conv3DBpInputTiling::GetTiling(optiling::Conv3DBackpropInputTilingData& tiling)
{
    int64_t ret = Compute();
    if (ret == -1) {
        TILING_LOG_ERROR("can not gen Conv3dBackpropInput api tiling");
        return -1;
    }

    SetFinalTiling(tiling);
    PrintTilingData();
    return 0;
}

void Conv3DBpInputTiling::SetInitOutput()
{
    int64_t fmapDepthWithPadding = shapeInfo.orgDi + attrInfo.padFront + attrInfo.padBack;
    int64_t fmapHeightWithPadding = shapeInfo.orgHi + attrInfo.padUp + attrInfo.padDown;

    int64_t filterDepthWithDilation = (shapeInfo.orgkD - 1) * attrInfo.dilationD + 1;
    int64_t filterHeightWithDilation = (shapeInfo.orgkH - 1) * attrInfo.dilationH + 1;

    int32_t doModulo = (fmapDepthWithPadding - filterDepthWithDilation) % attrInfo.strideD;
    int32_t hoModulo = (fmapHeightWithPadding - filterHeightWithDilation) % attrInfo.strideH;
    if (doModulo > attrInfo.padBack || hoModulo > attrInfo.padDown || attrInfo.strideH > shapeInfo.orgkH
        || (opType_ == OpType::kConv3DTranspose && (attrInfo.backpropPadDown > 0 || attrInfo.backpropPadHead > 0))
        || attrInfo.dilationD > 1) {
        // 1 is init output with l0C, 2 is init output with l1, defualt is 0 means not init output
        initOutputFlag = 1;
    }
}

void Conv3DBpInputTiling::SetBasicBlockAttrsTiling()
{
    blockSize_ = BYTE_BLOCK / g_dtypeSizeTab.at(descInfo.weightType.dtype);
    dtypeByte_ = g_dtypeSizeTab.at(descInfo.weightType.dtype);

    mmInfo_.mValue = ConvBackpropApi::CeilAlign(
        static_cast<uint64_t>(shapeInfo.orgHi) * shapeInfo.orgWi, static_cast<uint64_t>(blockSize_));
    mmInfo_.nValue = shapeCalc.cin1G * blockSize_;
    mmInfo_.kValue = static_cast<uint64_t>(shapeInfo.orgkH) * shapeInfo.orgkW * shapeCalc.cout1G
                     * blockSize_; // kernel_d是单独的循环，不算在L0的K值上
    lenHkWkC0_ = shapeInfo.orgkH * shapeInfo.orgkW * blockSize_;
}

void Conv3DBpInputTiling::SetBackpropPadInfo()
{
    int64_t bpPadTail = shapeInfo.orgDi - (static_cast<int64_t>(shapeInfo.orgDo - 1) * attrInfo.strideD + 1)
                        + (shapeInfo.orgkD - 1) * attrInfo.dilationD - attrInfo.backpropPadHead;
    TILING_LOG_DEBUG("backprop tail pad: %ld, origin backprop_pad_t: %ld", bpPadTail, attrInfo.backpropPadTail);
    if (CheckRange(bpPadTail, PAD_DIM_LOW, PAD_DIM_UP)) {
        attrInfo.backpropPadTail = static_cast<uint32_t>(bpPadTail);
    }

    int64_t bpPadDown = shapeInfo.orgHi - (static_cast<int64_t>(shapeInfo.orgHo - 1) * attrInfo.strideH + 1)
                        + (shapeInfo.orgkH - 1) * attrInfo.dilationH - attrInfo.backpropPadUp;
    TILING_LOG_DEBUG("backprop down pad: %ld, origin backprop_pad_d: %ld", bpPadDown, attrInfo.backpropPadDown);
    if (CheckRange(bpPadDown, PAD_DIM_LOW, PAD_DIM_UP)) {
        attrInfo.backpropPadDown = static_cast<uint32_t>(bpPadDown);
    }

    int64_t bpPadRight = shapeInfo.orgWi - (static_cast<int64_t>(shapeInfo.orgWo - 1) * attrInfo.strideW + 1)
                         + (shapeInfo.orgkW - 1) * attrInfo.dilationW - attrInfo.backpropPadLeft;
    TILING_LOG_DEBUG("backprop right pad: %ld, origin backprop_pad_r: %ld", bpPadRight, attrInfo.backpropPadRight);
    if (CheckRange(bpPadRight, PAD_DIM_LOW, PAD_DIM_UP)) {
        attrInfo.backpropPadRight = static_cast<uint32_t>(bpPadRight);
    }
}

void Conv3DBpInputTiling::SetFinalTiling(optiling::Conv3DBackpropInputTilingData& tiling)
{
    Conv3DBpInputTilingBase::SetFinalTiling(tiling);
    optiling::TConv3DBackpropInputTiling& dxt = tiling.conv3DDxTiling;
    dxt.set_baseD(1);
    dxt.set_baseBatch(1);
    dxt.set_baseGroup(1);

    dxt.set_c0(blockSize_);
    if (dtypeByte_ == F16_DATA_SIZE) {
        dxt.set_c0Bits(B16_BITS);
    } else if (dtypeByte_ == FP32_DATA_SIZE) {
        dxt.set_c0Bits(FP32_BITS);
    }
    dxt.set_initOutputFlag(initOutputFlag);

    // singleCore
    dxt.set_singleCoreBatch(1);
    dxt.set_singleCoreGroup(1);
    dxt.set_singleCoreDin(1);
    dxt.set_singleCoreHo(1);

    dxt.set_stepBatch(1);
    dxt.set_stepGroup(1);

    dxt.set_singleCoreM(tilingParams.singleCoreM);
    dxt.set_singleCoreCout(tilingParams.singleCoreCout);
    dxt.set_singleCoreCout1(tilingParams.singleCoreCout1);
    dxt.set_singleCoreCin1(tilingParams.singleCoreCin1);
    dxt.set_singleCoreCin(tilingParams.singleCoreCin);

    dxt.set_baseM(tilingParams.baseM);
    dxt.set_baseK(tilingParams.baseK);
    dxt.set_baseN(tilingParams.baseN);
    dxt.set_stepM(tilingParams.stepM);
    dxt.set_stepN(tilingParams.stepN);
    dxt.set_stepKa(tilingParams.stepKa);
    dxt.set_stepKb(tilingParams.stepKb);

    dxt.set_al0Pbuffer(tilingParams.al0Pbuffer); // 默认开
    dxt.set_bl0Pbuffer(tilingParams.bl0Pbuffer); // 默认开
    dxt.set_cl0Pbuffer(tilingParams.cl0Pbuffer);
    dxt.set_al1Pbuffer(tilingParams.al1Pbuffer);
    dxt.set_bl1Pbuffer(tilingParams.bl1Pbuffer);
    dxt.set_iterateOrder(tilingParams.iterateOrder);

    if (shapeInfo.orgkH * shapeInfo.orgkW == 1) {
        loadB2Condition_ = 2; // 2表示Hk*Wk = 1的情况
    } else if (tilingParams.baseK / blockSize_ >= static_cast<uint32_t>(shapeInfo.orgkH * shapeInfo.orgkW)) {
        loadB2Condition_ = 1;
    } else {
        loadB2Condition_ = 0;
    }
}

int64_t Conv3DBpInputTiling::Compute()
{
    OP_TILING_CHECK(!CheckInputParam(), TILING_LOG_ERROR("Tiling compute params check don't pass."), return -1;);

    Conv3DBpInputTilingBase::ShapeInitCalc();
    SetBasicBlockAttrsTiling();

    OP_TILING_CHECK(!CalModifyBackpropPadD(), TILING_LOG_ERROR("CalModifyBackpropPadD failed."), return -1;);
    OP_TILING_CHECK(!CalModifyBackpropPadHW(), TILING_LOG_ERROR("CalModifyBackpropPadHW failed."), return -1;);
    SetBackpropPadInfo(); // 基本块对backpropPad进行修正
    SetInitOutput();

    if (!MultiCoreSplitMN()) {
        return -1;
    }
    return 0;
}

int32_t Conv3DBpInputTiling::CalFmapH(const uint32_t& mL1Size) const
{
    int32_t hiCal;
    if (mL1Size == 0) {
        return 0; // mL1Size == 0 时，hoCal为 0
    }
    if (mL1Size % shapeInfo.orgWi == 0 || shapeInfo.orgWi % mL1Size == 0) {
        hiCal = CeilDiv(mL1Size, static_cast<uint32_t>(shapeInfo.orgWi));
    } else if (mL1Size > shapeInfo.orgWi) {
        hiCal = mL1Size / shapeInfo.orgWi + FMAP_H_NUM;
    } else {
        hiCal = FMAP_H_NUM;
    }
    int32_t khDilation = (shapeInfo.orgkH - 1) * attrInfo.dilationH + 1;
    int32_t hoCal = (hiCal - 1) + khDilation;
    int64_t hoExpand = static_cast<int64_t>(shapeInfo.orgHo - 1) * attrInfo.strideH + 1;
    return static_cast<int32_t>(std::min(static_cast<int64_t>(hoCal), hoExpand));
}

bool Conv3DBpInputTiling::IsStepL1Valid(const uint32_t& stepKa, const uint32_t& stepKb)
{
    if (lenHkWkC0_ == 0) {
        return false;
    }
    uint64_t kernelHW = static_cast<uint64_t>(shapeInfo.orgkH) * shapeInfo.orgkW;
    bool isHkWkAligned = stepKa * tilingParams.baseK % lenHkWkC0_ == 0 && stepKb * tilingParams.baseK % lenHkWkC0_ == 0;
    if (!isHkWkAligned) {
        return false;
    }

    uint64_t bL1Size = 0;
    uint64_t kBl1Size = stepKb * tilingParams.baseK;
    if (kBl1Size == 0) {
        return false;
    }
    uint64_t copyLine = 0;
    if (kBl1Size % lenHkWkC0_ == 0 || lenHkWkC0_ % kBl1Size == 0) {
        copyLine = ConvBackpropApi::CeilDiv(kBl1Size, lenHkWkC0_);
    } else if (kBl1Size > lenHkWkC0_) {
        copyLine = kBl1Size / lenHkWkC0_ + TWO;
    } else {
        copyLine = TWO;
    }
    bL1Size = tilingParams.bl1Pbuffer * dtypeByte_ * tilingParams.stepN * tilingParams.baseN * copyLine * lenHkWkC0_;

    uint64_t coutNum = std::max(stepKa * tilingParams.baseK / kernelHW, ONE_U64);
    uint64_t a1PixelNum = static_cast<uint64_t>(CalFmapH(tilingParams.stepM * tilingParams.baseM)) * shapeInfo.orgWo
                          * attrInfo.strideW * coutNum;
    uint64_t aL1Size = a1PixelNum * dtypeByte_ * tilingParams.al1Pbuffer;
    return aL1Size + bL1Size <= L1_SIZE;
}

void Conv3DBpInputTiling::InitBaseMNK()
{
    tilingParams.al0Pbuffer = DB_ON;
    tilingParams.bl0Pbuffer = DB_ON;
    tilingParams.cl0Pbuffer = DB_OFF;

    // 选取原则一: 计算访存比最高的256*64*128基本块
    // 选取原则二：L0A的MTE1效率是L0B两倍，对称场景优先让BaseM用256
    // 选取原则三: 右矩阵转置逆序需要处理BaseK/C0次，控制BaseK不要太大，避免指令队列阻塞
    uint32_t baseM = mmInfo_.mValue >= mmInfo_.nValue ? BASIC_BLOCK_SIZE_256 : BASIC_BLOCK_SIZE_128;
    uint32_t baseN = mmInfo_.mValue >= mmInfo_.nValue ? BASIC_BLOCK_SIZE_128 : BASIC_BLOCK_SIZE_256;
    uint32_t baseK = BASIC_BLOCK_SIZE_128 / dtypeByte_;

    AdjustBaseMNK(tilingParams.al0Pbuffer, tilingParams.cl0Pbuffer, baseM, baseN, baseK);

    tilingParams.baseM = baseM;
    tilingParams.baseK = baseK;
    tilingParams.baseN = baseN;
}

void Conv3DBpInputTiling::AdjustBaseMNK(
    const uint32_t l0abPingPong, const uint32_t l0cPingPong, uint32_t& baseM, uint32_t& baseN, uint32_t& baseK)
{
    uint32_t l0abMaxNum = L0_AB_SIZE / l0abPingPong / dtypeByte_;
    uint32_t l0cMaxNum = L0C_SIZE / l0cPingPong / ge::GetSizeByDataType(ge::DT_FLOAT);
    uint64_t alingedMValue = ConvBackpropApi::CeilAlign(mmInfo_.mValue, static_cast<uint64_t>(blockSize_));

    if (lenHkWkC0_ == 0) {
        return;
    }
    // K对齐约束大，优先做调整, 从最优基本块往下找到能满足搬运对齐的块
    baseK = std::min(static_cast<uint64_t>(baseK), mmInfo_.kValue);
    while (baseK > static_cast<uint32_t>(blockSize_)) {
        if (baseK % lenHkWkC0_ == 0 || lenHkWkC0_ % baseK == 0) {
            break;
        }
        baseK = std::max(baseK - blockSize_, static_cast<uint32_t>(blockSize_));
    }

    baseN = std::min(static_cast<uint64_t>(baseN), mmInfo_.nValue);
    baseM = std::min(static_cast<uint64_t>(baseM), alingedMValue);
    if (baseK == 0 || baseN == 0 || baseM == 0) {
        return;
    }

    uint32_t mnL0Max = std::max(l0abMaxNum / baseK / blockSize_, ONE_U32) * blockSize_;

    // N和K方向如果都比较小，M方向优化满足搬运对齐，而且做边界保护
    if (baseN < BASIC_BLOCK_SIZE_256 && baseK < BASIC_BLOCK_SIZE_128 / dtypeByte_) {
        uint32_t mL0cMax = std::max(l0cMaxNum / baseN / blockSize_, ONE_U32) * blockSize_;
        baseM = std::min(mnL0Max, mL0cMax);
        baseM = std::min(static_cast<uint64_t>(baseM), alingedMValue);
    }

    // M和K方向如果都比较小，N方向优化满足搬运对齐，而且做边界保护
    if (baseM < BASIC_BLOCK_SIZE_256 && baseK < BASIC_BLOCK_SIZE_128 / dtypeByte_) {
        uint32_t nL0cMax = std::max(l0cMaxNum / baseM / blockSize_, ONE_U32) * blockSize_;
        baseN = std::min(mnL0Max, nL0cMax);
        baseN = std::min(static_cast<uint64_t>(baseN), mmInfo_.nValue);
    }

    uint32_t maxBaseK = std::max(l0abMaxNum / std::max(baseM, baseN) / blockSize_, ONE_U32) * blockSize_;
    maxBaseK = std::min(static_cast<uint64_t>(maxBaseK), mmInfo_.kValue);
    while (maxBaseK > static_cast<uint32_t>(blockSize_)) {
        if (maxBaseK % lenHkWkC0_ == 0 || lenHkWkC0_ % maxBaseK == 0) {
            baseK = maxBaseK;
            break;
        }
        maxBaseK = std::max(maxBaseK - blockSize_, static_cast<uint32_t>(blockSize_));
    }
}

void Conv3DBpInputTiling::AlignCout1(uint32_t& cout1A, uint32_t& cout1B, bool adaptFP32) const
{
    if (cout1A == cout1B) {
        return;
    } else if (cout1B > cout1A) {
        cout1A = GetMaxDivisor(cout1B, cout1A, 1);
        return;
    }

    if (!adaptFP32) {
        cout1B = GetMaxDivisor(cout1A, cout1B, 1);
        return;
    }

    uint32_t tempCout1A = cout1A;
    while (tempCout1A % cout1B > 0) { tempCout1A--; }
    uint64_t cout1AB = static_cast<uint64_t>(tempCout1A) * cout1B;
    uint32_t step = BLOCK_CUBE / blockSize_;
    uint32_t tempCout1B = GetMaxDivisor(cout1A, cout1B, step);
    if (tempCout1B == 0) {
        cout1A = tempCout1A;
        return;
    }

    uint64_t cout1ABSmallerB = tempCout1B * cout1A;
    if (cout1ABSmallerB > cout1AB) {
        cout1B = tempCout1B;
    } else {
        cout1A = tempCout1A;
    }
}

void Conv3DBpInputTiling::EqualL1MatchStepMNK(uint32_t& stepKa, uint32_t& stepKb)
{
    uint32_t hoCal = CalFmapH(tilingParams.baseM); // 此处默认stepM=1
    uint64_t baseNHkWkC0Size = lenHkWkC0_ * tilingParams.baseN * dtypeByte_;
    uint64_t l1BSize = L1_SIZE / TWO / tilingParams.bl1Pbuffer;
    uint64_t l1ASize = L1_SIZE / TWO / tilingParams.al1Pbuffer;

    if (baseNHkWkC0Size == 0) {
        return;
    }
    // fp32场景下Cout0为16，c0为8，而tiling中的Cout1是以C0对其，因此需保证加载的cout1要为2的倍数
    uint32_t cout1B1 = std::max(ONE_U64, l1BSize / baseNHkWkC0Size);
    uint64_t curHiWiSize = static_cast<uint64_t>(dtypeByte_) * hoCal * shapeInfo.orgWo * attrInfo.strideW * blockSize_;
    if (curHiWiSize == 0) {
        return;
    }
    uint32_t cout1A1 = std::max(ONE_U64, l1ASize / curHiWiSize);
    if (cout1A1 >= static_cast<uint32_t>(shapeCalc.cout1G)) {
        cout1A1 = shapeCalc.cout1G;
    }

    if (cout1B1 >= static_cast<uint32_t>(shapeCalc.cout1G)) {
        cout1B1 = shapeCalc.cout1G;
    }
    AlignCout1(cout1A1, cout1B1, false);

    stepKa = std::max(ONE_U64, ConvBackpropApi::CeilDiv(static_cast<uint64_t>(cout1A1) * lenHkWkC0_,
                                   static_cast<uint64_t>(tilingParams.baseK)));
    stepKa = std::min(stepKa, UINT16_MAX / tilingParams.baseK);
    stepKb = std::max(ONE_U64, ConvBackpropApi::CeilDiv(static_cast<uint64_t>(cout1B1) * lenHkWkC0_,
                                   static_cast<uint64_t>(tilingParams.baseK)));
    if (stepKa > stepKb) {
        stepKa = ConvBackpropApi::FloorAlign(stepKa, stepKb);
    } else {
        stepKb = ConvBackpropApi::FloorAlign(stepKb, stepKa);
    }
    // fp32场景下需单独适配，以符合fp32场景要求
}

void Conv3DBpInputTiling::CalStepMNK()
{
    tilingParams.stepM = 1;
    tilingParams.stepN = 1;
    tilingParams.al1Pbuffer = DB_ON;
    tilingParams.bl1Pbuffer = DB_OFF;

    // 右矩阵是权重，总数据量小，优先让右矩阵全载
    uint64_t kIter = ConvBackpropApi::CeilDiv(mmInfo_.kValue, static_cast<uint64_t>(tilingParams.baseK));
    if (IsStepL1Valid(1, kIter) && shapeInfo.orgkD == 1) {
        uint32_t stepKaStrategy0 = 1;
        uint32_t stepKbStrategy0 = kIter;
        LadderMatchStepKWithFullLoad(stepKaStrategy0, stepKbStrategy0);

        if (IsStepL1Valid(stepKaStrategy0, stepKbStrategy0)) {
            tilingParams.stepKa = stepKaStrategy0;
            tilingParams.stepKb = stepKbStrategy0;
            return;
        }
    }
    tilingParams.bl1Pbuffer = DB_ON;

    uint32_t stepKaStrategy1 = 1;
    uint32_t stepKbStrategy1 = 1;
    EqualL1MatchStepMNK(stepKaStrategy1, stepKbStrategy1);

    uint32_t stepKaStrategy2 = 1;
    uint32_t stepKbStrategy2 = 1;
    LadderMatchStepMNK(stepKaStrategy2, stepKbStrategy2);

    // 优选基本块个数多的，载入一次尽可能多算
    if (IsStepL1Valid(stepKaStrategy1, stepKbStrategy1)
        && (stepKaStrategy1 + stepKbStrategy1 > stepKaStrategy2 + stepKbStrategy2)) {
        tilingParams.stepKa = stepKaStrategy1;
        tilingParams.stepKb = stepKbStrategy1;
    } else {
        tilingParams.stepKa = stepKaStrategy2;
        tilingParams.stepKb = stepKbStrategy2;
    }
}

void Conv3DBpInputTiling::LadderMatchStepKWithFullLoad(uint32_t& stepKa, const uint32_t& stepKb)
{
    stepKa = std::min(ConvBackpropApi::CeilDiv(mmInfo_.kValue, static_cast<uint64_t>(tilingParams.baseK)),
        static_cast<uint64_t>(stepKb));
    while (stepKa > 1) {
        if (IsStepL1Valid(stepKa, stepKb) && stepKb % stepKa == 0) {
            break;
        }
        --stepKa;
    }
    // 待kernel支持不对齐, 对齐HkWkC0的策略如果找不到，预期要按照再找一次
}

void Conv3DBpInputTiling::LadderMatchStepMNK(uint32_t& stepKa, uint32_t& stepKb)
{
    stepKa = std::min(ConvBackpropApi::CeilDiv(mmInfo_.kValue, static_cast<uint64_t>(tilingParams.baseK)),
        static_cast<uint64_t>(shapeInfo.orgkH * shapeInfo.orgkW));
    stepKb = stepKa;
    while (stepKa > 1 && stepKb > 1) {
        if (IsStepL1Valid(stepKa, stepKb)) {
            break;
        }
        --stepKa;
        --stepKb;
    }
    // 待kernel支持不对齐, 对齐HkWkC0的策略如果找不到，预期要按照再找一次
}

void Conv3DBpInputTiling::ShrinkBasicBlock()
{
    // 在阶梯匹配过程中, stepK已经衰减为1，合法性保护主要从减少基本块层大小来入手
    uint32_t baseMOri = tilingParams.baseM;
    uint32_t baseNOri = tilingParams.baseN;
    uint32_t baseKOri = tilingParams.baseK;

    uint32_t baseMStart = tilingParams.baseM;
    uint32_t baseNStart = tilingParams.baseN;
    uint32_t baseKStart = tilingParams.baseK;

    // K方向要满足对齐，载入量基本是固定的，优先从M和N方向调整
    uint32_t minBaseM = std::max(static_cast<uint32_t>(shapeInfo.orgWi), static_cast<uint32_t>(blockSize_));
    while (baseMStart > minBaseM || baseNStart > static_cast<uint32_t>(blockSize_)) {
        if (baseMStart > minBaseM && baseMStart > baseNStart) {
            baseMStart = std::max(baseMStart - blockSize_, static_cast<uint32_t>(blockSize_));
        } else {
            baseNStart = std::max(baseNStart - blockSize_, static_cast<uint32_t>(blockSize_));
        }
        tilingParams.baseM = baseMStart;
        tilingParams.baseN = baseNStart;
        tilingParams.baseK = baseKStart;

        LadderMatchStepMNK(tilingParams.stepKa, tilingParams.stepKb);
        if (IsStepL1Valid(tilingParams.stepKa, tilingParams.stepKb)) {
            break;
        }

        EqualL1MatchStepMNK(tilingParams.stepKa, tilingParams.stepKb);
        if (IsStepL1Valid(tilingParams.stepKa, tilingParams.stepKb)) {
            break;
        }
    }

    uint32_t l0MaxKNum = L0_AB_SIZE / tilingParams.al0Pbuffer / dtypeByte_ / std::max(baseMStart, baseNStart);
    baseKStart =
        std::min(static_cast<uint64_t>(std::max(l0MaxKNum / blockSize_, ONE_U32) * blockSize_), mmInfo_.kValue);

    if (lenHkWkC0_ == 0) {
        return;
    }
    while (baseKStart > static_cast<uint32_t>(blockSize_)) {
        baseKStart = std::max(baseKStart - blockSize_, static_cast<uint32_t>(blockSize_));
        tilingParams.baseK = baseKStart;
        if (baseKStart % lenHkWkC0_ != 0 && lenHkWkC0_ % baseKStart != 0) {
            continue;
        }

        LadderMatchStepMNK(tilingParams.stepKa, tilingParams.stepKb);
        if (IsStepL1Valid(tilingParams.stepKa, tilingParams.stepKb)) {
            return;
        }

        EqualL1MatchStepMNK(tilingParams.stepKa, tilingParams.stepKb);
        if (IsStepL1Valid(tilingParams.stepKa, tilingParams.stepKb)) {
            return;
        }
    }
    // 如果stepK * baseK支持了不对齐KernelHW，这里可以考虑回来适当调大baseK
    tilingParams.baseM = baseMOri;
    tilingParams.baseN = baseNOri;
    tilingParams.baseK = baseKOri;
}

void Conv3DBpInputTiling::LegalProtection()
{
    // L1合法，直接结束
    if (IsStepL1Valid(tilingParams.stepKa, tilingParams.stepKb)) {
        return;
    }

    // 减小基本块，L1合法，直接结束
    ShrinkBasicBlock();
    if (IsStepL1Valid(tilingParams.stepKa, tilingParams.stepKb)) {
        return;
    }

    // 从右往左依次关闭DB，再次尝试
    if (tilingParams.al1Pbuffer == DB_ON && tilingParams.bl1Pbuffer == DB_ON) {
        tilingParams.bl1Pbuffer = DB_OFF;
        LegalProtection();
    }

    if (tilingParams.al1Pbuffer == DB_ON && tilingParams.bl1Pbuffer == DB_OFF) {
        tilingParams.al1Pbuffer = DB_OFF;
        tilingParams.bl1Pbuffer = DB_ON;
        LegalProtection();
    }

    if (tilingParams.al1Pbuffer == DB_OFF && tilingParams.bl1Pbuffer == DB_ON) {
        tilingParams.bl1Pbuffer = DB_OFF;
        LegalProtection();
    }
}

bool Conv3DBpInputTiling::MultiCoreSplitMN()
{
    // 更新并设置L0基本块
    InitBaseMNK();

    // 更新并设置L1载入策略
    CalStepMNK();

    // L1合法性兜底
    LegalProtection();
    if (!IsStepL1Valid(tilingParams.stepKa, tilingParams.stepKb)) {
        TILING_LOG_ERROR("params exceed max L1 limit size");
        return false;
    }

    // 设置L2 Cache和核间切分策略
    SetSingleCoreInfo();
    return true;
}

bool Conv3DBpInputTiling::IsL2Efficient(const uint64_t singleCoreM, const uint64_t singleCoreN,
    const uint64_t singleCoreK, const uint64_t transdataWorkSpace)
{
    uint32_t doutCopyLine = std::max(shapeInfo.orgkD / attrInfo.strideD, ONE_S64);
    uint32_t houtCopyLine = std::max(CalFmapH(singleCoreM) / attrInfo.strideH, ONE_S64);
    uint64_t inputL2Cache =
        static_cast<uint64_t>(houtCopyLine) * shapeInfo.orgWo * doutCopyLine * shapeCalc.cout1G * blockSize_;
    uint64_t l2CacheSize =
        (inputL2Cache + singleCoreN * singleCoreK * doutCopyLine + singleCoreM * singleCoreN + transdataWorkSpace)
        * dtypeByte_ * coreNum_;
    return l2CacheSize <= L2_CACHE_SIZE_THRESHHOLD;
}

void Conv3DBpInputTiling::SetSingleCoreInfo()
{
    tilingParams.iterateOrder = 1; // 默认orderN, 暂无左矩阵全载逻辑
    tilingParams.singleCoreCout = shapeCalc.cout1G * blockSize_;
    tilingParams.singleCoreCout1 = shapeCalc.cout1G;
    tilingParams.singleCoreCin = tilingParams.stepN * tilingParams.baseN;
    tilingParams.singleCoreCin1 =
        ConvBackpropApi::CeilDiv(tilingParams.singleCoreCin, static_cast<uint64_t>(blockSize_));

    uint64_t batchDepth = static_cast<uint64_t>(shapeInfo.orgN) * shapeInfo.orgDi;
    uint64_t hwI = static_cast<uint64_t>(shapeInfo.orgHi) * shapeInfo.orgWi;
    uint64_t mAl1 = tilingParams.stepM * tilingParams.baseM;
    uint64_t nCnt = ConvBackpropApi::CeilDiv(static_cast<uint64_t>(shapeInfo.orgCi), tilingParams.singleCoreCin);
    uint64_t singleCoreK = static_cast<uint64_t>(shapeInfo.orgkH) * shapeInfo.orgkW * shapeCalc.cout1G * blockSize_;
    tilingParams.singleCoreM = std::max(mAl1 / shapeInfo.orgWi, ONE_U64) * shapeInfo.orgWi;

    // 场景一：其他方向核间分配均匀时，适度合并M方向的任务，减少头开销
    // 场景二：因为搬运对齐，导致单轮基本块计算仿存比过低, 适度合并M方向的任务
    if (shapeInfo.orgWi > W_MERGE_THRESHHOLD && mAl1 % shapeInfo.orgWi != 0 && shapeInfo.orgWi % mAl1 != 0) {
        uint64_t maxMCnt = ConvBackpropApi::CeilDiv(hwI, mAl1);
        for (uint64_t i = 1; i <= maxMCnt; ++i) {
            uint64_t tmpSingleCoreHWI =
                ConvBackpropApi::CeilDiv(static_cast<uint64_t>(shapeInfo.orgHi), i) * shapeInfo.orgWi;
            uint64_t tmpCnt = ConvBackpropApi::CeilDiv(hwI, tmpSingleCoreHWI) * batchDepth * nCnt;
            // 核间任务伦次原本就不能均分时，拖尾影响不超过1 / coreNum
            if (tmpCnt % coreNum_ != 0 && tmpCnt < coreNum_ * coreNum_) {
                continue;
            }

            // 不超过L2 Cache，评估L时要充分考虑Load3D的影响
            if (!IsL2Efficient(tmpSingleCoreHWI, tilingParams.singleCoreCin, singleCoreK,
                    tilingParams.baseM * tilingParams.baseN)) {
                continue;
            }
            tilingParams.singleCoreM = tmpSingleCoreHWI;
            break;
        }
    } else {
        uint64_t tmpSingleCoreM = mAl1;
        uint64_t tmpSingleCoreHWI = tilingParams.singleCoreM;
        while (tmpSingleCoreHWI < hwI && batchDepth % coreNum_ == 0) {
            tmpSingleCoreM += mAl1;
            tmpSingleCoreHWI = std::min(std::max(tmpSingleCoreM / shapeInfo.orgWi, ONE_U64) * shapeInfo.orgWi, hwI);
            if (hwI % tmpSingleCoreHWI != 0) {
                continue;
            }

            // 不超过L2 Cache，评估L时要充分考虑Load3D的影响
            if (!IsL2Efficient(tmpSingleCoreHWI, tilingParams.singleCoreCin, singleCoreK,
                    tilingParams.baseM * tilingParams.baseN)) {
                continue;
            }
            tilingParams.singleCoreM = tmpSingleCoreHWI;
        }
    }
}

static int32_t CalBackpropPadBefore(int32_t filter, int32_t dilation, int32_t pad)
{
    return (filter - 1) * dilation - pad;
}

static int64_t CalBackpropPadAfter(int64_t inputDim, int64_t outputDim, int32_t stride, int32_t pad)
{
    // orginal formula is inputDim = (outputDim * stride + 1) - padBefore + filterDilation, it can be simplified as
    // follow.
    return inputDim - outputDim * stride + pad;
}

bool Conv3DBpInputTiling::CalModifyBackpropPadD()
{
    int64_t pad_head_before = CalBackpropPadBefore(shapeInfo.orgkD, attrInfo.dilationD, attrInfo.padFront);
    int64_t pad_tail_after = CalBackpropPadAfter(shapeInfo.orgDi, shapeInfo.orgDo, attrInfo.strideD, attrInfo.padFront);
    OP_TILING_CHECK(
        IsOverflowInt32(pad_tail_after) || !CheckRange(static_cast<int32_t>(pad_tail_after), -PAD_DIM_UP, PAD_DIM_UP),
        TILING_LOG_ERROR(
            "pad_tail_after = (inputD - outputD * strideD + padHead)=%ld is invalid, it should be in[%d, %d]",
            pad_tail_after, -PAD_DIM_UP, PAD_DIM_UP),
        return false);
    pad_tail_after = (pad_tail_after + abs(pad_tail_after)) / K_NUM_TWO;
    attrInfo.backpropPadHead = pad_head_before;
    attrInfo.backpropPadTail = pad_tail_after;
    return true;
}

bool Conv3DBpInputTiling::CalModifyBackpropPadHW()
{
    int32_t padLeftBefore = CalBackpropPadBefore(shapeInfo.orgkW, attrInfo.dilationW, attrInfo.padLeft);
    int32_t padUpBefore = CalBackpropPadBefore(shapeInfo.orgkH, attrInfo.dilationH, attrInfo.padUp);

    OP_TILING_CHECK(CheckRange(padLeftBefore, 0, PAD_DIM_UP) == false,
        TILING_LOG_ERROR("backpropPadLeft=((kw - 1) * dilationW - padLeft)=[%d] is invalid, it should be in [%d, %d]",
            padLeftBefore, 0, PAD_DIM_UP),
        return false);
    OP_TILING_CHECK(CheckRange(padUpBefore, 0, PAD_DIM_UP) == false,
        TILING_LOG_ERROR("backpropPadUp=((kh - 1) * dilationH - padUp)=[%d] is invalid, it should be in [%d, %d]",
            padUpBefore, 0, PAD_DIM_UP),
        return false);

    int64_t padRightAfter = CalBackpropPadAfter(shapeInfo.orgWi, shapeInfo.orgWo, attrInfo.strideW, attrInfo.padLeft);
    int64_t padDownAfter = CalBackpropPadAfter(shapeInfo.orgHi, shapeInfo.orgHo, attrInfo.strideH, attrInfo.padUp);

    OP_TILING_CHECK(
        IsOverflowInt32(padRightAfter) || !CheckRange(static_cast<int32_t>(padRightAfter), -PAD_DIM_UP, PAD_DIM_UP),
        TILING_LOG_ERROR(
            "backpropRightPad = (inputW - outputW * strideW + padLeft)=%ld is invalid, it should be in[%d, %d]",
            padRightAfter, -PAD_DIM_UP, PAD_DIM_UP),
        return false);
    OP_TILING_CHECK(
        IsOverflowInt32(padDownAfter) || !CheckRange(static_cast<int32_t>(padDownAfter), -PAD_DIM_UP, PAD_DIM_UP),
        TILING_LOG_ERROR(
            "backpropDownPad = (inputH - outputH * strideH + padUp)=%ld is invalid, it should be in[%d, %d]",
            padDownAfter, -PAD_DIM_UP, PAD_DIM_UP),
        return false);

    padLeftBefore = (padLeftBefore + abs(padLeftBefore)) / K_NUM_TWO;
    padDownAfter = (padDownAfter + abs(padDownAfter)) / K_NUM_TWO;
    padRightAfter = (padRightAfter + abs(padRightAfter)) / K_NUM_TWO;

    attrInfo.backpropPadUp = padUpBefore;
    attrInfo.backpropPadDown = padDownAfter;
    attrInfo.backpropPadLeft = padLeftBefore;
    attrInfo.backpropPadRight = padRightAfter;
    return true;
}

void Conv3DBpInputTiling::PrintTilingData() const
{
    Conv3DBpInputTilingBase::PrintTilingData();

    TILING_LOG_DEBUG("singleCoreM: %lu", tilingParams.singleCoreM);
    TILING_LOG_DEBUG("singleCoreCout: %u", tilingParams.singleCoreCout);
    TILING_LOG_DEBUG("singleCoreCout1: %u", tilingParams.singleCoreCout1);
    TILING_LOG_DEBUG("singleCoreCin: %lu", tilingParams.singleCoreCin);
    TILING_LOG_DEBUG("singleCoreCin1: %u", tilingParams.singleCoreCin1);
    TILING_LOG_DEBUG("al0Pbuffer: %u", tilingParams.al0Pbuffer);
    TILING_LOG_DEBUG("bl0Pbuffer: %u", tilingParams.bl0Pbuffer);
    TILING_LOG_DEBUG("cl0Pbuffer: %u", tilingParams.cl0Pbuffer);
    TILING_LOG_DEBUG("al1Pbuffer: %u", tilingParams.al1Pbuffer);
    TILING_LOG_DEBUG("bl1Pbuffer: %u", tilingParams.bl1Pbuffer);
    TILING_LOG_DEBUG("baseM: %u", tilingParams.baseM);
    TILING_LOG_DEBUG("baseK: %u", tilingParams.baseK);
    TILING_LOG_DEBUG("baseN: %u", tilingParams.baseN);
    TILING_LOG_DEBUG("stepM: %u", tilingParams.stepM);
    TILING_LOG_DEBUG("stepN: %u", tilingParams.stepN);
    TILING_LOG_DEBUG("stepKa: %u", tilingParams.stepKa);
    TILING_LOG_DEBUG("stepKb: %u", tilingParams.stepKb);
    TILING_LOG_DEBUG("iterateOrder: %u", tilingParams.iterateOrder);
}
} // namespace ConvBackpropApi
