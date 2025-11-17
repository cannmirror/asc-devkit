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
 * \file conv3d_tiling_base.h
 * \brief
 */

#ifndef API_ASCENDC_TIKCFW_TILING_CONV3D_TILING_BASE_H
#define API_ASCENDC_TIKCFW_TILING_CONV3D_TILING_BASE_H

#include "conv3d_tilingdata.h"
#include "../../../../impl/adv_api/tiling/conv/conv3d_tiling_util.h"

namespace Conv3dTilingApi {
struct Conv3DL1Tiling {
    uint64_t kAL1 = 0;
    uint64_t kBL1 = 0;
    uint64_t mAL1Value = 0;
    uint64_t nBL1Value = 0;
    uint64_t mAL1DivmL0 = 0;
    uint64_t nBL1DivnL0 = 0;
    uint64_t cin1InAL1 = 0;
    uint64_t kAL1Tail = 0;
    uint64_t cin1InAL1Tail = 0;
    uint64_t kBL1DivK0 = 0;
    uint64_t kBL1Tail = 0;
    uint64_t kBL1TailDivK0 = 0;

    IterateMNOrder iterateMNOrder = IterateMNOrder::INVALID;
    bool isWeightBypass = false;
    bool biasFullLoadFlag = false;
    bool fixpParamsFullLoadFlag = false;
    bool al1FullLoad = false;
    bool bl1FullLoad = false;
};

struct Conv3DL0Tiling {
    uint64_t mL0 = 0;
    uint64_t kL0 = 0;
    uint64_t nL0 = 0;
    uint64_t nL0xk0 = 0;
    uint64_t kL0xorgCoAlignN0 = 0;
};

struct Conv3DInputshape {
    int64_t orgBatch = 1;
    int64_t orgkH = -1;
    int64_t orgkW = -1;
    int64_t orgkD = -1;
    int64_t orgCo = -1;
    int64_t coutOpt = -1;
    int64_t orgCi = -1;
    int64_t cinOpt = -1;
    int64_t orgDi = -1;
    int64_t orgHi = -1;
    int64_t orgWi = -1;

    int64_t singlekH = -1;
    int64_t singlekW = -1;
    int64_t singlekD = -1;
    int64_t singleBatch = 1;
    int64_t singleCi = -1;
    int64_t singleCo = -1;
    int64_t singleDo = -1;
    int64_t singleM = -1;
    int64_t singleHo = -1;
    int64_t singleWo = -1;
    int64_t singleCoreGroupOpt = -1;
};

struct Conv3DInputAttr {
    int64_t groups = 1;
    int64_t groupOpt = 1;

    int64_t padHead = 0;
    int64_t padTail = 0;
    int64_t padUp = 0;
    int64_t padDown = 0;
    int64_t padLeft = 0;
    int64_t padRight = 0;

    int64_t strideH = 1;
    int64_t strideW = 1;
    int64_t strideD = 1;

    int64_t dilationH = 1;
    int64_t dilationW = 1;
    int64_t dilationD = 1;
    int8_t offsetx = 0;
    uint8_t hf32Enable = 0;
    uint8_t hf32TransMode = 0;
};

struct Conv3DCalcShape {
    uint64_t singleCi1 = 0;
    uint64_t singleCo1 = 0;
    uint64_t singleM1 = 0;
    uint64_t orgHo = 0;
    uint64_t orgWo = 0;
    uint64_t orgDo = 0;
};

struct Conv3DDesc {
    ConvType fMapType  = {ConvCommonApi::ConvFormat::NDC1HWC0, ConvCommonApi::ConvDtype::FLOAT16, ConvCommonApi::TPosition::GM};
    ConvType weightType  = {ConvCommonApi::ConvFormat::FRACTAL_Z_3D, ConvCommonApi::ConvDtype::FLOAT16, ConvCommonApi::TPosition::GM};
    ConvType biasType = {ConvCommonApi::ConvFormat::ND, ConvCommonApi::ConvDtype::FLOAT16, ConvCommonApi::TPosition::GM};
    ConvType outputType = {ConvCommonApi::ConvFormat::NDC1HWC0, ConvCommonApi::ConvDtype::FLOAT16, ConvCommonApi::TPosition::CO1};
    ConvType quantScaleType = {ConvCommonApi::ConvFormat::ND, ConvCommonApi::ConvDtype::INT64, ConvCommonApi::TPosition::GM};
};

struct DoubleBufferValue {
    uint8_t pbAL1 = 1;
    uint8_t pbBL1 = 1;
    uint8_t pbAL0 = 2;
    uint8_t pbBL0 = 2;
    uint8_t pbCL0 = 1;
    uint64_t pBufferFlag = 0;
};

struct CubeInfo {
    uint32_t m0 = 0;
    uint32_t k0 = 0;
    uint32_t n0 = 0;
    ConvCommonApi::ConvDtype madType = ConvCommonApi::ConvDtype::CONVDTYPEMAX;
    ConvCommonApi::ConvDtype biasType = ConvCommonApi::ConvDtype::CONVDTYPEMAX;
    uint32_t minBurstNum = 0;
};

class Conv3dTilingBase {
public:
    explicit Conv3dTilingBase(const platform_ascendc::PlatformAscendC& ascendcPlatform);
    explicit Conv3dTilingBase(const PlatformInfo& platform);
    virtual ~Conv3dTilingBase() = default;
    virtual int64_t GetTiling(optiling::TConv3DApiTiling& tiling) = 0;
    void SetOrgWeightShape(int64_t orgCo, int64_t orgKd, int64_t orgKh, int64_t orgKw);
    void SetOrgInputShape(int64_t orgCi, int64_t orgDi, int64_t orgHi, int64_t orgWi);
    void SetSingleWeightShape(int64_t singleCi, int64_t singleKd, int64_t singleKh, int64_t singleKw);
    void SetSingleOutputShape(int64_t singleCo, int64_t singleDo, int64_t singleM);
    void SetWeightType(const ConvCommonApi::TPosition pos, const ConvCommonApi::ConvFormat format, const ConvCommonApi::ConvDtype dtype);
    void SetInputType(const ConvCommonApi::TPosition pos, const ConvCommonApi::ConvFormat format, const ConvCommonApi::ConvDtype dtype);
    void SetBiasType(const ConvCommonApi::TPosition pos, const ConvCommonApi::ConvFormat format, const ConvCommonApi::ConvDtype dtype);
    void SetOutputType(const ConvCommonApi::TPosition pos, const ConvCommonApi::ConvFormat format, const ConvCommonApi::ConvDtype dtype);
    void SetPadding(int64_t padHead, int64_t padTail, int64_t padUp, int64_t padDown,
        int64_t padLeft, int64_t padRight);
    void SetDilation(int64_t dilationD, int64_t dilationH, int64_t dilationW);
    void SetStride(int64_t strideD, int64_t strideH, int64_t strideW);
    void SetGroups(int64_t groups);

    Conv3DDesc descInfo;
    Conv3DInputshape shapeInfo;
    Conv3DCalcShape shapeCalc;
    Conv3DInputAttr attrInfo;
    CubeInfo cubeInfo;
    Conv3DL1Tiling l1TilingInfo;
    Conv3DL0Tiling l0TilingInfo;
    DoubleBufferValue dbValue;
    PlatformInfo platformInfo;

    bool hasBias = false;
    bool hasQuantScale = false;

    bool hf32Enable_ = false;
    bool hf32TransMode_ = false;

protected:
    virtual int64_t Compute() = 0;
    void SetFinalTilingBasicInfo(optiling::TConv3DApiTiling& tiling);
    void SetFinalTilingDecisionInfo(optiling::TConv3DApiTiling& tiling);
    void SetFinalTiling(optiling::TConv3DApiTiling& tiling);
    void PrintTilingDataBasicInfo() const;
    void PrintTilingDataDecision() const;
    void PrintTilingData() const;
    bool CheckInputParam();
    bool CheckSocVersion();
    void GetCubeInfo();
    bool ShapeInitCalc();
    bool CheckParamsOverflow();

private:
    bool CheckInputAttr();
    bool CheckOrgInputInfo();
    bool CheckSingleInputInfo();
    bool CheckInputConstraint();
    bool CheckInputShape();
    bool CheckInputFormat();
    bool CheckParamsDtype();
    bool CheckLoad3DLimits();
    bool CheckLoadL1SizeLimits();
    bool CheckInstructionLimits();
    bool CheckHF32();
    bool CheckPaddedInput();
    void SetSingleOutputShape(int64_t singleCo, int64_t singleDo, int64_t singleHo, int64_t singleWo);
    void SetHF32(bool hf32Enable, bool hf32TransMode);
};
} // namespace Conv3dTilingApi

#endif