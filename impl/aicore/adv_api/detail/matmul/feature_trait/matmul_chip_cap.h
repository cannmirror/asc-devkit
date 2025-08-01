/**
 * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_chip_cap.h
 * \brief
 */

#ifndef AICORE_ADV_API_DETAIL_MATMUL_FEATURE_TRAIT_MATMUL_CHIP_CAP_H
#define AICORE_ADV_API_DETAIL_MATMUL_FEATURE_TRAIT_MATMUL_CHIP_CAP_H

namespace AscendC {
namespace Impl {
namespace Detail {

enum class FixpipeParamsType : int8_t { V220, V300, V310, NONE };

class MatmulChipCap {
public:
    struct Feature {
        bool supportUnitFlag;
        bool ifNeedUB;        // if some func needs vec, such as nd2nz
        bool ifSupportUBToL1; // if cube support ub to l1
        bool supportMNL0DB;
        FixpipeParamsType fixpipeParamsType;
        bool ifSupportLoad3dV2;
        bool ifSupportLoad2dTranspose;
        bool ifSupportLoad2dV2;
        bool ifSupportCmatrixInitVal;
        bool ifSupportFmatrixB;
        bool ifSupportUserDefine;
        bool ifSupportUBToL1Singleshape; // if support ub to l1 fullload, from vec
        bool ifMmadInstrSupportAntiQuant;
        bool ifSupportL0CToUB;
        bool ifSupportTrianMatmul;
    };

    __aicore__ constexpr static const Feature& GetFeatures()
    {
        return features[GetChipType()];
    }

private:
    enum { CHIP_TYPE_100, CHIP_TYPE_200, CHIP_TYPE_220, CHIP_TYPE_300, CHIP_TYPE_310, CHIP_TYPE_310R6 };

    __aicore__ inline constexpr static uint8_t GetChipType()
    {
#if __CCE_AICORE__ == 100
        return CHIP_TYPE_100;
#elif __CCE_AICORE__ == 200
        return CHIP_TYPE_200;
#elif __CCE_AICORE__ == 220
        return CHIP_TYPE_220;
#elif __CCE_AICORE__ == 300
        return CHIP_TYPE_300;
#elif defined(__DAV_C310__)
        return CHIP_TYPE_310;
#elif defined(__DAV_310R6__)
        return CHIP_TYPE_310R6;
#else
        static_assert("Find invalid chip type!");
        return CHIP_TYPE_100;
#endif
    }

private:
    constexpr static Feature features[] = {
        /*supportUnitFlag, ifNeedUB, ifSupportUBToL1, supportMNL0DB, fixpipeParamsType,
        ifSupportLoad3dV2, ifSupportLoad2dTranspose, ifSupportLoad2dV2,
        ifOnlyUseIsBiasForMmad, ifSupportFmatrixB, ifSupportUserDefine, ifSupportUBToL1Singleshape,
        ifMmadInstrSupportAntiQuant, ifSupportL0CToUB, ifSupportTrianMatmul*/
        /*100*/ {false, true, true, false, FixpipeParamsType::NONE, false, false, false, true, false, false, false,
            false, false, false},
        /*200*/
        {false, true, true, false, FixpipeParamsType::NONE, true, false, false, false, false, false, false, false,
            false, false},
        /*220*/
        {true, false, false, true, FixpipeParamsType::V220, true, true, false, false, true, true, false, false, false,
            true},
        /*300*/
        {true, false, true, false, FixpipeParamsType::V220, true, true, false, false, true, false, false, false, false,
            false},
        /*310*/
        {true, false, false, true, FixpipeParamsType::V310, true, false, true, false, true, true, true, false, true,
            false},
        /*310R6*/
        {true, false, false, true, FixpipeParamsType::V310, true, false, true, false, true, true, true, true, true,
            false}};
};
} // namespace Detail
} // namespace Impl
} // namespace AscendC
#endif // _MATMUL_CHIP_CAP_H_
