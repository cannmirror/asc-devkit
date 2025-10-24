/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv_status_utils.h
 * \brief
 */

#ifndef UTILS_CONV_STATUS_UTILS_H
#define UTILS_CONV_STATUS_UTILS_H

#include <map>
namespace Act {
namespace Conv {

enum class ConvStatus : std::uint8_t {
    success = 0,
    batchErrorMatrixExceedsLimit,
    fmCiErrorMatrixExceedsLimit,
    fmHinErrorMatrixExceedsLimit,
    fmWinErrorMatrixExceedsLimit,
    khErrorMatrixExceedsLimit,
    kwErrorMatrixExceedsLimit,
    coutErrorMatrixExceedsLimit,
    houtErrorMatrixExceedsLimit,
    woutErrorMatrixExceedsLimit,
    max = 0xff
};

static std::map<ConvStatus, const char*> StatusToString = {
    {ConvStatus::success, "Success.\n"},
    {ConvStatus::batchErrorMatrixExceedsLimit, "[ERROR] batch out of range.\n"},
    {ConvStatus::fmCiErrorMatrixExceedsLimit, "[ERROR] featureMap matrix ci  out of range.\n"},
    {ConvStatus::fmHinErrorMatrixExceedsLimit, "[ERROR] featureMap matrix hin out of range.\n"},
    {ConvStatus::fmWinErrorMatrixExceedsLimit, "[ERROR] featureMap matrix win out of range.\n"},
    {ConvStatus::khErrorMatrixExceedsLimit, "[ERROR] weight matrix hout of range.\n"},
    {ConvStatus::kwErrorMatrixExceedsLimit, "[ERROR] weight matrix wout of range.\n"},
    {ConvStatus::coutErrorMatrixExceedsLimit, "[ERROR] output matrix co out of range.\n"},
    {ConvStatus::houtErrorMatrixExceedsLimit, "[ERROR] output matrix hout out of range.\n"},
    {ConvStatus::woutErrorMatrixExceedsLimit, "[ERROR] outputoutput wout co out of range.\n"}
};

#define CONV_CHECK_AND_RETURN(status)                      \
    do {                                                   \
        ConvStatus ret = status;                           \
        if (ret != ConvStatus::success) {                  \
            return ret;                                    \
        }                                                  \
    } while (0)

#define ACT_CONV_CHECK (status)                             \
    do {                                                    \
        ConvStatus ret = status;                            \
        if (ret != ConvStatus::success) {                   \
            std::cerr << "Got act error: " << std::endl;    \
        }                                                   \
    } while (0)

#define CHECK_AND_RETURN(status)                            \
    do {                                                    \
        ConvStatus ret = status;                            \
        if (ret != ConvStatus::success) {                   \
            return ret;                                     \
        }                                                   \
    } while (0)

#define ACT_CHECK(status)                                   \
    do {                                                    \
        ConvStatus ret = status;                            \
        if (ret != ConvStatus::success) {                   \
            std::cerr << "Got act error: " << std::endl;    \
            return;                                         \
        }                                                   \
    } while (0)
} // namespace Conv
} // namespace Act
#endif