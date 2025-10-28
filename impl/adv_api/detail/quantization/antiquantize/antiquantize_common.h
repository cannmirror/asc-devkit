/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file antiquantize_common.h
 * \brief
 */
#ifndef IMPL_QUANTIZATION_ANTIQUANTIZE_ANTIQUANTIZE_COMMON_H
#define IMPL_QUANTIZATION_ANTIQUANTIZE_ANTIQUANTIZE_COMMON_H

namespace AscendC {
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
enum class AntiQuantizePolicy : int32_t {
    PER_TENSOR,
    PER_CHANNEL,
    PER_TOKEN,
    PER_GROUP,
};

struct AntiQuantizeConfig {
    AntiQuantizePolicy policy;
    bool hasOffset;
    int32_t kDim = 1;
};

struct AntiQuantizeParams {
    uint32_t m;
    uint32_t n;
    uint32_t groupSize = 0;
};
#endif
} // namespace AscendC
#endif // IMPL_QUANTIZATION_ANTIQUANTIZE_ANTIQUANTIZE_COMMON_H
