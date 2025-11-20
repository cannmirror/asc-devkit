/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file ascend_antiquant_utils.h
 * \brief
 */
#ifndef LIB_QUANTIZATION_ASCEND_ANTIQUANT_UTILS_H
#define LIB_QUANTIZATION_ASCEND_ANTIQUANT_UTILS_H

namespace AscendC {
struct AntiQuantShapeInfo {
    uint32_t offsetHeight{0};
    uint32_t offsetWidth{0};
    uint32_t scaleHeight{0};
    uint32_t scaleWidth{0};
};

}; // namespace AscendC
#endif // LIB_QUANTIZATION_ASCEND_ANTIQUANT_UTILS_H