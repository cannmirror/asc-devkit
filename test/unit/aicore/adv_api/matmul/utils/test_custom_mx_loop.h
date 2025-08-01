/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include "test_custom_loop.h"

namespace TestCustomModules {
template <typename IMPL, class A_TYPE, const auto& MM_CFG, typename = void>
class CustomMxLoop : public CustomLoop<IMPL, A_TYPE, MM_CFG> {
public:
    __aicore__ inline int32_t GetOuterScaleKaIdx()
    {
        return 0;
    }
};

template <typename IMPL, typename TRANS_T, class A_TYPE, const auto& MM_CFG, typename = void>
class CustomMxKLoop : public CustomKLoop<IMPL, TRANS_T, A_TYPE, MM_CFG> {
public:
    __aicore__ inline int32_t GetScaleFactorA() const
    {
        return 1;
    }

    __aicore__ inline int32_t GetScaleFactorB() const
    {
        return 1;
    }

    __aicore__ inline int32_t GetNextOuterScaleKaIdx() const
    {
        return 0;
    }

    __aicore__ inline int32_t GetNextOuterScaleKbIdx() const
    {
        return 0;
    }

    __aicore__ inline int32_t GetOuterScaleKaIdx()
    {
        return 0;
    }

    __aicore__ inline int32_t GetOuterScaleKbIdx()
    {
        return 0;
    }

    __aicore__ inline int32_t GetTileShapeScaleA() const
    {
        return 0;
    }

    __aicore__ inline int32_t GetTileShapeScaleB() const
    {
        return 0;
    }

    __aicore__ inline bool IsScaleAKL1FullLoad() const
    {
        return true;
    }

    __aicore__ inline bool IsScaleBKL1FullLoad() const
    {
        return true;
    }
};
} // namespace TestCustomModules
