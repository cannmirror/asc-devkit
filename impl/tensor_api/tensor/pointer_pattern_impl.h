/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file pointer_pattern_impl.h
* \brief
*/
#ifndef IMPL_TENSOR_API_TENSOR_POINTER_PATTERN_IMPL_H
#define IMPL_TENSOR_API_TENSOR_POINTER_PATTERN_IMPL_H

#include "impl/tensor_api/tensor/pointer_mem_impl.h"

namespace AscendC {
namespace Te {

struct MakeGMMemPtr {
    template <typename TraitType, typename Iterator>
    __aicore__ inline static auto Make(Iterator iter) {
        return HardwareMemPtr<Hardware::GM, Iterator>{iter};
    }
};

struct MakeUBMemPtr {
    template <typename TraitType, typename Iterator>
    __aicore__ inline static auto Make(Iterator iter) {
        return HardwareMemPtr<Hardware::UB, Iterator>{iter};
    }
};

struct MakeL1MemPtr {
    template <typename TraitType, typename Iterator>
    __aicore__ inline static auto Make(Iterator iter) {
        return HardwareMemPtr<Hardware::L1, Iterator>{iter};
    }
};

struct MakeL0AMemPtr {
    template <typename TraitType, typename Iterator>
    __aicore__ inline static auto Make(Iterator iter) {
        return HardwareMemPtr<Hardware::L0A, Iterator>{iter};
    }
};

struct MakeL0BMemPtr {
    template <typename TraitType, typename Iterator>
    __aicore__ inline static auto Make(Iterator iter) {
        return HardwareMemPtr<Hardware::L0B, Iterator>{iter};
    }
};

struct MakeL0CMemPtr {
    template <typename TraitType, typename Iterator>
    __aicore__ inline static auto Make(Iterator iter) {
        return HardwareMemPtr<Hardware::L0C, Iterator>{iter};
    }
};

struct MakeBiasMemPtr {
    template <typename TraitType, typename Iterator>
    __aicore__ inline static auto Make(Iterator iter) {
        return HardwareMemPtr<Hardware::BIAS, Iterator>{iter};
    }
};

struct MakeFixbufMemPtr {
    template <typename TraitType, typename Iterator>
    __aicore__ inline static auto Make(Iterator iter) {
        return HardwareMemPtr<Hardware::FIXBUF, Iterator>{iter};
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_POINTER_PATTERN_IMPL_H
