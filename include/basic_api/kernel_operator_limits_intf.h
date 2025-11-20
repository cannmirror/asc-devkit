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
 * \file kernel_operator_limits_intf.h
 * \brief
 */
#ifndef ASCENDC_KERNEL_OPERATOR_LIMITS_INTF_H
#define ASCENDC_KERNEL_OPERATOR_LIMITS_INTF_H

#include "kernel_operator_vec_duplicate_intf.h"
#if (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102)
namespace AscendC {
template <typename T>
struct NumericLimits {
public:
    constexpr __aicore__ static inline T Max()
    {
        static_assert(SupportType<T, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float, bfloat16_t>(),
            "current data type is not supported on current device!");
        if constexpr (std::is_same_v<T, uint8_t>) {
            return GetScalarBitcodeValue<uint8_t, T>(0xFFu);
        } else if constexpr (std::is_same_v<T, int8_t>) {
            return GetScalarBitcodeValue<uint8_t, T>(0x7Fu);
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            return GetScalarBitcodeValue<uint16_t, T>(0xFFFFu);
        } else if constexpr (std::is_same_v<T, int16_t>) {
            return GetScalarBitcodeValue<uint16_t, T>(0x7FFFu);
        } else if constexpr (std::is_same_v<T, uint32_t>) {
            return GetScalarBitcodeValue<uint32_t, T>(0xFFFFFFFFu);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return GetScalarBitcodeValue<uint32_t, T>(0x7FFFFFFFu);
        } else if constexpr (std::is_same_v<T, half>) {
            return GetScalarBitcodeValue<uint16_t, T>(0x7BFFu);
        } else if constexpr (std::is_same_v<T, float>) {
            return GetScalarBitcodeValue<uint32_t, T>(0x7F7FFFFFu);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            return GetScalarBitcodeValue<uint16_t, T>(0x7F7Fu);
        }
        return T();
    }

    constexpr __aicore__ static inline T Lowest()
    {
        static_assert(SupportType<T, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float, bfloat16_t>(),
            "current data type is not supported on current device!");
        if constexpr (std::is_same_v<T, uint8_t>) {
            return GetScalarBitcodeValue<uint8_t, T>(0x0u);
        } else if constexpr (std::is_same_v<T, int8_t>) {
            return GetScalarBitcodeValue<uint8_t, T>(0x80u);
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            return GetScalarBitcodeValue<uint16_t, T>(0x0u);
        } else if constexpr (std::is_same_v<T, int16_t>) {
            return GetScalarBitcodeValue<uint16_t, T>(0x8000u);
        } else if constexpr (std::is_same_v<T, uint32_t>) {
            return GetScalarBitcodeValue<uint32_t, T>(0x0u);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return GetScalarBitcodeValue<uint32_t, T>(0x80000000u);
        } else if constexpr (std::is_same_v<T, half>) {
            return GetScalarBitcodeValue<uint16_t, T>(0xFBFFu);
        } else if constexpr (std::is_same_v<T, float>) {
            return GetScalarBitcodeValue<uint32_t, T>(0xFF7FFFFFu);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            return GetScalarBitcodeValue<uint16_t, T>(0xFF7Fu);
        }
        return T();
    }

    constexpr __aicore__ static inline T Min()
    {
        static_assert(SupportType<T, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float, bfloat16_t>(),
            "current data type is not supported on current device!");
        if constexpr (std::is_same_v<T, uint8_t>) {
            return GetScalarBitcodeValue<uint8_t, T>(0x0u);
        } else if constexpr (std::is_same_v<T, int8_t>) {
            return GetScalarBitcodeValue<uint8_t, T>(0x80u);
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            return GetScalarBitcodeValue<uint16_t, T>(0x0u);
        } else if constexpr (std::is_same_v<T, int16_t>) {
            return GetScalarBitcodeValue<uint16_t, T>(0x8000u);
        } else if constexpr (std::is_same_v<T, uint32_t>) {
            return GetScalarBitcodeValue<uint32_t, T>(0x0u);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return GetScalarBitcodeValue<uint32_t, T>(0x80000000u);
        } else if constexpr (std::is_same_v<T, half>) {
            return GetScalarBitcodeValue<uint16_t, T>(0x0400u);
        } else if constexpr (std::is_same_v<T, float>) {
            return GetScalarBitcodeValue<uint32_t, T>(0x00800000u);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            return GetScalarBitcodeValue<uint16_t, T>(0x0080u);
        }
        return T();
    }

    constexpr __aicore__ static inline T Infinity()
    {
        static_assert(
            SupportType<T, half, float, bfloat16_t>(), "current data type is not supported on current device!");
        if constexpr (std::is_same_v<T, half>) {
            return GetScalarBitcodeValue<uint16_t, T>(0x7C00u);
        } else if constexpr (std::is_same_v<T, float>) {
            return GetScalarBitcodeValue<uint32_t, T>(0x7F800000u);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            return GetScalarBitcodeValue<uint16_t, T>(0x7F80u);
        }
        return T();
    }

    constexpr __aicore__ static inline T NegativeInfinity()
    {
        static_assert(
            SupportType<T, half, float, bfloat16_t>(), "current data type is not supported on current device!");
        if constexpr (std::is_same_v<T, half>) {
            return GetScalarBitcodeValue<uint16_t, T>(0xFC00u);
        } else if constexpr (std::is_same_v<T, float>) {
            return GetScalarBitcodeValue<uint32_t, T>(0xFF800000u);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            return GetScalarBitcodeValue<uint16_t, T>(0xFF80u);
        }
        return T();
    }

    constexpr __aicore__ static inline T QuietNaN()
    {
        static_assert(
            SupportType<T, half, float, bfloat16_t>(), "current data type is not supported on current device!");
        if constexpr (std::is_same_v<T, half>) {
            return GetScalarBitcodeValue<uint16_t, T>(0x7E00u);
        } else if constexpr (std::is_same_v<T, float>) {
            return GetScalarBitcodeValue<uint32_t, T>(0x7FC00000u);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            return GetScalarBitcodeValue<uint16_t, T>(0x7FC0u);
        }
        return T();
    }

    constexpr __aicore__ static inline T SignalingNaN()
    {
        static_assert(
            SupportType<T, half, float, bfloat16_t>(), "current data type is not supported on current device!");
        if constexpr (std::is_same_v<T, half>) {
            return GetScalarBitcodeValue<uint16_t, T>(0x7D00u);
        } else if constexpr (std::is_same_v<T, float>) {
            return GetScalarBitcodeValue<uint32_t, T>(0x7FA00000u);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            return GetScalarBitcodeValue<uint16_t, T>(0x7FA0u);
        }
        return T();
    }

    constexpr __aicore__ static inline T DeNormMin()
    {
        static_assert(
            SupportType<T, half, float, bfloat16_t>(), "current data type is not supported on current device!");
        if constexpr (std::is_same_v<T, half>) {
            return GetScalarBitcodeValue<uint16_t, T>(0x0001u);
        } else if constexpr (std::is_same_v<T, float>) {
            return GetScalarBitcodeValue<uint32_t, T>(0x00000001u);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            return GetScalarBitcodeValue<uint16_t, T>(0x0001u);
        }
        return T();
    }

    __aicore__ static inline void Max(const LocalTensor<T> &dst, uint32_t count)
    {
        Duplicate(dst, Max(), count);
    }

    __aicore__ static inline void Lowest(const LocalTensor<T> &dst, uint32_t count)
    {
        Duplicate(dst, Lowest(), count);
    }

    __aicore__ static inline void Min(const LocalTensor<T> &dst, uint32_t count)
    {
        Duplicate(dst, Min(), count);
    }

    __aicore__ static inline void Infinity(const LocalTensor<T> &dst, uint32_t count)
    {
        Duplicate(dst, Infinity(), count);
    }

    __aicore__ static inline void NegativeInfinity(const LocalTensor<T> &dst, uint32_t count)
    {
        Duplicate(dst, NegativeInfinity(), count);
    }

    __aicore__ static inline void QuietNaN(const LocalTensor<T> &dst, uint32_t count)
    {
        Duplicate(dst, QuietNaN(), count);
    }

    __aicore__ static inline void SignalingNaN(const LocalTensor<T> &dst, uint32_t count)
    {
        Duplicate(dst, SignalingNaN(), count);
    }

    __aicore__ static inline void DeNormMin(const LocalTensor<T> &dst, uint32_t count)
    {
        Duplicate(dst, DeNormMin(), count);
    }
};
}  // namespace AscendC
#endif
#endif  // ASCENDC_KERNEL_OPERATOR_LIMITS_INTF_H
