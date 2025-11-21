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
 * \file kernel_utils.h
 * \brief
 */
#ifndef ASCENDC_MODULE_UTILS_H
#define ASCENDC_MODULE_UTILS_H
#include "utils/kernel_utils_macros.h"
#include "utils/kernel_utils_ceil_oom_que.h"
#include "utils/kernel_utils_constants.h"
#include "utils/kernel_utils_mode.h"
#include "utils/kernel_utils_struct_confusion_pad.h"
#include "utils/kernel_utils_struct_dma_params.h"
#include "utils/kernel_utils_struct_norm_sort.h"
#include "utils/kernel_utils_struct_param.h"

#include "kernel_struct_data_copy.h"
#include "kernel_scalar_convert.h"

#if ENABLE_CV_COMM_VIA_SSBUF != 0 && __MIX_CORE_AIC_RATION__ != 1
#define KFC_C310_SSBUF 1
#else
#define KFC_C310_SSBUF 0
#endif

inline __gm__ void* g_sysFftsAddr = nullptr;
namespace AscendC {

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102))
namespace Internal {
// global varaibles g_cmpMaskLow and g_cmpMaskHigh are used to simulate the registr CMPMASK in 1971
// both of them are 64 bits and they are used to store the result of API Compare
__BLOCK_LOCAL__ __inline__ uint64_t g_cmpMaskLow;
__BLOCK_LOCAL__ __inline__ uint64_t g_cmpMaskHigh;
// the global variable g_deqScale is used to store the scale offset and signMode of API CastDeq
// when you are using API "SetDeqScaleImpl(float scale, int16_t offset, bool signMode)", g_deqScale will save
// the result of the transformation of three variable data
// otherwise, if you are using API "SetDeqScaleImpl(const LocalTensor<T> &vdeq, const VdeqInfo &vdeqInfo)"
// g_deqScale will store the UB address of vdeq, and the data of vdeqInfo will be stored in vdeq
__BLOCK_LOCAL__ __inline__ uint64_t g_deqScale;
// manage the global id for get/rls buff.
__BLOCK_LOCAL__ __inline__ uint32_t g_bufId;
__BLOCK_LOCAL__ __inline__ uint32_t g_sharedEvtId;
// global varaibles g_aipp* are used to simulate the spr for SetAippFunctions and LoadImageToLocal, they will save
// the configs and apply them to pre-process the input image in LoadImageToLocal function.
__BLOCK_LOCAL__ __inline__ uint64_t g_aippSrc0;
__BLOCK_LOCAL__ __inline__ uint64_t g_aippSrc1;
__BLOCK_LOCAL__ __inline__ uint64_t g_aippCscRc0;
__BLOCK_LOCAL__ __inline__ uint64_t g_aippCscRc1;
__BLOCK_LOCAL__ __inline__ uint64_t g_aippCscBias;
__BLOCK_LOCAL__ __inline__ uint64_t g_aippDtcMean;
__BLOCK_LOCAL__ __inline__ uint64_t g_aippDtcMin;
__BLOCK_LOCAL__ __inline__ uint64_t g_aippDtcVar;
__BLOCK_LOCAL__ __inline__ uint64_t g_aippPaddingVal;
__BLOCK_LOCAL__ __inline__ uint64_t g_aippArgs;

} // namespace Internal
#endif

class AscendCUtils {
public:
    __aicore__ static inline int32_t GetBitSize(int32_t byteSize)
    {
        return byteSize * ONE_BYTE_BIT_SIZE;
    }

    __aicore__ static inline int32_t GetC0Size()
    {
        return DEFAULT_C0_SIZE;
    }

    __aicore__ static inline void InitSocStateImpl()
    {
    #if defined(__NPU_ARCH__) && (((__NPU_ARCH__ == 3113)) || (__NPU_ARCH__ == 3103))
    #else
        set_atomic_none();
    #endif
    #if __NPU_ARCH__ == 2201
        set_mask_norm();
        if ASCEND_IS_AIC {
            set_l1_3d_size(static_cast<uint64_t>(0));
            set_padding(static_cast<uint64_t>(0));
        } else {
            set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        }
    #elif __NPU_ARCH__ == 3101
        set_mask_norm();
        uint64_t prevCtrl = get_ctrl() & 0x1000000000000;
        uint64_t val = 0x1000000000000008 | prevCtrl;
        set_ctrl(val);
        if ASCEND_IS_AIC {
            set_padding(static_cast<uint64_t>(0));
        } else {
            set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
            uint64_t loopSizePara = (1uL << 21) | 1uL;
            set_loop_size_ubtoout(loopSizePara);
            set_loop_size_outtoub(loopSizePara);
        }
        set_st_atomic_cfg(0b00100100);
    #elif __NPU_ARCH__ == 3002
        set_padding(static_cast<uint64_t>(0));
    #elif (__NPU_ARCH__ == 5102)
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    #endif
    }

    __aicore__ static inline int32_t GetC0Count(const int32_t dtypeSize)
    {
        ASCENDC_ASSERT((dtypeSize != 0), { KERNEL_LOG(KERNEL_ERROR, "dtypeSize can not be 0"); });
        return GetC0Size() / dtypeSize;
    }

    __aicore__ static inline int32_t GetDefaultBlockNum()
    {
        return DEFAULT_BLK_NUM;
    }

    __aicore__ static inline int64_t GetRsvdCnt()
    {
        return get_rsvd_cnt();
    }

    template <typename T, bool isSetMask = true>
    __aicore__ static inline void SetMask(const uint64_t& maskHigh, const uint64_t& maskLow)
    {
        if constexpr (!isSetMask) {
            return;
        }

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102))
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        if (sizeof(T) >= sizeof(int32_t)) {
            ASCENDC_ASSERT((maskHigh == 0ULL),
                           { KERNEL_LOG(KERNEL_ERROR, "maskHigh must be 0 for type b32 and b64"); });
        }
        ASCENDC_ASSERT(((maskLow != 0ULL) || (maskHigh != 0ULL)),
                       { KERNEL_LOG(KERNEL_ERROR, "maskLow and maskHigh can not be zero at the same time"); });
#endif
#endif
        if ASCEND_IS_NOT_AIC {
            set_vector_mask(maskHigh, maskLow);
        }
    }

    template <typename T, bool isSetMask = true> __aicore__ static inline void SetMask(int32_t len)
    {
        if constexpr (!isSetMask) {
            return;
        }

        int32_t typeLen = 0;
        if constexpr (IsSameType<T, int4b_t>::value) {
            typeLen = DEFAULT_BLOCK_SIZE * INT4_TWO;
#if (__NPU_ARCH__ == 5102)
        } else if constexpr (IsSameType<T, int2b_t>::value) {
            typeLen = DEFAULT_BLOCK_SIZE * INT2_FOUR;
#endif
        } else {
            typeLen = DEFAULT_BLOCK_SIZE / sizeof(T);
        }
        constexpr int32_t halfTypeLen = 64;  // 1 register -> 64 bits -> 64 elements
        constexpr int32_t lenCoeff = 2;      // 2 registers for masks
        if (len == halfTypeLen) {
            SetMask<T>(0, FULL_MASK);
            return;
        } else if (len == typeLen || len >= halfTypeLen * lenCoeff) { // len = max ele per repeat / len >= 128
            SetMask<T>(FULL_MASK, FULL_MASK);
            return;
        }
        SetMask<T>(static_cast<uint64_t>(
            (len > halfTypeLen) ? (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(len - halfTypeLen)) - 1) : 0),
            static_cast<uint64_t>(
            (len > halfTypeLen) ? FULL_MASK : (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(len)) - 1)));
    }

    template <typename T> __aicore__ static inline void SetMaskCount()
    {
        set_mask_count();
    }

    template <typename T> __aicore__ static inline void SetMaskNorm()
    {
        set_mask_norm();
    }

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2201) || (__NPU_ARCH__ == 3002) ||       \
    (__NPU_ARCH__ == 3102) || (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102))
    __aicore__ static inline void SetOverflow(uint64_t ctrlValue)
    {
        // set CTRL[48] is 1 --- inf/nan mode
        // set CTRL[48] is 0 --- saturated mode
        if (ctrlValue == 1) {
            set_ctrl(sbitset1(get_ctrl(), CTRL_48_BIT));
        } else {
            set_ctrl(sbitset0(get_ctrl(), CTRL_48_BIT));
        }
    }

#elif __NPU_ARCH__ == 2002
    __aicore__ static inline void SetOverflow(uint64_t ctrlValue)
    {
        // set CTRL[53] is 1 --- saturated mode
        // set CTRL[53] is 0 --- inf/nan mode
        if (ctrlValue == 0) {
            set_ctrl(sbitset1(get_ctrl(), CTRL_53_BIT));
        } else {
            set_ctrl(sbitset0(get_ctrl(), CTRL_53_BIT));
        }
    }
#endif

    template <bool isSetMask = true> __aicore__ static inline void ResetMask()
    {
        if constexpr (!isSetMask) {
            return;
        }
        if ASCEND_IS_NOT_AIC {
            set_vector_mask(FULL_MASK, FULL_MASK);
        }
    }

    template <bool isInt4 = false>
    __aicore__ inline static IntriInfo CalIntriInfo(
        const uint32_t dtypeSize, const uint32_t count, uint32_t repStride = DEFAULT_BLK_NUM)
    {
        IntriInfo retIntriInfo;
        retIntriInfo.c0Count = GetC0Count(dtypeSize);
        if constexpr (isInt4) {
            retIntriInfo.c0Count = GetC0Size() * INT4_TWO;
        }
        uint32_t repeatCount = repStride * retIntriInfo.c0Count;
        retIntriInfo.repeat = count / repeatCount;
        retIntriInfo.tail = count % repeatCount;
        retIntriInfo.repeatRounding = retIntriInfo.repeat / MAX_REPEAT_TIMES;
        retIntriInfo.repeatRemaining = retIntriInfo.repeat % MAX_REPEAT_TIMES;

        return retIntriInfo;
    }

    template <typename T>
    __aicore__ static inline __ubuf__ T* GetTemporaryBufferAddr(const int32_t bufferOffset, const int32_t bufferSize)
    {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        ASCENDC_ASSERT((bufferOffset % ONE_BLK_SIZE == 0),
                       { KERNEL_LOG(KERNEL_ERROR, "bufferOffset is %d, which must be 32B aligned", bufferOffset); });
        ASCENDC_ASSERT(
            (bufferOffset + bufferSize * sizeof(T) <= ConstDefiner::Instance().bufferInitLen.at(Hardware::UB)), {
                KERNEL_LOG(KERNEL_ERROR, "bufferOffset is %d, bufferSize is %d, which exceed the limit of ub %d",
                    bufferOffset, bufferSize, ConstDefiner::Instance().bufferInitLen.at(Hardware::UB));
            });
        const int32_t maxTempSize = 0x100000;
        ASCENDC_ASSERT((bufferSize < maxTempSize), {
            KERNEL_LOG(KERNEL_ERROR, "bufferSize is %d, which exceed the maxTempSize limits %d", bufferSize,
                maxTempSize);
        });
        T* addr = reinterpret_cast<T*>(ConstDefiner::Instance().hardwareCpuBufferMap.at(Hardware::UB) + bufferOffset);
#else
        (void)bufferSize;
        __ubuf__ T* addr = reinterpret_cast<__ubuf__ T*>(get_imm(0) + bufferOffset);
#endif
        return addr;
    }

    template <typename T> __aicore__ static inline void FreeTemporaryBuffer(__ubuf__ T* addr)
    {
        (void)addr;
    }

#if defined(__NPU_ARCH__) &&                                                                    \
     ((__NPU_ARCH__ == 2201) || (__NPU_ARCH__ == 3002) || (__NPU_ARCH__ == 3102) ||             \
      (__NPU_ARCH__ == 5102) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) ||             \
      (__NPU_ARCH__ == 3113) || (__NPU_ARCH__ == 3101))
    template <typename T>
    __aicore__ static inline __fbuf__ T* GetTemporaryFbBufferAddr(const int32_t bufferOffset, const int32_t bufferSize)
    {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        ASCENDC_ASSERT((bufferOffset % ONE_BLK_SIZE == 0),
                       { KERNEL_LOG(KERNEL_ERROR, "bufferOffset is %d, which must be 32B aligned", bufferOffset); });
        ASCENDC_ASSERT(
            (bufferOffset + bufferSize * sizeof(T) <= ConstDefiner::Instance().bufferInitLen.at(Hardware::FIXBUF)), {
                KERNEL_LOG(KERNEL_ERROR, "bufferOffset is %d, bufferSize is %d, which exceed the limit of fixbuf %d",
                    bufferOffset, bufferSize, ConstDefiner::Instance().bufferInitLen.at(Hardware::FIXBUF));
            });
        T* addr =
            reinterpret_cast<T*>(ConstDefiner::Instance().hardwareCpuBufferMap.at(Hardware::FIXBUF) + bufferOffset);
#else
        (void)bufferSize;
        __fbuf__ T* addr = reinterpret_cast<__fbuf__ T*>(get_imm(0) + bufferOffset);
#endif
        return addr;
    }

    template <typename T> __aicore__ static inline void FreeTemporaryFbBuffer(__fbuf__ T* addr)
    {
        (void)addr;
    }
#endif

    __aicore__ static inline uint64_t GetGMLen(const DataCopyParams& intriParams, const bool& isSrc,
                                               const bool& isMovAlignIntri)
    {
        uint16_t stride = intriParams.dstStride;
        uint16_t burstLenUnit = 32;
        uint16_t strideUnit = 32;
        if (isSrc) {
            stride = intriParams.srcStride;
        }
        if (isMovAlignIntri) {
            burstLenUnit = 1;
            strideUnit = 1;
        }
        if (intriParams.blockLen == 0) {
            return 0;
        }
        uint64_t gmLen = static_cast<uint64_t>(intriParams.blockCount) * intriParams.blockLen * burstLenUnit
                         + (intriParams.blockCount - 1) * stride * strideUnit;
        return gmLen;
    }

    __aicore__ static inline uint64_t GetGMLen(const DataCopyExtParams& intriParams, const bool& isSrc,
                                               const bool& isMovAlignIntri)
    {
        uint16_t stride = intriParams.dstStride;
        uint16_t burstLenUnit = 32;
        uint16_t strideUnit = 32;
        if (isSrc) {
            stride = intriParams.srcStride;
        }
        if (isMovAlignIntri) {
            burstLenUnit = 1;
            strideUnit = 1;
        }
        if (intriParams.blockLen == 0) {
            return 0;
        }
        uint64_t gmLen = static_cast<uint64_t>(intriParams.blockCount) * intriParams.blockLen * burstLenUnit
                         + (intriParams.blockCount - 1) * stride * strideUnit;
        return gmLen;
    }

    __aicore__ static inline uint64_t GetGMLen(const uint64_t& srcEleSize, const Nd2NzParams& intriParams)
    {
        uint64_t gmLen = (static_cast<uint64_t>(intriParams.ndNum) - 1) * srcEleSize * intriParams.srcNdMatrixStride
                         + (intriParams.nValue - 1) * intriParams.srcDValue * srcEleSize
                         + intriParams.dValue * srcEleSize;
        return gmLen;
    }

#if defined(__NPU_ARCH__) &&                                                                                    \
    ((__NPU_ARCH__ == 5102) || (__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) ||    \
     (__NPU_ARCH__ == 3113) || (__NPU_ARCH__ == 3101))
    __aicore__ static inline uint64_t GetGMLen(const uint64_t& srcEleSize, const Dn2NzParams& intriParams)
    {
        uint64_t gmLen = (intriParams.dnNum - 1) * intriParams.srcDnMatrixStride * srcEleSize
                         + intriParams.nValue * srcEleSize
                         + (intriParams.dValue - 1) * intriParams.srcDValue * srcEleSize;
        return gmLen;
    }
#endif

    __aicore__ static inline bool OOMCheckAddrIsOverflow(uintptr_t gmAddrConvert, const uint64_t& gmLen)
    {
        (void)gmAddrConvert;
        (void)gmLen;
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
        uintptr_t inputOutputAddr = 0;
        uint64_t inputOutputLen = 0;

        for (uint64_t index = 0; index < g_oomAddrArange.count; index++) {
            if (g_oomAddrArange.addr[index] == 0 || g_oomAddrArange.len[index] == 0) {
                continue;
            }
            if (g_oomAddrArange.isLevelOnePointer[index] == 0
                && OOMCheckAddrInTensorList(index, gmAddrConvert, inputOutputAddr, inputOutputLen)) {
                break;
            } else {
                inputOutputAddr = g_oomAddrArange.addr[index];
                inputOutputLen = g_oomAddrArange.len[index];
                if (gmAddrConvert >= inputOutputAddr && gmAddrConvert < inputOutputAddr + inputOutputLen) {
                    break;
                }
            }
            if (index == g_oomAddrArange.count - 1) {
                return true;
            }
        }
        if (gmAddrConvert + gmLen > inputOutputAddr + inputOutputLen) {
            return true;
        }
#endif
        (void)gmAddrConvert;
        (void)gmLen;
        return false;
    }

    template <typename T>
    __aicore__ static inline void CheckGmMemOverflow(__gm__ T* gmAddr, const bool& isSrc, const uint64_t& gmLen)
    {
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
        if (gmLen == 0) {
            return;
        }
        if (g_oomAddrArange.count == 0) {
            return;
        }
        uintptr_t gmAddrConvert = reinterpret_cast<uintptr_t>(gmAddr);
        bool status = OOMCheckAddrIsOverflow(gmAddrConvert, gmLen);
#if defined(L2_CACHE_HINT) && (__NPU_ARCH__ == 2201)
        if ASCEND_IS_NOT_AIV {
            if (status) {
                uint64_t oriGmAddr = reinterpret_cast<uint64_t>(gmAddr);
#ifdef __NPU_DEVICE__
                const uint64_t l2Cacheoffset = g_opL2CacheHintCfg.l2Cacheoffset;
                if (oriGmAddr >= l2Cacheoffset) {
                    oriGmAddr -= l2Cacheoffset;
                }
#else // ifndef __NPU_DEVICE__
                if (oriGmAddr >= g_opSystemRunCfg.l2Cacheoffset) {
                    oriGmAddr -= g_opSystemRunCfg.l2Cacheoffset;
                }
#endif // __NPU_DEVICE__
                gmAddrConvert = reinterpret_cast<uintptr_t>(oriGmAddr);
                status = OOMCheckAddrIsOverflow(gmAddrConvert, gmLen);
            }
        }
#endif // L2_CACHE_HINT
        constexpr uint64_t errCode = 0X5A5A0001;
        if (status) {
#if defined(__NPU_ARCH__) &&                                                                                    \
    ((__NPU_ARCH__ == 3002) || (__NPU_ARCH__ == 3102) || (__NPU_ARCH__ == 5102) || (__NPU_ARCH__ == 2103) ||    \
     (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) || (__NPU_ARCH__ == 3113) ||    \
     (__NPU_ARCH__ == 3101))
            trap();
#else
            trap(errCode);
#endif
        }
#endif
    }

    template <typename T>
    __aicore__ static inline void CheckGmMemOverflowNormal(__gm__ T* gmAddr, __gm__ uint8_t* workSpace,
                                                           const bool isSrc, const bool isMovAlignIntri,
                                                           const DataCopyParams& intriParams)
    {
        (void)(workSpace);
        uint64_t gmLen = GetGMLen(intriParams, isSrc, isMovAlignIntri);
        CheckGmMemOverflow(gmAddr, isSrc, gmLen);
    }

    template <typename T>
    __aicore__ static inline void CheckGmMemOverflowNormal(__gm__ T* gmAddr, __gm__ uint8_t* workSpace,
                                                           const bool isSrc, const bool isMovAlignIntri,
                                                           const DataCopyExtParams& intriParams)
    {
        (void)(workSpace);
        uint64_t gmLen = GetGMLen(intriParams, isSrc, isMovAlignIntri);
        CheckGmMemOverflow(gmAddr, isSrc, gmLen);
    }

    template <typename T>
    __aicore__ static inline void CheckGmMemOverflowNd2Nz(__gm__ T* gmAddr, __gm__ uint8_t* workSpace, const bool isSrc,
                                                          const Nd2NzParams& intriParams)
    {
        (void)(workSpace);
        uint64_t srcEleSize = sizeof(T);
        uint64_t gmLen = GetGMLen(srcEleSize, intriParams);
        CheckGmMemOverflow(gmAddr, isSrc, gmLen);
    }

#if defined(__NPU_ARCH__) &&                                                                                    \
    ((__NPU_ARCH__ == 5102) || (__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) ||    \
     (__NPU_ARCH__ == 3113) || (__NPU_ARCH__ == 3101))
    template <typename T>
    __aicore__ static inline void CheckGmMemOverflowDn2Nz(__gm__ T* gmAddr, __gm__ uint8_t* workSpace,
                                                          const bool& isSrc, const Dn2NzParams& intriParams)
    {
        (void)(workSpace);
        uint64_t srcEleSize = sizeof(T);
        uint64_t gmLen = GetGMLen(srcEleSize, intriParams);
        CheckGmMemOverflow(gmAddr, isSrc, gmLen);
    }

    template <typename T, uint8_t dim>
    __aicore__ static inline void CheckGmMemOverflowNddma(__gm__ T* gmAddr, const MultiCopyLoopInfo<dim>& params)
    {
        uint64_t maxOffset = 1;
        for (int32_t i = dim - 1; i >= 0; i--) {
            if (params.loopSize[i] == 0) {
                maxOffset = 0;
                break;
            }
            maxOffset += params.loopSrcStride[i] * (params.loopSize[i] - 1);
        }
        CheckGmMemOverflow(gmAddr, true, maxOffset * sizeof(T));
    }
#endif
};

#ifdef ASCENDC_CPU_DEBUG
enum AtomicType {
    SUM,
    MAX,
    MIN
};
extern bool g_isAtomic;
extern AtomicType g_atomicType;

template <typename T>
__aicore__ inline void DataCopyWithAtomic(__gm__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams)
{
    if (!g_isAtomic) {
        return;
    }
    const uint16_t nBurst = intriParams.blockCount;
    const uint16_t lenBurst = intriParams.blockLen;
    const uint16_t srcStride = intriParams.srcStride;
    const uint16_t dstStride = intriParams.dstStride;
    // new one buffer and do add
    uint32_t dstOffset = 0;
    uint32_t srcOffset = 0;
    const int repeatTime = (lenBurst * ONE_BLK_SIZE + ONE_REPEAT_BYTE_SIZE - 1) / ONE_REPEAT_BYTE_SIZE;
    for (int index = 0; index < nBurst; ++index) {
        for (int indexJ = 0; indexJ < lenBurst * ONE_BLK_SIZE / sizeof(T); ++indexJ) {
            if (g_atomicType == SUM) {
                *(static_cast<T*>(src) + srcOffset + indexJ) =
                    *(static_cast<T*>(dst) + dstOffset + indexJ) + *(static_cast<T*>(src) + srcOffset + indexJ);
            } else if (g_atomicType == MAX) {
                *(static_cast<T*>(src) + srcOffset + indexJ) = std::max(*(static_cast<T*>(dst) + dstOffset + indexJ),
                    *(static_cast<T*>(src) + srcOffset + indexJ));
            } else {
                *(static_cast<T*>(src) + srcOffset + indexJ) = std::min(*(static_cast<T*>(dst) + dstOffset + indexJ),
                    *(static_cast<T*>(src) + srcOffset + indexJ));
            }
        }
        dstOffset += ((lenBurst + dstStride) * ONE_BLK_SIZE) / sizeof(T);
        srcOffset += ((lenBurst + srcStride) * ONE_BLK_SIZE) / sizeof(T);
    }
}

template <typename T>
__aicore__ inline void DataCopyWithAtomicCom(__gm__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams)
{
    const uint16_t nBurst = intriParams.blockCount;
    const uint16_t lenBurst = intriParams.blockLen;
    const uint16_t srcStride = intriParams.srcStride;
    const uint16_t dstStride = intriParams.dstStride;
    const uint16_t halfSize = sizeof(T);
    // new one buffer and do add
    uint32_t dstOffset = 0;
    uint32_t srcOffset = 0;
    const int repeatTime = (lenBurst * ONE_BLK_SIZE) / ONE_REPEAT_BYTE_SIZE;
    const int countInRepeat = (ONE_REPEAT_BYTE_SIZE / halfSize);
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 1001) || (__NPU_ARCH__ == 2002) || (__NPU_ARCH__ == 2201))
    const int tail = lenBurst * ONE_BLK_SIZE / halfSize - repeatTime * countInRepeat;
#endif
    for (int index = 0; index < nBurst; ++index) {
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 1001) || (__NPU_ARCH__ == 2002) || (__NPU_ARCH__ == 2201))
        __ubuf__ T* dstAddr = static_cast<__ubuf__ T*>(src) + srcOffset;
        __ubuf__ T* src0Addr = static_cast<__ubuf__ T*>(dst) + dstOffset;
        __ubuf__ T* src1Addr = static_cast<__ubuf__ T*>(src) + srcOffset;
        if (repeatTime > 0) {
            AscendCUtils::SetMask<T>(countInRepeat);
            if (g_atomicType == SUM) {
                vadd(static_cast<T*>(dstAddr), static_cast<T*>(src0Addr), static_cast<T*>(src1Addr), repeatTime, 1, 1, 1,
                    DEFAULT_BLK_NUM, DEFAULT_BLK_NUM, DEFAULT_BLK_NUM);
            } else if (g_atomicType == MAX) {
                vmax(static_cast<T*>(dstAddr), static_cast<T*>(src0Addr), static_cast<T*>(src1Addr), repeatTime, 1, 1, 1,
                    DEFAULT_BLK_NUM, DEFAULT_BLK_NUM, DEFAULT_BLK_NUM);
            } else {
                vmin(static_cast<T*>(dstAddr), static_cast<T*>(src0Addr), static_cast<T*>(src1Addr), repeatTime, 1, 1, 1,
                    DEFAULT_BLK_NUM, DEFAULT_BLK_NUM, DEFAULT_BLK_NUM);
            }
            AscendCUtils::ResetMask();
        }
        if (tail != 0) {
            dstAddr = dstAddr + repeatTime * countInRepeat;
            src0Addr = src0Addr + repeatTime * countInRepeat;
            src1Addr = src1Addr + repeatTime * countInRepeat;
            AscendCUtils::SetMask<T>(tail);
            if (g_atomicType == SUM) {
                vadd(static_cast<T*>(dstAddr), static_cast<T*>(src0Addr), static_cast<T*>(src1Addr), 1, 1, 1, 1,
                    DEFAULT_BLK_NUM, DEFAULT_BLK_NUM, DEFAULT_BLK_NUM);
            } else if (g_atomicType == MAX) {
                vmax(static_cast<T*>(dstAddr), static_cast<T*>(src0Addr), static_cast<T*>(src1Addr), 1, 1, 1, 1,
                    DEFAULT_BLK_NUM, DEFAULT_BLK_NUM, DEFAULT_BLK_NUM);
            } else {
                vmin(static_cast<T*>(dstAddr), static_cast<T*>(src0Addr), static_cast<T*>(src1Addr), 1, 1, 1, 1,
                    DEFAULT_BLK_NUM, DEFAULT_BLK_NUM, DEFAULT_BLK_NUM);
            }
            AscendCUtils::ResetMask();
        }
#endif
        dstOffset += ((lenBurst + dstStride) * ONE_BLK_SIZE) / halfSize;
        srcOffset += ((lenBurst + srcStride) * ONE_BLK_SIZE) / halfSize;
    }
}

__aicore__ inline void DataCopyWithAtomic(__gm__ half* dst, __ubuf__ half* src, const DataCopyParams& intriParams)
{
    if (!g_isAtomic) {
        return;
    }
    DataCopyWithAtomicCom(dst, src, intriParams);
}
__aicore__ inline void DataCopyWithAtomic(__gm__ float* dst, __ubuf__ float* src, const DataCopyParams& intriParams)
{
    if (!g_isAtomic) {
        return;
    }
    DataCopyWithAtomicCom(dst, src, intriParams);
}

#if (__NPU_ARCH__ == 3002)
__aicore__ inline void DataCopyWithAtomic(__gm__ int16_t* dst, __ubuf__ int16_t* src, const DataCopyParams& intriParams)
{
    if (!g_isAtomic) {
        return;
    }
    DataCopyWithAtomicCom(dst, src, intriParams);
}

__aicore__ inline void DataCopyWithAtomic(__gm__ int32_t* dst, __ubuf__ int32_t* src, const DataCopyParams& intriParams)
{
    if (!g_isAtomic) {
        return;
    }
    DataCopyWithAtomicCom(dst, src, intriParams);
}
#endif
#endif // ASCENDC_CPU_DEBUG

/***************内部定义time stamp id**************************
定义值范围: 0x000 - 0xfff

time stamp id按块分组，快说明如下:
TIME_STAMP_WRAP: NPU套壳函数中的时间戳打点
TIME_STAMP_TPIPE/BUFFER: TPIPE、BUFFER中的时间戳打点
TIME_STAMP_MATMUL: MATMUL相关时间戳打点
TIME_STAMP_TILING_DATA: TILING DATA模块时间戳打点
TIME_STAMP_MC2_START/END: MC2模块使用打点id范围

TimeStampId更新原则：每个分组新增ID不可改变原有定义的ID值！

***************************************************************/
enum class TimeStampId : uint32_t {
    TIME_STAMP_WRAP_FIRST = 0x000,
    TIME_STAMP_WRAP_MC2_CTX,
    TIME_STAMP_WRAP_WK_SPACE,
    TIME_STAMP_WRAP_INIT_DUMP,
    TIME_STAMP_WRAP_FFTS_ADDR,
    TIME_STAMP_WRAP_CLEAR_WK_SPAC,

    TIME_STAMP_TPIPE = 0x030,
    TIME_STAMP_BUFFER,

    TIME_STAMP_MATMUL_SERVER = 0x060,
    TIME_STAMP_MATMUL_SERVER_INIT,
    TIME_STAMP_MATMUL_SERVER_OBJ,
    TIME_STAMP_MATMUL_MATRIX_KFC,
    TIME_STAMP_MATMUL_CLIENT_KFC,
    TIME_STAMP_MATMUL_WAIT_EVE,
    TIME_STAMP_MATMUL_OBJ,

    TIME_STAMP_TILING_DATA = 0x090,
    TIME_STAMP_TILING_DATA_STRUCT,
    TIME_STAMP_TILING_DATA_MEMBER,

    // MC2 :0x1000-0x1fff
    TIME_STAMP_MC2_START = 0x1000,
    TIME_STAMP_MC2_END = 0x1fff,

    TIME_STAMP_MAX = 0xffff,
};
} // namespace AscendC
#endif // ASCENDC_MODULE_UTILS_H
