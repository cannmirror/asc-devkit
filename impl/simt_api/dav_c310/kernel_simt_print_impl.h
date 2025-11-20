/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ASCENDC_MODULE_SIMT_PRINT_IMPL_H
#define ASCENDC_MODULE_SIMT_PRINT_IMPL_H

#include "kernel_simt_constant.h"

namespace AscendC {
namespace Simt {

#ifdef ASCENDC_DUMP

static __gm__ uint8_t* __gm__ g_simtDumpWorkspaceReserved;

__aicore__ inline void SetSimtDumpWorkspace(GM_ADDR workspace)
{
    g_simtDumpWorkspaceReserved = workspace - ConstantsInternal::SIMT_DUMP_BLOCK_NUM *
                                  ConstantsInternal::SIMT_MAX_THREAD_NUM * ConstantsInternal::SIMT_DUMP_SIZE;
    DataSyncBarrier<MemDsbT::DDR>();
}

#ifndef ASCENDC_CPU_DEBUG

__aicore__ inline uint32_t GetArgsNum()
{
    return 0;
}

template <typename T, typename... Args>
__aicore__ inline uint32_t GetArgsNum(T scalar, Args... args)
{
    return 1 + Simt::GetArgsNum(args...);
}

__aicore__ inline uint32_t GetStringLength(const __gm__ char *s)
{
    uint32_t i = 0;
    while (*(s + i) != '\0') {
        i++;
    }
    return i + 1;
}

__aicore__ inline uint32_t GetArgsSize()
{
    return 0;
}

template <typename... Args>
__aicore__ inline uint32_t GetArgsSize(Args &&...args);

template <typename... Args>
__aicore__ inline uint32_t GetArgsSizeImpl(const __gm__ char *s, Args &&...args)
{
    uint32_t strLen = Simt::GetStringLength(s);
    uint32_t strParamSize = ONE_PARAM_SIZE + strLen;
    return strParamSize + Simt::GetArgsSize(args...);
}

template <typename T, typename... Args>
__aicore__ inline uint32_t GetArgsSizeImpl(T scalar, Args &&...args)
{
    return ONE_PARAM_SIZE + Simt::GetArgsSize(args...);
}

template <typename... Args>
__aicore__ inline uint32_t GetArgsSize(Args &&...args)
{
    return Simt::GetArgsSizeImpl(args...);
}

template <typename... Args>
__aicore__ inline uint32_t GetParamSize(const __gm__ char *fmt, Args &&...args)
{
    uint32_t fmtSize = Simt::GetStringLength(fmt);
    uint32_t argsSize = Simt::GetArgsSize(args...);
    return fmtSize + argsSize + ONE_PARAM_SIZE;
}

__aicore__ inline void WriteTLHead(DumpType printType, __gm__ uint8_t *tlv, uint32_t valueSize)
{
    *((__gm__ uint32_t *)tlv) = static_cast<uint32_t>(printType);
    *((__gm__ uint32_t *)tlv + 1) = valueSize;
}

__aicore__ inline void WriteString(__gm__ uint8_t *paramAddr, uint32_t paramIdx, const __gm__ char *s, uint32_t &offset)
{
    __gm__ uint64_t *stringAddr = reinterpret_cast<__gm__ uint64_t *>(paramAddr) + paramIdx;
    __gm__ uint64_t *dstStrAddr = reinterpret_cast<__gm__ uint64_t *>(paramAddr + offset);

    // write string value offset
    *((__gm__ uint64_t *)stringAddr) = static_cast<uint64_t>(offset - ONE_PARAM_SIZE * paramIdx);

    // write string content
    __gm__ char *d = (__gm__ char *)(dstStrAddr);
    uint32_t strLen = Simt::GetStringLength(s);

    for (uint32_t i = 0; i < strLen; i++) {
        *(d + i) = *(s + i);
    }
    offset += strLen;
}

template <typename T>
__aicore__ inline void WriteScalar(__gm__ uint8_t *paramAddr, uint32_t paramIdx, T scalar)
{
    __gm__ uint64_t *scalarAddr = (__gm__ uint64_t *)paramAddr + paramIdx;
    *scalarAddr = 0;

    if constexpr (SupportType<T, half, float>()) {
        *((__gm__ float *)scalarAddr) = static_cast<float>(scalar);
    } else if constexpr (SupportType<T, double>()) {
        *((__gm__ double *)scalarAddr) = static_cast<double>(scalar);
    } else if constexpr (std::is_signed<T>::value) {
        *((__gm__ int64_t *)scalarAddr) = static_cast<int64_t>(scalar);
    } else if constexpr (std::is_unsigned<T>::value) {
        *((__gm__ uint64_t *)scalarAddr) = static_cast<uint64_t>(scalar);
    } else if constexpr (SupportType<T, bfloat16_t, fp8_e5m2_t, fp8_e8m0_t, fp8_e4m3fn_t, hifloat8_t>()) {
        *((__gm__ float *)scalarAddr) = ToFloat(scalar);
    } else if constexpr (std::is_pointer<T>::value) {
        *((__gm__ uint64_t *)scalarAddr) = (uintptr_t)scalar;
    } else if constexpr (std::is_enum<T>::value) {
        *((__gm__ uint64_t *)scalarAddr) = static_cast<uint64_t>(scalar);
    }
}

__aicore__ inline void SetParam(__gm__ uint8_t *paramAddr, uint32_t paramIdx, uint32_t &offset)
{
    return;
}

template <typename... Args>
__aicore__ inline void SetParam(__gm__ uint8_t *paramAddr, uint32_t paramIdx, uint32_t &offset, Args &&...args);

template <typename... Args>
__aicore__ inline void SetParamImpl(__gm__ uint8_t *paramAddr, uint32_t paramIdx, uint32_t &offset,
                                    const __gm__ char *s, Args &&...args)
{
    Simt::WriteString(paramAddr, paramIdx, s, offset);
    Simt::SetParam(paramAddr, paramIdx + 1, offset, args...);
}

template <typename T, typename... Args>
__aicore__ inline void SetParamImpl(__gm__ uint8_t *paramAddr, uint32_t paramIdx, uint32_t &offset, T scalar,
                                    Args &&...args)
{
    Simt::WriteScalar(paramAddr, paramIdx, scalar);
    Simt::SetParam(paramAddr, paramIdx + 1, offset, args...);
}

template <typename... Args>
__aicore__ inline void SetParam(__gm__ uint8_t *paramAddr, uint32_t paramIdx, uint32_t &offset, Args &&...args)
{
    Simt::SetParamImpl(paramAddr, paramIdx, offset, args...);
}

__aicore__ inline uint32_t GetDumpBlockIdx()
{
    return bisheng::cce::simt::get_block_idx();
}

__aicore__ inline __gm__ BlockInfo *GetBlockInfo()
{
    uint64_t dumpWorkspaceStart = reinterpret_cast<uint64_t>(g_simtDumpWorkspaceReserved);

    uint32_t blockIdx = Simt::GetDumpBlockIdx();
    uint64_t blockDumpWorkspaceStart =
        dumpWorkspaceStart + blockIdx * ConstantsInternal::SIMT_MAX_THREAD_NUM * ConstantsInternal::SIMT_DUMP_SIZE;

    uint32_t threadId = GetThreadIdxImpl<0>() * GetThreadNumImpl<1>() * GetThreadNumImpl<2>() +
                        GetThreadIdxImpl<1>() * GetThreadNumImpl<2>() + GetThreadIdxImpl<2>();
    return (__gm__ BlockInfo *)(blockDumpWorkspaceStart + threadId * ConstantsInternal::SIMT_DUMP_SIZE);
}

__aicore__ inline void UpdateBlockInfo(uint32_t tlvSize)
{
    __gm__ BlockInfo *blockInfo = Simt::GetBlockInfo();
    uint32_t remainSize = blockInfo->dumpOffset;
    uint64_t lastDumpAddr = blockInfo->dumpAddr;

    __gm__ uint8_t *blockInfoStart = (__gm__ uint8_t *)blockInfo;
    *((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_DUMPOFFSET_POS) = remainSize - tlvSize;
    *((__gm__ uint64_t *)((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_DUMP_ADDR)) = lastDumpAddr + tlvSize;
}

__aicore__ inline void InitDumpForThread()
{
    if (g_simtDumpWorkspaceReserved == nullptr) {
        return;
    }

    uint32_t blockIdx = Simt::GetDumpBlockIdx();
    if (blockIdx >= ConstantsInternal::SIMT_DUMP_BLOCK_NUM) {
        return;
    }

    __gm__ BlockInfo *blockInfo = Simt::GetBlockInfo();
    uint32_t blockInfoLen = sizeof(BlockInfo) + sizeof(SimtDumpMeta);
    uint64_t blockInfoStart = reinterpret_cast<uint64_t>(blockInfo);

    uint32_t threadId = GetThreadIdxImpl<0>() * GetThreadNumImpl<1>() * GetThreadNumImpl<2>() +
                        GetThreadIdxImpl<1>() * GetThreadNumImpl<2>() + GetThreadIdxImpl<2>();

    *((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_LEN_POS) = ConstantsInternal::SIMT_DUMP_SIZE;
    *((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_CORE_POS) = blockIdx;
    *((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_BLOCKNUM_POS) = GetBlockNumImpl();
    *((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_DUMPOFFSET_POS) = ConstantsInternal::SIMT_DUMP_SIZE - blockInfoLen;
    *((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_MAGIC_POS) = BLOCK_INFO_MAGIC_NUM;
    *((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_RSV_POS) = 0;
    *((__gm__ uint64_t *)((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_DUMP_ADDR)) = blockInfoStart + blockInfoLen;

    blockInfoStart = blockInfoStart + sizeof(BlockInfo);
    *(__gm__ uint32_t *)((__gm__ uint8_t *)blockInfoStart + DUMP_META_TYPE_POS) =
        static_cast<uint32_t>(DumpType::DUMP_META);
    *(__gm__ uint32_t *)((__gm__ uint8_t *)blockInfoStart + DUMP_META_LEN_POS) = 8; // 8: simt thread(uint32_t) id and rsv(uint32_t)
    *(__gm__ uint32_t *)((__gm__ uint8_t *)blockInfoStart + DUMP_META_SIMT_THREAD_ID_POS) = threadId;
    *(__gm__ uint32_t *)((__gm__ uint8_t *)blockInfoStart + DUMP_META_RSV_POS) = 0;
}

template <class... Args>
__aicore__ inline void PrintfImpl(const __gm__ char *fmt, Args &&...args)
{
    if (g_simtDumpWorkspaceReserved == nullptr) {
        return;
    }

    uint32_t blockIdx = Simt::GetDumpBlockIdx();
    if (blockIdx >= ConstantsInternal::SIMT_DUMP_BLOCK_NUM) {
        return;
    }

    __gm__ BlockInfo *blockInfo = Simt::GetBlockInfo();
    if (blockInfo->magic != BLOCK_INFO_MAGIC_NUM) {
        InitDumpForThread();
    }
    uint32_t remainSize = blockInfo->dumpOffset;
    uint64_t dumpAddr = blockInfo->dumpAddr;

    uint32_t paramSize = Simt::GetParamSize(fmt, args...);
    uint32_t paramNum = Simt::GetArgsNum(args...) + 1;
    // ONE_PARAM_SIZE(8 byte) algin
    paramSize = (paramSize + ONE_PARAM_SIZE - 1) & (~(ONE_PARAM_SIZE - 1));

    // sizeof(DumpType + DumpLength) = ONE_PARAM_SIZE(8 byte)
    uint32_t tlvSize = paramSize + ONE_PARAM_SIZE;
    if (tlvSize > remainSize) {
        *((__gm__ uint32_t *)blockInfo + BLOCK_INFO_RSV_POS) = DUMP_EXC_FLAG;
        return;
    }

    __gm__ uint8_t *tlvAddr = (__gm__ uint8_t *)dumpAddr;
    Simt::WriteTLHead(DumpType::DUMP_SIMT, tlvAddr, paramSize);

    // sizeof(DumpType + DumpLength) = ONE_PARAM_SIZE(8 byte)
    __gm__ uint8_t *paramAddr = tlvAddr + ONE_PARAM_SIZE;
    uint32_t offset = paramNum * ONE_PARAM_SIZE;
    Simt::WriteString(paramAddr, 0, fmt, offset);
    uint32_t paramIdx = 1;
    Simt::SetParam(paramAddr, paramIdx, offset, args...);

    Simt::UpdateBlockInfo(tlvSize);
}

#endif
#endif

}  // namespace Simt
}  // namespace AscendC

#endif  // ASCENDC_MODULE_SIMT_PRINT_IMPL_H
