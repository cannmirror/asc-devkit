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
 * \file kernel_operator_dump_tensor_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_DUMP_TENSOR_IMPL_H
#define ASCENDC_MODULE_OPERATOR_DUMP_TENSOR_IMPL_H

#include "kernel_tpipe_impl.h"
#include "kernel_operator_common_impl.h"
#include "kernel_operator_data_copy_impl.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_operator_sys_var_intf.h"
#include "kernel_struct_data_copy.h"
#include "kernel_struct_fixpipe.h"


namespace AscendC {
__BLOCK_LOCAL__ __inline__ __gm__ uint8_t* g_dumpWorkspaceReserved;

template <typename T>
__aicore__ constexpr inline Internal::DumpTensorDataType GetTensorDataType();

template <typename T> __aicore__ inline uint32_t GetDataType(T data)
{
    return static_cast<uint32_t>(GetTensorDataType<T>());
}

__aicore__ inline uint8_t GetDumpBlockIdx()
{
    if (ASCEND_IS_AIV) {
        return GetBlockIdxImpl();
    } else {
        return GetBlockIdxImpl() + AIV_CORE_NUM;
    }
}


__aicore__ inline int64_t GetBlockNum();
__aicore__ inline void InitDumpImpl(bool mixFlag, uint32_t gmLen)
{
    uint64_t firstTimeStamp = static_cast<uint64_t>(GetSystemCycle());
    uint32_t totalBlockNum;

    if (g_dumpWorkspaceReserved == nullptr) {
        ASCENDC_ASSERT((false),
            { KERNEL_LOG(KERNEL_ERROR, "init dump get nullptr system workspace ptr"); });
        return;
    }
    uint64_t dumpWorkspaceStart = reinterpret_cast<uint64_t>(g_dumpWorkspaceReserved) - DUMP_WORKSPACE_SIZE;

    if (mixFlag == true) {
        totalBlockNum = GetBlockNum() * (1 + MIX_NUM);
    } else {
        totalBlockNum = GetBlockNum();
    }
    uint32_t blockDumpSize = DUMP_UINTSIZE; // DUMP_UINTSIZE is 1M

    uint32_t blockDim = GetDumpBlockIdx();
    if (blockDim >= DUMP_CORE_COUNT) {
        return;
    }
#ifdef ASCENDC_TIME_STAMP_ON
    uint32_t blkInfoLen = sizeof(BlockInfo) + sizeof(DumpMeta) + sizeof(DumpTimeStamp);
#else
    uint32_t blkInfoLen = sizeof(BlockInfo) + sizeof(DumpMeta);
#endif
    uint64_t blockInfoStart = dumpWorkspaceStart + blockDim * DUMP_UINTSIZE;
    *((__gm__ uint32_t*)blockInfoStart + BLOCK_INFO_LEN_POS) = blockDumpSize;
    *((__gm__ uint32_t*)blockInfoStart + BLOCK_INFO_CORE_POS) = blockDim;
    *((__gm__ uint32_t*)blockInfoStart + BLOCK_INFO_BLOCKNUM_POS) = totalBlockNum;
    *((__gm__ uint32_t*)blockInfoStart + BLOCK_INFO_DUMPOFFSET_POS) = blockDumpSize - blkInfoLen;
    *((__gm__ uint32_t*)blockInfoStart + BLOCK_INFO_MAGIC_POS) = 0x5aa5bccd;
    *((__gm__ uint32_t*)blockInfoStart + BLOCK_INFO_RSV_POS) = 0;
    *((__gm__ uint64_t*)((__gm__ uint32_t*)blockInfoStart + BLOCK_INFO_DUMP_ADDR)) = blockInfoStart + blkInfoLen;
    // add DUMP_META info
    blockInfoStart = blockInfoStart + sizeof(BlockInfo);
    *(__gm__ uint32_t*)((__gm__ uint8_t*)blockInfoStart + DUMP_META_TYPE_POS) =
        static_cast<uint32_t>(DumpType::DUMP_META);
    *(__gm__ uint32_t*)((__gm__ uint8_t*)blockInfoStart + DUMP_META_LEN_POS) = 8;
    *(__gm__ uint16_t*)((__gm__ uint8_t*)blockInfoStart + DUMP_META_BLOCK_DIM_POS) =
        static_cast<uint16_t>(GetBlockNum());
    *(__gm__ uint8_t*)((__gm__ uint8_t*)blockInfoStart + DUMP_META_CORE_TYPE_POS) =
        static_cast<uint8_t>(g_coreType);
    *(__gm__ uint8_t*)((__gm__ uint8_t*)blockInfoStart + DUMP_META_TASK_RATION) =
        static_cast<uint8_t>(mixFlag);
    *((__gm__ uint32_t*)blockInfoStart + DUMP_META_RSV_POS) = 0;
#ifdef ASCENDC_TIME_STAMP_ON
    blockInfoStart = blockInfoStart + sizeof(DumpMeta);
    // // WriteTLHead
    *((__gm__ uint32_t *)blockInfoStart) = static_cast<uint32_t>(DumpType::DUMP_TIME_STAMP);
    *((__gm__ uint32_t *)blockInfoStart + DUMP_TIME_STAMP_LEN_POS) = DUMP_TIME_STAMP_LEN;
    // write value
    *((__gm__ uint32_t *)blockInfoStart + DUMP_TIME_STAMP_ID_POS) = 0;
    *((__gm__ uint64_t *)((__gm__ uint32_t *)blockInfoStart + DUMP_TIME_STAMP_CYCLE_POS)) = firstTimeStamp;
    *((__gm__ uint64_t *)((__gm__ uint32_t *)blockInfoStart + DUMP_TIME_STAMP_PTR_POS)) = 0;
#endif
    dcci((__gm__ uint64_t*)blockInfoStart, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
}
__aicore__ inline DataCopyParams GetDataCopyParamImpl(uint32_t offset)
{
    DataCopyParams repeatParams;
    repeatParams.blockCount = 1;
    repeatParams.blockLen = offset / ONE_BLK_SIZE;
    repeatParams.srcStride = 0;
    repeatParams.dstStride = 0;
    return repeatParams;
}

__aicore__ inline void GetMatCopyParam(
    uint32_t dumpSize, uint16_t& n, uint16_t& m, uint16_t& dstStrideDstD, uint16_t& srcStride);

template <typename T>
__aicore__ inline void DumpTensorL0C2GMImpl(const LocalTensor<T>& src, __gm__ BlockInfo* ptr, uint32_t dumpSize)
{
    uint16_t n, m, dstStrideDstD, srcStride;
    GetMatCopyParam(dumpSize, n, m, dstStrideDstD, srcStride);

    copy_matrix_cc_to_gm((__gm__ float *)(ptr->dumpAddr), (__cc__ float *)(src.GetPhyAddr()),
        0, n, m, dstStrideDstD, srcStride, 0, QuantMode_t::NoQuant,
        static_cast<uint8_t>(false), false, false);
}

template <typename T>
__aicore__ inline uint32_t CheckValidPosition(const LocalTensor<T>& src)
{
    // set the head struct value
    uint32_t position = 0;
    if ((Hardware)GetPhyType((TPosition)src.GetPosition()) == Hardware::UB) {
        position = static_cast<uint32_t>(AscendC::Hardware::UB);
        return position;
    } else if ((Hardware)GetPhyType((TPosition)src.GetPosition()) == Hardware::L1) {
        position = static_cast<uint32_t>(AscendC::Hardware::L1);
        return position;
    } else if ((Hardware)GetPhyType((TPosition)src.GetPosition()) == Hardware::L0C) {
        position = static_cast<uint32_t>(AscendC::Hardware::L0C);
        return position;
    } else {
        return false;
    }
}

__aicore__ inline void WriteDumpShapeInfo(const ShapeInfo &shapeInfo)
{
    uint8_t core = GetDumpBlockIdx();
    if (core >= DUMP_CORE_COUNT) {
        return;
    }
    uint32_t valueSize = sizeof(DumpShapeMessageHead);
    uint64_t dumpWorkspaceStart = reinterpret_cast<uint64_t>(g_dumpWorkspaceReserved) - DUMP_WORKSPACE_SIZE;
    __gm__ BlockInfo* ptr = (__gm__ BlockInfo*)(dumpWorkspaceStart + DUMP_UINTSIZE * core);
    uint32_t tlvSize = valueSize + DUMP_SHAPE_MESSAGE_TL_LEN;
    if (ptr->dumpOffset < tlvSize) {
        ASCENDC_ASSERT((false), {
            KERNEL_LOG(KERNEL_ERROR,
                "Space not enough! need %u Bytes, current remained dump space is %u Bytes",
                tlvSize,
                ptr->dumpOffset);
        });
        *((__gm__ uint32_t*)ptr + BLOCK_INFO_RSV_POS) = DUMP_EXC_FLAG;
        dcci((__gm__ uint64_t*)ptr, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
        return;
    }
    *((__gm__ uint32_t*)ptr->dumpAddr + DUMP_SHAPE_MESSAGE_HEAD_TYPE_POS) = static_cast<uint32_t>(DumpType::DUMP_SHAPE);
    *((__gm__ uint32_t*)ptr->dumpAddr + DUMP_SHAPE_MESSAGE_HEAD_LEN_POS) = valueSize;
    *((__gm__ uint32_t*)ptr->dumpAddr + DUMP_SHAPE_MESSAGE_HEAD_DIM_POS) = shapeInfo.shapeDim;
    for (uint32_t idx = 0; idx < shapeInfo.shapeDim && idx < K_MAX_SHAPE_DIM; idx++) {
        *((__gm__ uint32_t*)ptr->dumpAddr + DUMP_SHAPE_MESSAGE_HEAD_SHAPE_START_POS + idx) = shapeInfo.shape[idx];
    }
    *((__gm__ uint32_t*)ptr->dumpAddr + DUMP_SHAPE_MESSAGE_HEAD_RSV_POS) = 0;
    // update block info
    ptr->dumpAddr += tlvSize;
    ptr->dumpOffset -= tlvSize;
    dcci((__gm__ uint64_t*)ptr, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
}

__aicore__ inline void WriteFifoShapeInfo(const ShapeInfo &shapeInfo);

__aicore__ inline void DumpShapeImpl(const ShapeInfo &shapeInfo)
{
    if (g_sysPrintFifoSpace != nullptr) {
        WriteFifoShapeInfo(shapeInfo);
    } else {
        WriteDumpShapeInfo(shapeInfo);
    }
}

template <typename T>
__aicore__ inline void DumpTensorLocal2GMEntityImpl(const LocalTensor<T>& src, uint32_t desc, uint32_t dumpSize)
{
    uint32_t position = CheckValidPosition(src);
    // set the head struct value
    if (position == 0) {
        ASCENDC_ASSERT((false),
                   { KERNEL_LOG(KERNEL_ERROR, "dump tensor only support dump tensor from local to gm"); });
        return;
    }

    T data;
    uint8_t core = GetDumpBlockIdx();
    if (core >= DUMP_CORE_COUNT) {
        return;
    }
    uint32_t offset = dumpSize * sizeof(T);
    uint32_t padOffset = AlignUp(offset, ONE_BLK_SIZE);

    uint64_t dumpWorkspaceStart = reinterpret_cast<uint64_t>(g_dumpWorkspaceReserved) - DUMP_WORKSPACE_SIZE;

    __gm__ BlockInfo* ptr = (__gm__ BlockInfo*)(dumpWorkspaceStart + DUMP_UINTSIZE * core);
    if (ptr->dumpOffset < (padOffset + sizeof(DumpMessageHead))) {
        ASCENDC_ASSERT((false), {
            KERNEL_LOG(KERNEL_ERROR,
                "Space not enough! need %u Bytes, current remained dump space is %u Bytes",
                (padOffset + sizeof(DumpMessageHead)),
                ptr->dumpOffset);
        });
        *((__gm__ uint32_t*)ptr + BLOCK_INFO_RSV_POS) = DUMP_EXC_FLAG;
        dcci((__gm__ uint64_t*)ptr, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
        return;
    }

    *((__gm__ uint32_t*)ptr->dumpAddr + DUMP_MESSAGE_HEAD_TYPE_POS) = static_cast<uint32_t>(DumpType::DUMP_TENSOR);
    *((__gm__ uint32_t*)ptr->dumpAddr + DUMP_MESSAGE_HEAD_LEN_POS) = padOffset + DUMP_MSG_HEAD_SIZE;
    *((__gm__ uint32_t*)ptr->dumpAddr + DUMP_MESSAGE_HEAD_ADDR_POS) =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(src.GetPhyAddr()));
    *((__gm__ uint32_t*)ptr->dumpAddr + DUMP_MESSAGE_HEAD_DATA_TYPE_POS) = GetDataType(data);
    *((__gm__ uint32_t*)ptr->dumpAddr + DUMP_MESSAGE_HEAD_DESC_POS) = desc;
    *((__gm__ uint32_t*)ptr->dumpAddr + DUMP_MESSAGE_HEAD_BUFFERID_POS) = 0;
    *((__gm__ uint32_t*)ptr->dumpAddr + DUMP_MESSAGE_HEAD_POSITION_POS) = position;
    *((__gm__ uint32_t*)ptr->dumpAddr + DUMP_MESSAGE_HEAD_DUMP_SIZE_POS) = dumpSize;
    // update block info
    ptr->dumpAddr += sizeof(DumpMessageHead);
    ptr->dumpOffset -= sizeof(DumpMessageHead);
    dcci((__gm__ uint64_t*)ptr, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
    DataCopyParams repeatParams = GetDataCopyParamImpl(padOffset);
    const Hardware srcHWPos = GetPhyType((TPosition)src.GetPosition());

    PipeBarrier<PIPE_ALL>();
    if (srcHWPos == Hardware::UB) {
        DataCopyUB2GMImpl((__gm__ T*)(ptr->dumpAddr), (__ubuf__ T*)src.GetPhyAddr(), repeatParams); // UB to GM
    } else if (srcHWPos == Hardware::L1) {
        DataCopyL12GMImpl((__gm__ T*)(ptr->dumpAddr), (__cbuf__ T*)src.GetPhyAddr(), repeatParams); // L1 to GM
    } else if (srcHWPos == Hardware::L0C) {
        if ASCEND_IS_NOT_AIC {
            return;
        }
        DumpTensorL0C2GMImpl(src, ptr, dumpSize);
    }
    PipeBarrier<PIPE_ALL>();
    ptr->dumpOffset -= padOffset;
    ptr->dumpAddr += padOffset;
    dcci((__gm__ uint64_t*)ptr, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
}


template <template<typename> class Tensor, typename T>
__aicore__ inline void DumpTensorFifoImpl(const Tensor<T>& src, uint32_t desc, uint32_t dumpSize);

template <typename T>
__aicore__ inline void DumpTensorLocal2GMImpl(const LocalTensor<T>& src, uint32_t desc, uint32_t dumpSize)
{
    uint64_t ctrlValue = get_ctrl();
    set_atomic_none();
    if (g_sysPrintFifoSpace != nullptr) {
        DumpTensorFifoImpl(src, desc, dumpSize);
    } else {
        DumpTensorLocal2GMEntityImpl(src, desc, dumpSize);
    }
    set_ctrl(ctrlValue);
}

__aicore__ inline uint32_t GetLoopCount(uint32_t offset)
{
    uint32_t loopCount = 0;
    if (offset % ONE_DUMP_BACKUP_SIZE != 0) {
        loopCount = 1 + offset / ONE_DUMP_BACKUP_SIZE;
    } else {
        loopCount = offset / ONE_DUMP_BACKUP_SIZE;
    }
    return loopCount;
}

template <typename T>
__aicore__ inline void InitTmpTensor(LocalTensor<T>& tmp, uint8_t quePos)
{
    TBuffAddr tbufTmpLocal;
    tbufTmpLocal.logicPos = quePos;
    tmp.SetAddr(tbufTmpLocal);
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    tmp.address_.absAddr = reinterpret_cast<uint8_t *>(ConstDefiner::Instance().cpuUB);
#else
    tmp.address_.bufferAddr = get_imm(0);
#endif
    tmp.address_.dataLen = ONE_DUMP_BACKUP_SIZE;
}
__aicore__ inline bool CheckDumpValid(uint32_t padOffset)
{
    uint8_t core = GetDumpBlockIdx();
    if (core >= DUMP_CORE_COUNT) {
        return false;
    }
    uint64_t dumpWorkspaceStart = reinterpret_cast<uint64_t>(g_dumpWorkspaceReserved) - DUMP_WORKSPACE_SIZE;
    if (reinterpret_cast<uint64_t>(g_dumpWorkspaceReserved) < DUMP_WORKSPACE_SIZE) {
        KERNEL_LOG(KERNEL_ERROR, "DumpWorkSpace addr is %lu, which must be larger than 75M", reinterpret_cast<uint64_t>(g_dumpWorkspaceReserved));
        return false;
    }
    __gm__ BlockInfo* ptr = (__gm__ BlockInfo*)(dumpWorkspaceStart + DUMP_UINTSIZE * core);
    if (ptr->dumpOffset < (padOffset + sizeof(DumpMessageHead) + ONE_DUMP_BACKUP_SIZE)) {
        KERNEL_LOG(KERNEL_ERROR,
            "Space not enough! need %u Bytes, current remained dump space is %lu Bytes",
            (padOffset + sizeof(DumpMessageHead) + ONE_DUMP_BACKUP_SIZE),
            ptr->dumpOffset);
        *((__gm__ uint32_t*)ptr + BLOCK_INFO_RSV_POS) = DUMP_EXC_FLAG;
        dcci((__gm__ uint64_t*)ptr, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
        return false;
    }

    return true;
}
template <typename T>
__aicore__ inline void DumpBlockInfoImpl(const GlobalTensor<T>& src, uint32_t desc, uint32_t dumpSize, uint32_t padOffset)
{
    uint64_t dumpWorkspaceStart = reinterpret_cast<uint64_t>(g_dumpWorkspaceReserved) - DUMP_WORKSPACE_SIZE;
    uint32_t position =  static_cast<uint32_t>(AscendC::Hardware::GM);
    T data;

    __gm__ BlockInfo* ptr = (__gm__ BlockInfo*)(dumpWorkspaceStart + DUMP_UINTSIZE * GetDumpBlockIdx());
    *((__gm__ uint32_t*)ptr->dumpAddr + DUMP_MESSAGE_HEAD_TYPE_POS) = static_cast<uint32_t>(DumpType::DUMP_TENSOR);
    *((__gm__ uint32_t*)ptr->dumpAddr + DUMP_MESSAGE_HEAD_LEN_POS) = padOffset + DUMP_MSG_HEAD_SIZE;
    *((__gm__ uint32_t*)ptr->dumpAddr + DUMP_MESSAGE_HEAD_ADDR_POS) =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(src.GetPhyAddr()));
    *((__gm__ uint32_t*)ptr->dumpAddr + DUMP_MESSAGE_HEAD_DATA_TYPE_POS) = GetDataType(data);
    *((__gm__ uint32_t*)ptr->dumpAddr + DUMP_MESSAGE_HEAD_DESC_POS) = desc;
    *((__gm__ uint32_t*)ptr->dumpAddr + DUMP_MESSAGE_HEAD_BUFFERID_POS) = 0;
    *((__gm__ uint32_t*)ptr->dumpAddr + DUMP_MESSAGE_HEAD_POSITION_POS) = position;
    *((__gm__ uint32_t*)ptr->dumpAddr + DUMP_MESSAGE_HEAD_DUMP_SIZE_POS) = dumpSize;

    ptr->dumpAddr += sizeof(DumpMessageHead);
    ptr->dumpOffset -= sizeof(DumpMessageHead);
    dcci((__gm__ uint64_t*)ptr, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
}
template <typename T>
__aicore__ inline void DumpGMTailImpl(LocalTensor<T>& tmp, uint32_t alignSize, uint64_t tmpAddr,
                                      uint64_t gmAddr, uint32_t offset)
{
    DataCopyParams tailParams = GetDataCopyParamImpl((alignSize + ONE_BLK_SIZE - 1) & (~(ONE_BLK_SIZE - 1)));
    if (g_coreType == AIV) {
        DataCopyGM2UBImpl((__ubuf__ T*)tmp.GetPhyAddr(),
                          (__gm__ T*)(tmpAddr + offset - alignSize), tailParams);
        PipeBarrier<PIPE_ALL>();
        DataCopyUB2GMImpl((__gm__ T*)gmAddr, (__ubuf__ T*)tmp.GetPhyAddr(), tailParams);
    } else if (g_coreType == AIC) {
        DataCopyGM2L1Impl((__cbuf__ T*)tmp.GetPhyAddr(),
                          (__gm__ T*)(tmpAddr + offset - alignSize), tailParams);
        PipeBarrier<PIPE_ALL>();
        DataCopyL12GMImpl((__gm__ T*)gmAddr, (__cbuf__ T*)tmp.GetPhyAddr(), tailParams);
    }
    PipeBarrier<PIPE_ALL>();
}
template <typename T>
__aicore__ inline void DumpTensorGM2GMEntityImpl(const GlobalTensor<T>& src, uint32_t desc, uint32_t dumpSize)
{
    uint32_t position =  static_cast<uint32_t>(AscendC::Hardware::GM);
    T data;
    uint32_t offset = dumpSize * sizeof(T);
    uint32_t padOffset = AlignUp(offset, ONE_BLK_SIZE);
    if (!CheckDumpValid(padOffset)) {
        return;
    }
    DumpBlockInfoImpl(src, desc, dumpSize, padOffset);
    uint64_t dumpWorkspaceStart = reinterpret_cast<uint64_t>(g_dumpWorkspaceReserved) - DUMP_WORKSPACE_SIZE;
    __gm__ BlockInfo* ptr = (__gm__ BlockInfo*)(dumpWorkspaceStart + DUMP_UINTSIZE * GetDumpBlockIdx());
    DataCopyParams backupParams = GetDataCopyParamImpl(ONE_DUMP_BACKUP_SIZE); // 1K unit
    LocalTensor<T> tmp;
    uint64_t gmBackAddr = dumpWorkspaceStart + DUMP_UINTSIZE * (GetDumpBlockIdx() + 1) - ONE_DUMP_BACKUP_SIZE;

    // 1、alloc 1k UB 2、 backup static GM addr 3、loop copy 4、recover
    PipeBarrier<PIPE_ALL>();
    if (g_coreType == AIV) {  // BACKUP
        InitTmpTensor(tmp, static_cast<uint8_t>(TPosition::VECIN));
        DataCopyUB2GMImpl((__gm__ T*)(gmBackAddr), (__ubuf__ T*)tmp.GetPhyAddr(), backupParams);
    } else if (g_coreType == AIC) {
        InitTmpTensor(tmp, static_cast<uint8_t>(TPosition::A1));
        DataCopyL12GMImpl((__gm__ T*)(gmBackAddr), (__cbuf__ T*)tmp.GetPhyAddr(), backupParams);
    }
    PipeBarrier<PIPE_ALL>();
    dcci((__gm__ uint64_t*)gmBackAddr, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);

    uint32_t alignSize = padOffset % ONE_DUMP_BACKUP_SIZE;
    uint64_t tmpAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src.GetPhyAddr()));
    uint64_t gmAddr = ptr->dumpAddr;
    for (int i = 0; i < padOffset / ONE_DUMP_BACKUP_SIZE; i++) {
        if (g_coreType == AIV) { // LOOP COPY
            DataCopyGM2UBImpl((__ubuf__ T*)tmp.GetPhyAddr(),
                              (__gm__ T*)(tmpAddr + ONE_DUMP_BACKUP_SIZE * i), backupParams);
            PipeBarrier<PIPE_ALL>();
            DataCopyUB2GMImpl((__gm__ T*)gmAddr, (__ubuf__ T*)tmp.GetPhyAddr(), backupParams);
            gmAddr += ONE_DUMP_BACKUP_SIZE;
        } else if (g_coreType == AIC) {
            DataCopyGM2L1Impl((__cbuf__ T*)tmp.GetPhyAddr(),
                              (__gm__ T*)(tmpAddr + ONE_DUMP_BACKUP_SIZE * i), backupParams);
            PipeBarrier<PIPE_ALL>();
            DataCopyL12GMImpl((__gm__ T*)gmAddr, (__cbuf__ T*)tmp.GetPhyAddr(), backupParams);
            gmAddr += ONE_DUMP_BACKUP_SIZE;
        }
        PipeBarrier<PIPE_ALL>();
    }
    if (alignSize != 0) {
        DumpGMTailImpl(tmp, alignSize, tmpAddr, gmAddr, padOffset);
    }
    if (g_coreType == AIV) { // RECOVER
        DataCopyGM2UBImpl((__ubuf__ T*)tmp.GetPhyAddr(), (__gm__ T*)gmBackAddr, backupParams);
    } else if (g_coreType == AIC) {
        DataCopyGM2L1Impl((__cbuf__ T*)tmp.GetPhyAddr(), (__gm__ T*)gmBackAddr, backupParams);
    }
    PipeBarrier<PIPE_ALL>();
    ptr->dumpOffset -= padOffset;
    ptr->dumpAddr += padOffset;
    dcci((__gm__ uint64_t*)ptr, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
}

template <typename T>
__aicore__ inline void DumpTensorGM2GMImpl(const GlobalTensor<T>& src, uint32_t desc, uint32_t dumpSize)
{
    uint64_t ctrlValue = get_ctrl();
    set_atomic_none();
    if (g_sysPrintFifoSpace != nullptr) {
        DumpTensorFifoImpl(src, desc, dumpSize);
    } else {
        DumpTensorGM2GMEntityImpl(src, desc, dumpSize);
    }
    set_ctrl(ctrlValue);
}

__aicore__ inline uint32_t GetArgsNum()
{
    return 0;
}

template <typename T, typename... Args>
__aicore__ inline uint32_t GetArgsNum(T scalar, Args... args)
{
    return 1 + GetArgsNum(args...);
}

__aicore__ inline uint32_t GetStringLength(__gm__ const char* s)
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
__aicore__ inline uint32_t GetArgsSize(Args&&... args);

template <typename... Args>
__aicore__ inline uint32_t GetArgsSizeImpl(__gm__ const char* s, Args&&... args)
{
    uint32_t strLen = GetStringLength(s);
    uint32_t strParamSize = ONE_PARAM_SIZE + strLen;
    return strParamSize + GetArgsSize(args...);
}

template <typename T, typename... Args>
__aicore__ inline uint32_t GetArgsSizeImpl(T scalar, Args&&... args)
{
    return ONE_PARAM_SIZE + GetArgsSize(args...);
}

template <typename... Args>
__aicore__ inline uint32_t GetArgsSize(Args&&... args)
{
    return GetArgsSizeImpl(args...);
}

template <typename... Args>
__aicore__ inline uint32_t GetParamSize(__gm__ const char* fmt, Args&&... args)
{
    uint32_t fmtSize = GetStringLength(fmt);
    uint32_t argsSize = GetArgsSize(args...);
    return fmtSize + argsSize + ONE_PARAM_SIZE;
}

__aicore__ __gm__ inline BlockInfo *GetBlockInfo()
{
    uint8_t core = GetDumpBlockIdx();
    uint64_t dumpWorkspaceStart = reinterpret_cast<uint64_t>(g_dumpWorkspaceReserved) - DUMP_WORKSPACE_SIZE;
    __gm__ BlockInfo *blockInfo = (__gm__ BlockInfo *)(dumpWorkspaceStart +  DUMP_UINTSIZE * core);
    return blockInfo;
}

__aicore__ inline void WriteString(__gm__ uint8_t* paramAddr, uint32_t paramIdx, __gm__ const char* s, uint32_t& offset)
{
    __gm__ uint64_t *stringAddr = reinterpret_cast<__gm__ uint64_t *>(paramAddr) + paramIdx;
    __gm__ uint64_t *dstStrAddr = reinterpret_cast<__gm__ uint64_t *>(paramAddr + offset);

    // write string value offset
    *((__gm__ uint64_t *)stringAddr) = static_cast<uint64_t>(offset - ONE_PARAM_SIZE * paramIdx);
    dcci((__gm__ uint64_t*)stringAddr, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);

    // write string content
    __gm__ char *d = (__gm__ char *)(dstStrAddr);
    uint32_t strLen = GetStringLength(s);

    for (uint32_t i = 0; i < strLen; i++) {
        *(d + i) = *(s + i);
        dcci((__gm__ uint64_t*)d, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
    }
    offset += strLen;
}

template <typename T>
__aicore__ inline void WriteScalar(__gm__ uint8_t* paramAddr, uint32_t paramIdx, T scalar)
{
    __gm__ uint64_t *scalarAddr = (__gm__ uint64_t *)paramAddr + paramIdx;
    *scalarAddr = 0;

    static_assert(!SupportType<T, double>(), "printf unsupport double type");

    if constexpr (SupportType<T, half, float>()) {
        *((__gm__ float *)scalarAddr) = static_cast<float>(scalar);
    } else if constexpr(SupportType<T, bfloat16_t>()) {
        *((__gm__ float *)scalarAddr) = ToFloat(scalar);
    } else if constexpr (std::is_signed<T>::value) {
        *((__gm__ int64_t *)scalarAddr) = static_cast<int64_t>(scalar);
    } else if constexpr(std::is_unsigned<T>::value) {
        *((__gm__ uint64_t *)scalarAddr) = static_cast<uint64_t>(scalar);
    } else if constexpr(std::is_pointer<T>::value) {
        *((__gm__ uint64_t *)scalarAddr) = (uintptr_t)scalar;
    } else if constexpr(std::is_enum<T>::value) {
        *((__gm__ uint64_t *)scalarAddr) = static_cast<uint64_t>(scalar);
    }

    dcci((__gm__ uint64_t*)scalarAddr, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
}

__aicore__ inline void SetParam(__gm__ uint8_t* paramAddr, uint32_t paramIdx, uint32_t& offset)
{
    return;
}

template <typename... Args>
__aicore__ inline void SetParam(__gm__ uint8_t* paramAddr, uint32_t paramIdx, uint32_t& offset, Args&&... args);

template <typename... Args>
__aicore__ inline void SetParamImpl(__gm__ uint8_t *paramAddr, uint32_t paramIdx, uint32_t &offset,
                                    __gm__ const char *s, Args&&... args)
{
    WriteString(paramAddr, paramIdx, s, offset);
    SetParam(paramAddr, paramIdx + 1, offset, args...);
}

template <typename T, typename... Args>
__aicore__ inline void SetParamImpl(__gm__ uint8_t* paramAddr, uint32_t paramIdx, uint32_t& offset, T scalar,
                                    Args&&... args)
{
    WriteScalar(paramAddr, paramIdx, scalar);
    SetParam(paramAddr, paramIdx + 1, offset, args...);
}

template <typename... Args>
__aicore__ inline void SetParam(__gm__ uint8_t* paramAddr, uint32_t paramIdx, uint32_t& offset, Args&&... args)
{
    SetParamImpl(paramAddr, paramIdx, offset, args...);
}

__aicore__ inline void WriteTLHead(DumpType printType, __gm__ uint8_t *tlv, uint32_t valueSize)
{
    *((__gm__ uint32_t *)tlv) = static_cast<uint32_t>(printType);
    *((__gm__ uint32_t *)tlv + 1) = valueSize;
    dcci((__gm__ uint64_t*)tlv, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
}
__aicore__ inline void UpdateBlockInfo(uint32_t tlvSize)
{
    __gm__ BlockInfo *blockInfo = GetBlockInfo();
    uint32_t remainSize = blockInfo->dumpOffset;
    uint64_t lastDumpAddr = blockInfo->dumpAddr;

    __gm__ uint8_t *blockInfoStart = (__gm__ uint8_t *)blockInfo;
    *((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_DUMPOFFSET_POS) = remainSize - tlvSize;
    *((__gm__ uint64_t *)((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_DUMP_ADDR)) = lastDumpAddr + tlvSize;
    dcci((__gm__ uint64_t*)blockInfoStart, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
}

template <class... Args>
__aicore__ inline void PrintfEntityImpl(DumpType printType, __gm__ const char* fmt, Args&&... args)
{
#ifdef ASCENDC_DUMP
    uint8_t blockIdx = GetDumpBlockIdx();
    if (blockIdx >= DUMP_CORE_COUNT) {
        return;
    }
    __gm__ BlockInfo *blockInfo = GetBlockInfo();
    uint32_t remainSize = blockInfo->dumpOffset;
    uint64_t dumpAddr = blockInfo->dumpAddr;

    uint32_t paramSize = GetParamSize(fmt, args...);
    uint32_t paramNum = GetArgsNum(args...) + 1;
    paramSize = (paramSize + ONE_PARAM_SIZE - 1) & (~(ONE_PARAM_SIZE - 1));

    uint32_t tlvSize = paramSize + ONE_PARAM_SIZE;
    if (tlvSize > remainSize) {
        __gm__ uint8_t *blockInfoStart = (__gm__ uint8_t *)blockInfo;
        *((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_RSV_POS) = DUMP_EXC_FLAG;
        dcci((__gm__ uint64_t*)blockInfoStart, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
        return;
    }

    __gm__ uint8_t *tlvAddr = (__gm__ uint8_t *)dumpAddr;
    WriteTLHead(printType, tlvAddr, paramSize);
    __gm__ uint8_t *paramAddr = tlvAddr + ONE_PARAM_SIZE;
    uint32_t offset = paramNum * ONE_PARAM_SIZE;
    WriteString(paramAddr, 0, fmt, offset);
    uint32_t paramIdx = 1;
    SetParam(paramAddr, paramIdx, offset, args...);

    // update next print addr
    UpdateBlockInfo(tlvSize);
#endif
}

__aicore__ inline uint32_t GetArgsFifoLen(uint32_t& argsNum)
{
    return 0;
}

template <typename... Args>
__aicore__ inline uint32_t GetArgsFifoLen(uint32_t& argsNum, Args&&... args);

template <typename... Args>
__aicore__ inline uint32_t GetArgsFifoLenImpl(uint32_t& argsNum, __gm__ const char* s, Args&&... args)
{
    constexpr uint32_t paramSize = sizeof(uint64_t);
    const uint32_t& strLen = GetStringLength(s);
    argsNum += 1;
    return paramSize + strLen + GetArgsFifoLen(argsNum, args...);
}

template <typename T, typename... Args>
__aicore__ inline uint32_t GetArgsFifoLenImpl(uint32_t& argsNum, T scalar, Args&&... args)
{
    constexpr uint32_t paramSize = sizeof(uint64_t);
    argsNum += 1;
    return paramSize + GetArgsFifoLen(argsNum, args...);
}

template <typename... Args>
__aicore__ inline uint32_t GetArgsFifoLen(uint32_t& argsNum, Args&&... args)
{
    return GetArgsFifoLenImpl(argsNum, args...);
}

__aicore__ constexpr uint32_t AlignTlvLen(const uint32_t& dataLen)
{
    constexpr uint32_t num = 7;
    return ((dataLen + num) & ~num) + num + 1;
}

template <typename... Args>
__aicore__ inline uint32_t GetPrintFifoTlvLen(uint32_t& argsNum, __gm__ const char* fmt, Args&&... args)
{
    constexpr uint32_t printInfoLen = sizeof(PrintTlvInfoHead);
    const uint32_t& fmtLen = GetStringLength(fmt);
    const uint32_t& argsLen = GetArgsFifoLen(argsNum, args...);
    return AlignTlvLen(printInfoLen + argsLen + fmtLen); // gm need 8 byte align
}

__aicore__ __gm__ inline BlockPrintFiFoInfo* GetPrintFiFoHead()
{
    uint32_t blockIdx = GetDumpBlockIdx();
    if (blockIdx >= DUMP_CORE_COUNT) {
        return nullptr;
    }
    uint32_t blockLength = reinterpret_cast<__gm__ BlockPrintFiFoInfo*>(g_sysPrintFifoSpace)->length;
    __gm__ BlockPrintFiFoInfo* fifoHead =
        reinterpret_cast<__gm__ BlockPrintFiFoInfo*>(g_sysPrintFifoSpace + blockLength * blockIdx);
    return fifoHead->magic == 0xAE86 ? fifoHead : nullptr;
}

__aicore__ inline void SkipPrintFifoDirectly(__gm__ BlockWriteInfo* writeInfo)
{
    writeInfo->writeIdx = 0;
    dcci(reinterpret_cast<__gm__ uint64_t*>(writeInfo), cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
    return;
}

__aicore__ inline void SkipPrintFifoWithInfo(
    __gm__ BlockWriteInfo* writeInfo, __gm__ uint8_t* fifoBuffHead, const uint32_t& fifoBuffLen)
{
    __gm__ BlockSkipInfo* skipInfo = reinterpret_cast<__gm__ BlockSkipInfo*>(fifoBuffHead + writeInfo->writeIdx);
    skipInfo->blockType = static_cast<uint32_t>(DumpType::DUMP_SKIP);
    skipInfo->length = fifoBuffLen - writeInfo->writeIdx - sizeof(BlockSkipInfo);
    writeInfo->writeIdx = 0;
    writeInfo->packIdx += 1;
    dcci(reinterpret_cast<__gm__ uint64_t*>(skipInfo), cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
    dcci(reinterpret_cast<__gm__ uint64_t*>(writeInfo), cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
    return;
}

__aicore__ inline bool WaitFifoReadIdx(
    __gm__ BlockReadInfo* readInfo, uint64_t writeIdx, const uint32_t& fifoTlvLen)
{
    const uint64_t& firstTimeStamp = static_cast<uint64_t>(GetSystemCycle());
    constexpr uint64_t TIMEOUT_CYCLE = 50 * 1000 * 1000 * 5; // 5s
    while(writeIdx + fifoTlvLen > readInfo->readIdx) {
        uint64_t spendTime = static_cast<uint64_t>(GetSystemCycle()) - firstTimeStamp;
        if (spendTime > TIMEOUT_CYCLE) {
            return false;
        }
        dcci(reinterpret_cast<__gm__ uint64_t*>(readInfo), cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
    }
    return true;
}

__aicore__ inline void WriteFifoTlvHead(
    DumpType printType, __gm__ PrintTlvInfoHead* fifoTlvAddr, const uint32_t& fifoTlvLen, const uint32_t& argsNum)
{
    fifoTlvAddr->printfType = static_cast<uint32_t>(printType);
    fifoTlvAddr->printfLength = fifoTlvLen - sizeof(uint32_t[2]);
    fifoTlvAddr->fmtOffset = (argsNum + 1) * sizeof(uint64_t);
    dcci(reinterpret_cast<__gm__ uint64_t*>(fifoTlvAddr), cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
}

__aicore__ inline void MemCopyGm2Gm(__gm__ uint8_t* dst, __gm__ const uint8_t* src, const uint32_t& len)
{
    if (dst == nullptr || src == nullptr)
    {
        return;
    }
    for (uint32_t i = 0; i < len; i++) {
        *(dst + i) = *(src + i);
    }
    dcci((__gm__ uint64_t*)(dst), cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
}

template <typename... Args>
__aicore__ inline void WriteFifoTlvData(__gm__ PrintTlvInfoHead* fifoTlvAddr, __gm__ const char* fmt, Args&&... args)
{
    const uint32_t& strLen = GetStringLength(fmt);
    __gm__ uint8_t* paramAddr =
        reinterpret_cast<__gm__ uint8_t*>(fifoTlvAddr + 1);
    __gm__ uint8_t* fmtAddr = paramAddr + fifoTlvAddr->fmtOffset - sizeof(uint64_t);
    __gm__ uint8_t* strParamAddr = reinterpret_cast<__gm__ uint8_t*>(fmtAddr) + strLen;
    MemCopyGm2Gm(fmtAddr, reinterpret_cast<__gm__ const uint8_t*>(fmt), strLen);
    uint32_t strParamOffset = fifoTlvAddr->fmtOffset + strLen;
    SetParam(paramAddr, 0, strParamOffset, args...);
}

__aicore__ inline void UpdateWriteInfo(__gm__ BlockWriteInfo* writeInfo, const uint32_t& fifoTlvLen)
{
    writeInfo->writeIdx += fifoTlvLen;
    writeInfo->packIdx += 1;
    dcci(reinterpret_cast<__gm__ uint64_t*>(writeInfo), cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
}

__aicore__ __gm__ inline BlockReadInfo* GetBlockFifoReadInfo(__gm__ BlockPrintFiFoInfo* blockFifoInfo)
{
    __gm__ uint8_t* blockFifoHead = reinterpret_cast<__gm__ uint8_t*>(blockFifoInfo);

    return reinterpret_cast<__gm__ BlockReadInfo*>(blockFifoHead + sizeof(BlockPrintFiFoInfo));
}

__aicore__ __gm__ inline BlockWriteInfo* GetBlockFifoWriteInfo(__gm__ BlockPrintFiFoInfo* blockFifoInfo)
{
    __gm__ uint8_t* fifoBuffHead = reinterpret_cast<__gm__ uint8_t*>(blockFifoInfo->dumpAddr);

    return reinterpret_cast<__gm__ BlockWriteInfo*>(fifoBuffHead + blockFifoInfo->remainLen);
}

__aicore__ inline bool CheckAndWaitPrintFifoSpace(__gm__ BlockPrintFiFoInfo* blockFifoInfo, const uint32_t& fifoTlvLen)
{
    constexpr uint32_t minTlvLen = sizeof(BlockSkipInfo);

    __gm__ uint8_t* fifoBuffHead = reinterpret_cast<__gm__ uint8_t*>(blockFifoInfo->dumpAddr);
    uint32_t fifoBuffLen = blockFifoInfo->remainLen;

    __gm__ BlockReadInfo* readInfo = GetBlockFifoReadInfo(blockFifoInfo);
    __gm__ BlockWriteInfo* writeInfo = GetBlockFifoWriteInfo(blockFifoInfo);

    if (minTlvLen >= fifoBuffLen || fifoTlvLen > fifoBuffLen) {
        return false;
    } else if (writeInfo->writeIdx + minTlvLen >= fifoBuffLen){
        SkipPrintFifoDirectly(writeInfo);
    } else if (writeInfo->writeIdx + fifoTlvLen > fifoBuffLen) {
        SkipPrintFifoWithInfo(writeInfo, fifoBuffHead, fifoBuffLen);
    }
    if (writeInfo->packIdx > 0 &&
        writeInfo->writeIdx <= readInfo->readIdx &&
        writeInfo->writeIdx + fifoTlvLen > readInfo->readIdx) {
        return WaitFifoReadIdx(readInfo, writeInfo->writeIdx, fifoTlvLen);
    }
    return true;
}

__aicore__ __gm__ inline uint8_t* GetFifoTlvAddr(__gm__ BlockPrintFiFoInfo* blockFifoInfo)
{
    __gm__ BlockWriteInfo* writeInfo = GetBlockFifoWriteInfo(blockFifoInfo);
    __gm__ uint8_t* fifoBuffHead = reinterpret_cast<__gm__ uint8_t*>(blockFifoInfo->dumpAddr);
    return fifoBuffHead + writeInfo->writeIdx;
}

template <class... Args>
__aicore__ inline void PrintfFifoImpl(DumpType printType, __gm__ const char* fmt, Args&&... args)
{
#ifdef ASCENDC_DUMP
    __gm__ BlockPrintFiFoInfo* blockFifoInfo = GetPrintFiFoHead();
    if (blockFifoInfo == nullptr) {
        return;
    }
    uint32_t argsNum = 0;
    const uint32_t& fifoTlvLen = GetPrintFifoTlvLen(argsNum, fmt, args...);
    if (!CheckAndWaitPrintFifoSpace(blockFifoInfo, fifoTlvLen)) {
        return;
    }

    __gm__ PrintTlvInfoHead* fifoTlvAddr = reinterpret_cast<__gm__ PrintTlvInfoHead*>(GetFifoTlvAddr(blockFifoInfo));

    WriteFifoTlvHead(printType, fifoTlvAddr, fifoTlvLen, argsNum);
    WriteFifoTlvData(fifoTlvAddr, fmt, args...);

    __gm__ BlockWriteInfo* writeInfo = GetBlockFifoWriteInfo(blockFifoInfo);

    UpdateWriteInfo(writeInfo, fifoTlvLen);
#endif // ASCENDC_DUMP
}

template <typename T>
__aicore__ inline Hardware CheckDumpTensorPosition(const LocalTensor<T>& src)
{
    Hardware position = GetPhyType(static_cast<TPosition>(src.GetPosition()));
    if (position != Hardware::UB && position != Hardware::L1 && position != Hardware::L0C) {
        return Hardware::MAX;
    }
    return position;
}

template <typename T>
__aicore__ constexpr inline Internal::DumpTensorDataType GetTensorDataType()
{
    if constexpr (IsSameType<T, bool>::value) {
        return Internal::DumpTensorDataType::ACL_BOOL;
    } else if (IsSameType<T, uint8_t>::value) {
        return Internal::DumpTensorDataType::ACL_UINT8;
    } else if (IsSameType<T, int8_t>::value) {
        return Internal::DumpTensorDataType::ACL_INT8;
    } else if (IsSameType<T, int16_t>::value) {
        return Internal::DumpTensorDataType::ACL_INT16;
    } else if (IsSameType<T, uint16_t>::value) {
        return Internal::DumpTensorDataType::ACL_UINT16;
    } else if (IsSameType<T, int32_t>::value) {
        return Internal::DumpTensorDataType::ACL_INT32;
    } else if (IsSameType<T, uint32_t>::value) {
        return Internal::DumpTensorDataType::ACL_UINT32;
    } else if (IsSameType<T, uint64_t>::value) {
        return Internal::DumpTensorDataType::ACL_UINT64;
    } else if (IsSameType<T, int64_t>::value) {
        return Internal::DumpTensorDataType::ACL_INT64;
    } else if (IsSameType<T, float>::value) {
        return Internal::DumpTensorDataType::ACL_FLOAT;
    } else if (IsSameType<T, half>::value) {
        return Internal::DumpTensorDataType::ACL_FLOAT16;
    } else if (IsSameType<T, bfloat16_t>::value) {
        return Internal::DumpTensorDataType::ACL_BF16;
    } else {
        return Internal::DumpTensorDataType::ACL_MAX;
    }
}

__aicore__ inline void GetMatCopyParam(
    uint32_t dumpSize, uint16_t& n, uint16_t& m, uint16_t& dstStrideDstD, uint16_t& srcStride)
{
    // L0C to GM
    uint16_t align = (dumpSize % DEFAULT_BLOCK_SIZE == 0) ? 0 : 1;
    uint16_t countBlks = align + dumpSize / DEFAULT_BLOCK_SIZE;

    uint16_t burstLen = static_cast<uint16_t>(SRC_BURST_LEN_SIZE_ELE * SRC_BURST_LEN_SIZE_ELE
        * sizeof(float) / ONE_BLK_SIZE);
    n = countBlks * BLOCK_CUBE;
    m = (burstLen * ONE_BLK_SIZE / B32_BYTE_SIZE) / BLOCK_CUBE;
    uint16_t howo = (burstLen * ONE_BLK_SIZE / sizeof(float)) / BLOCK_CUBE;
    srcStride = DivCeil(howo, BLOCK_CUBE) * BLOCK_CUBE;
    dstStrideDstD = burstLen;
}

template <typename T>
__aicore__ inline void SetDumpDataL0C2GM(__gm__ uint8_t* dst, const LocalTensor<T>& src, uint32_t dumpSize)
{
    if ASCEND_IS_NOT_AIC {
        return;
    }
    uint16_t n, m, dstStrideDstD, srcStride;
    GetMatCopyParam(dumpSize, n, m, dstStrideDstD, srcStride);

    copy_matrix_cc_to_gm(reinterpret_cast<__gm__ float*>(dst), reinterpret_cast<__cc__ float*>(src.GetPhyAddr()),
        0, n, m, dstStrideDstD, srcStride, 0, QuantMode_t::NoQuant,
        static_cast<uint8_t>(false), false, false);
}

template <template<typename> class Tensor, typename T>
__aicore__ inline void WriteFifoTlvHead(const Tensor<T>& src, __gm__ DumpTensorTlvInfoHead* fifoTlvAddr,
    const uint32_t& alignDumpDataLen, const uint32_t& desc, const uint32_t& dumpSize)
{
    Hardware position;
    if constexpr (IsSameType<Tensor<T>, LocalTensor<T>>::value) {
        position = GetPhyType(static_cast<TPosition>(src.GetPosition()));
    } else if (IsSameType<Tensor<T>, GlobalTensor<T>>::value) {
        position = Hardware::GM;
    }
    fifoTlvAddr->dumpType = static_cast<uint32_t>(DumpType::DUMP_TENSOR);
    fifoTlvAddr->dumpLength = sizeof(uint32_t[6]) + alignDumpDataLen;
    fifoTlvAddr->addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(src.GetPhyAddr()));
    fifoTlvAddr->dataType = static_cast<uint32_t>(GetTensorDataType<T>());
    fifoTlvAddr->desc = desc;
    fifoTlvAddr->bufferId = 0;
    fifoTlvAddr->position = static_cast<uint32_t>(position);
    fifoTlvAddr->dumpSize = dumpSize * sizeof(T);
    dcci((__gm__ uint64_t*)(fifoTlvAddr), cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
}

template <typename T>
__aicore__ inline void WriteFifoTlvData(const LocalTensor<T>& src, __gm__ DumpTensorTlvInfoHead* fifoTlvAddr,
    const uint32_t& alignDumpDataLen, const uint32_t& dumpSize)
{
    __gm__ T* dumpDataAddr = reinterpret_cast<__gm__ T*>(fifoTlvAddr + 1);
    DataCopyParams copyParams = {1, static_cast<uint16_t>(alignDumpDataLen / ONE_BLK_SIZE), 0, 0};

    PipeBarrier<PIPE_ALL>();

    if (fifoTlvAddr->position == static_cast<uint32_t>(Hardware::UB)) {
        DataCopyUB2GMImpl(dumpDataAddr, reinterpret_cast<__ubuf__ T*>(src.GetPhyAddr()), copyParams); // UB to GM
    } else if (fifoTlvAddr->position == static_cast<uint32_t>(Hardware::L1)) {
        DataCopyL12GMImpl(dumpDataAddr, reinterpret_cast<__cbuf__ T*>(src.GetPhyAddr()), copyParams); // L1 to GM
    } else if (fifoTlvAddr->position == static_cast<uint32_t>(Hardware::L0C)) {
        SetDumpDataL0C2GM(reinterpret_cast<__gm__ uint8_t*>(dumpDataAddr), src, dumpSize); // L0C to GM
    }

    PipeBarrier<PIPE_ALL>();

    dcci((__gm__ uint64_t*)dumpDataAddr, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
}

template <typename T>
__aicore__ inline void WriteFifoTlvData(
    const GlobalTensor<T>& src, __gm__ DumpTensorTlvInfoHead* fifoTlvAddr, const uint32_t& dumpSize)
{
    __gm__ uint8_t* dst = reinterpret_cast<__gm__ uint8_t*>(fifoTlvAddr + 1);
    MemCopyGm2Gm(dst, reinterpret_cast<__gm__ const uint8_t*>(src.GetPhyAddr()), dumpSize * sizeof(T));
}

template <template<typename> class Tensor, typename T>
__aicore__ inline void DumpTensorFifoImpl(const Tensor<T>& src, uint32_t desc, uint32_t dumpSize)
{
    if constexpr (GetTensorDataType<T>() == Internal::DumpTensorDataType::ACL_MAX) {
        ASCENDC_ASSERT((false),
                   { KERNEL_LOG(KERNEL_ERROR, "dump tensor not support this data type"); });
        return;
    }

    if (dumpSize == 0) {
        return;
    }

    if constexpr (IsSameType<Tensor<T>, LocalTensor<T>>::value) {
        Hardware position = CheckDumpTensorPosition(src);
        // set the head struct value
        if (position == Hardware::MAX) {
            ASCENDC_ASSERT((false),
                    { KERNEL_LOG(KERNEL_ERROR, "dump tensor only support dump tensor from local to gm"); });
            return;
        } else if (position == Hardware::L0C) {
            if ASCEND_IS_NOT_AIC {
                return;
            }
        }
    }

    __gm__ BlockPrintFiFoInfo* blockFifoInfo = GetPrintFiFoHead();
    if (blockFifoInfo == nullptr) {
        return;
    }
    uint32_t alignDumpDataLen = AlignUp(dumpSize * sizeof(T), ONE_BLK_SIZE);
    uint32_t fifoTlvLen = sizeof(DumpTensorTlvInfoHead) + alignDumpDataLen;
    if (!CheckAndWaitPrintFifoSpace(blockFifoInfo, fifoTlvLen)) {
        return;
    }

    __gm__ DumpTensorTlvInfoHead* fifoTlvAddr =
        reinterpret_cast<__gm__ DumpTensorTlvInfoHead*>(GetFifoTlvAddr(blockFifoInfo));

    WriteFifoTlvHead(src, fifoTlvAddr, alignDumpDataLen, desc, dumpSize);
    if constexpr (IsSameType<Tensor<T>, LocalTensor<T>>::value) {
        WriteFifoTlvData(src, fifoTlvAddr, alignDumpDataLen, dumpSize);
    } else if (IsSameType<Tensor<T>, GlobalTensor<T>>::value) {
        WriteFifoTlvData(src, fifoTlvAddr, dumpSize);
    }

    __gm__ BlockWriteInfo* writeInfo = GetBlockFifoWriteInfo(blockFifoInfo);

    UpdateWriteInfo(writeInfo, fifoTlvLen);
}

__aicore__ inline void WriteFifoShapeInfo(const ShapeInfo &shapeInfo)
{
    __gm__ BlockPrintFiFoInfo* blockFifoInfo = GetPrintFiFoHead();
    if (blockFifoInfo == nullptr) {
        return;
    }
    uint32_t fifoTlvLen = sizeof(DumpShapeTlvInfo);
    if (!CheckAndWaitPrintFifoSpace(blockFifoInfo, fifoTlvLen)) {
        return;
    }
    __gm__ DumpShapeTlvInfo* fifoTlvAddr =
        reinterpret_cast<__gm__ DumpShapeTlvInfo*>(GetFifoTlvAddr(blockFifoInfo));
    fifoTlvAddr->dumpType = static_cast<uint32_t>(DumpType::DUMP_SHAPE);
    fifoTlvAddr->dumpLength = sizeof(uint32_t[10]);
    fifoTlvAddr->dim = shapeInfo.shapeDim;
    for (uint32_t i = 0; i < K_MAX_SHAPE_DIM; ++i) {
        fifoTlvAddr->shape[i] = shapeInfo.shape[i];
    }
    dcci((__gm__ uint64_t*)fifoTlvAddr, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);

    __gm__ BlockWriteInfo* writeInfo = GetBlockFifoWriteInfo(blockFifoInfo);

    UpdateWriteInfo(writeInfo, fifoTlvLen);
}

template <class... Args>
__aicore__ inline void PrintfImpl(DumpType printType, __gm__ const char* fmt, Args&&... args)
{
    uint64_t ctrlValue = get_ctrl();
    set_atomic_none();
    if (g_sysPrintFifoSpace != nullptr) {
        PrintfFifoImpl(printType, fmt, args...);
    } else {
        PrintfEntityImpl(printType, fmt, args...);
    }
    set_ctrl(ctrlValue);
}

__aicore__ inline void WriteTimeStampInfo(uint32_t descId)
{
#ifdef ASCENDC_TIME_STAMP_ON
    __gm__ BlockInfo *blockInfo = GetBlockInfo();
    uint64_t dumpAddr = blockInfo->dumpAddr;
    // // WriteTLHead
    *((__gm__ uint32_t *)dumpAddr) = static_cast<uint32_t>(DumpType::DUMP_TIME_STAMP);
    *((__gm__ uint32_t *)dumpAddr + DUMP_TIME_STAMP_LEN_POS) = DUMP_TIME_STAMP_LEN;
    // write value
    *((__gm__ uint32_t *)dumpAddr + DUMP_TIME_STAMP_ID_POS) = descId;
    *((__gm__ uint64_t *)((__gm__ uint32_t *)dumpAddr + DUMP_TIME_STAMP_CYCLE_POS)) = static_cast<uint64_t>(GetSystemCycle());
    *((__gm__ uint64_t *)((__gm__ uint32_t *)dumpAddr + DUMP_TIME_STAMP_PTR_POS)) = static_cast<uint64_t>(get_pc());
    // update block addr
    *((__gm__ uint64_t *)((__gm__ uint32_t *)blockInfo + BLOCK_INFO_DUMP_ADDR)) = dumpAddr + DUMP_TIME_STAMP_TOTAL_LEN;
    *((__gm__ uint32_t*)blockInfo + BLOCK_INFO_DUMPOFFSET_POS) = blockInfo->dumpOffset - DUMP_TIME_STAMP_TOTAL_LEN;
    dcci((__gm__ uint64_t*)blockInfo, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
#endif
}

__aicore__ inline void WriteFifoTimeStampInfo(uint32_t descId)
{
    __gm__ BlockPrintFiFoInfo* blockFifoInfo = GetPrintFiFoHead();
    if (blockFifoInfo == nullptr) {
        return;
    }
    uint32_t fifoTlvLen = sizeof(TimeStampTlvInfo);
    if (!CheckAndWaitPrintFifoSpace(blockFifoInfo, fifoTlvLen)) {
        return;
    }

    __gm__ TimeStampTlvInfo* fifoTlvAddr =
        reinterpret_cast<__gm__ TimeStampTlvInfo*>(GetFifoTlvAddr(blockFifoInfo));
    fifoTlvAddr->dumpType = static_cast<uint32_t>(DumpType::DUMP_TIME_STAMP);
    fifoTlvAddr->dumpLength = fifoTlvLen - sizeof(uint32_t[2]);
    fifoTlvAddr->descId = descId;
    fifoTlvAddr->resv = static_cast<uint32_t>(0U);
    fifoTlvAddr->cycle = static_cast<uint64_t>(GetSystemCycle());
    fifoTlvAddr->pc = static_cast<uint64_t>(get_pc());
    fifoTlvAddr->entry = static_cast<uint64_t>(0);
    dcci((__gm__ uint64_t*)fifoTlvAddr, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);

    __gm__ BlockWriteInfo* writeInfo = GetBlockFifoWriteInfo(blockFifoInfo);

    UpdateWriteInfo(writeInfo, fifoTlvLen);
}

__aicore__ inline void DumpTimeStampImpl(uint32_t descId)
{
    if (g_sysPrintFifoSpace != nullptr) {
        WriteFifoTimeStampInfo(descId);
    } else {
        WriteTimeStampInfo(descId);
    }
}

__aicore__ inline void AscendCTimeStamp(uint32_t descId, uint64_t pcPtr = 0)
{
#ifdef ASCENDC_TIME_STAMP_ON  // 打点开关宏
    DumpTimeStampImpl(descId);
#endif
}

__aicore__ inline void InitDump(bool mixFlag, uint32_t gmLen)
{
#if defined(ASCENDC_DUMP) || defined(ASCENDC_ACC_DUMP) || defined(ASCENDC_TIME_STAMP_ON)
    if (g_sysPrintFifoSpace != nullptr) {
        return;
    }
    g_dumpWorkspaceReserved = GetSysWorkSpacePtr();
    InitDumpImpl(mixFlag, gmLen);
#else
    return;
#endif
}
__aicore__ inline void InitDump(bool mixFlag, GM_ADDR dumpStartAddr, uint32_t gmLen)
{
#if defined(ASCENDC_DUMP) || defined(ASCENDC_ACC_DUMP) || defined(ASCENDC_TIME_STAMP_ON)
    if (g_sysPrintFifoSpace != nullptr) {
        return;
    }
    g_dumpWorkspaceReserved = dumpStartAddr + DUMP_WORKSPACE_SIZE;
    InitDumpImpl(mixFlag, gmLen);
#else
    return;
#endif
}
}  // namespace AscendC
#endif
