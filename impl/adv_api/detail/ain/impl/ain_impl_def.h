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
 * \file ain_impl_def.h
 * \brief Ain implementation definition
 */

#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#pragma message(                                                                                             \
    "impl/adv_api/detail/ain/impl/ain_impl_def.h is an internal header file and must not be used directly. " \
    "Please use public interface headers.")
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_AIN_IMPL_DEF_H__
#endif

#ifndef IMPL_ADV_API_DETAIL_AIN_IMPL_AIN_IMPL_DEF_H
#define IMPL_ADV_API_DETAIL_AIN_IMPL_AIN_IMPL_DEF_H

#include <cstddef>
#include <cstdint>
#include "../../../../basic_api/kernel_utils.h"

namespace AscendC {

typedef struct {
    uint32_t signal0;
    uint32_t reserved;
} HcclAinBarrierHandle;

typedef struct {
    CommAbiHeader abiHeader;
    ChannelHandle** entity;
    uint32_t* entityNumPerRank;
    uint32_t ainSignalCount;
    uint32_t ainCounterCount;
    uint64_t* ainSignalShadows;
    HcclAinBarrierHandle ainWorldBarrier;
    HcclAinBarrierHandle ainRailBarrier;
    HcclAinBarrierHandle ainCustomBarrier;
    uint64_t* symMemPool;
    uint8_t reserved[256];
} AivRes;

typedef struct {
    CommAbiHeader abiHeader;
    uint32_t rankSize;
    uint32_t rankId;
    void* AicpuRes;
    void* AivRes;
    void* HostDpuRes;
    void* CcuRes;
    uint8_t reserved[256];
} HcclDevComm;

enum class SymmetricMemoryMode { HCCS = 0, URMA = 1 };

typedef enum {
    COMM_MEM_TYPE_INVALID = -1,
    COMM_MEM_TYPE_DEVICE = 0,
    COMM_MEM_TYPE_HOST = 1,
} CommMemType;

typedef struct {
    CommMemType type;
    void *addr;
    uint64_t size;
} CommMem;

struct SymmetricWindow {
    void* userVa;
    size_t userSize;

    void* baseVa;
    size_t alignedHeapOffset;
    size_t alignedSize;
    uint32_t localRank;
    uint32_t rankSize;
    size_t stride;
    void* paHandle;
    SymmetricMemoryMode mode;
    CommMem* remoteMems;
    uint32_t remoteMemNum;
    void* devWin;
};

} // namespace AscendC

#endif // IMPL_ADV_API_DETAIL_AIN_IMPL_AIN_IMPL_DEF_H

#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_AIN_IMPL_DEF_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_AIN_IMPL_DEF_H__
#endif
