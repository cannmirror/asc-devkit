/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
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

#include "kernel_utils.h"
#include "kernel_tensor.h"
#include "kernel_operator_common_impl.h"


namespace AscendC {
/* **************************************************************************************************
 * DumpTensorImpl                                             *
 * ************************************************************************************************* */
__aicore__ inline void InitDumpImpl(bool mixFlag, uint32_t gmLen)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "InitDump");
}

template <typename T>
__aicore__ void DumpTensorLocal2GMImpl(const LocalTensor<T>& src, uint32_t desc, uint32_t size)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Dump tensor");
}

__aicore__ inline void DumpShapeImpl(const ShapeInfo &shapeInfo)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Dump shape");
}

template <typename T>
__aicore__ void DumpTensorGM2GMImpl(const GlobalTensor<T>& src, uint32_t desc, uint32_t size)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Dump tensor");
}

template <class... Args>
__aicore__ inline void PrintfImpl(DumpType printType, __gm__ const char* fmt, Args&&... args)
{
#ifdef ASCENDC_DUMP
    ASCENDC_REPORT_NOT_SUPPORT(false, "Dump scalar");
#endif
}

__aicore__ inline void InitDump(bool mixFlag, uint32_t gmLen)
{
    (void)gmLen;
    (void)mixFlag;
    return;
}
__aicore__ inline void InitDump(bool mixFlag, GM_ADDR dumpStartAddr, uint32_t gmLen)
{
    (void)dumpStartAddr;
    (void)gmLen;
    (void)mixFlag;
    return;
}

__aicore__ inline void DumpTimeStampImpl(uint32_t descId)
{
    return;
}
__aicore__ inline void AscendCTimeStamp(uint32_t descId, uint64_t pcPtr = 0)
{
    return;
}
}
#endif
