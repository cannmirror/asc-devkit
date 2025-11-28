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
 * \file kernel_operator_scm_data_copy_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_SCM_DATA_COPY_IMPL_H
#define ASCENDC_MODULE_OPERATOR_SCM_DATA_COPY_IMPL_H
#include "kfc/kfc_comm_client.h"
#include "kernel_struct_data_copy.h"

namespace AscendC {
struct Gm2L1Params {
    __cbuf__ void* dst = nullptr;
    __gm__ void* src = nullptr;
    DataCopyParams intri;
};
struct Gm2L1Nd2NzParams {
    __cbuf__ void* dst = nullptr;
    __gm__ void* src = nullptr;
    uint8_t dataTypeLen = 2;
    Nd2NzParams intri;
};

/*
    AIV
 */
__aicore__ inline void ScmDataCopyMsg(__cbuf__ void* dst, __gm__ void* src, const DataCopyParams& intriParams,
    int32_t ubAddr)
{
    ASSERT(g_coreType == AIV);
    ASSERT(GetKfcClient() != nullptr);
    auto msg = GetKfcClient()->AllocMessage();
    ASSERT(sizeof(msg->buffer) >= sizeof(struct Gm2L1Params));

    __ubuf__ struct Gm2L1Params* p = (__ubuf__ struct Gm2L1Params*)&(GetKfcClient()->ubMsg->buffer);
    p->dst = dst;
    p->src = src;
    p->intri.blockCount = intriParams.blockCount;
    p->intri.blockLen = intriParams.blockLen;
    p->intri.srcStride = intriParams.srcStride;
    p->intri.dstStride = intriParams.dstStride;
    GetKfcClient()->ubMsg->ubAddr = ubAddr;
    GetKfcClient()->ubMsg->head = KfcMsgMakeFlag(KFC_Enum::SCMFUN_GM2L1, 0);
    GetKfcClient()->PostMessage<false>(msg);
}

__aicore__ inline void ScmDataCopyND2NZMsg(__cbuf__ void* dst, __gm__ void* src, const uint8_t dataTypeSize,
    const Nd2NzParams& intriParams, int32_t ubAddr)
{
    ASSERT(g_coreType == AIV);
    ASSERT(dst != nullptr);
    ASSERT(src != nullptr);
    ASSERT(GetKfcClient() != nullptr);
    auto msg = GetKfcClient()->AllocMessage();
    ASSERT(sizeof(msg->buffer) >= sizeof(struct Gm2L1Nd2NzParams));

    auto p = (__ubuf__ struct Gm2L1Nd2NzParams*)&(GetKfcClient()->ubMsg->buffer);
    p->dst = dst;
    p->src = src;
    p->dataTypeLen = dataTypeSize;
    p->intri.ndNum = intriParams.ndNum;
    p->intri.nValue = intriParams.nValue;
    p->intri.dValue = intriParams.dValue;
    p->intri.srcNdMatrixStride = intriParams.srcNdMatrixStride;
    p->intri.dstNzC0Stride = intriParams.dstNzC0Stride;
    p->intri.dstNzNStride = intriParams.dstNzNStride;
    p->intri.dstNzMatrixStride = intriParams.dstNzMatrixStride;
    p->intri.srcDValue = intriParams.srcDValue;
    GetKfcClient()->ubMsg->ubAddr = ubAddr;
    GetKfcClient()->ubMsg->head = KfcMsgMakeFlag(KFC_Enum::SCMFUN_GM2L1ND2NZ, 0);
    GetKfcClient()->PostMessage<false>(msg);
}
} // namespace AscendC
#endif