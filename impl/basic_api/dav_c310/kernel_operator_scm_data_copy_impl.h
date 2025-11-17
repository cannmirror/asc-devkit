/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ASCENDC_MODULE_OPERATOR_SCM_DATA_COPY_IMPL_H
#define ASCENDC_MODULE_OPERATOR_SCM_DATA_COPY_IMPL_H
#if KFC_C310_SSBUF == 1
#include "dav_c310/kfc/kfc_comm_client.h"
#else
#include "dav_c310/kfc/kfc_comm_client_gm.h"
#endif

namespace AscendC {
struct Gm2L1Params {
    __cbuf__ void* dst = nullptr;
    __gm__ void* src = nullptr;
    uint32_t subBlockID = 0;
    uint32_t blockCount = 0;
    uint32_t blockLen = 0;
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;
};
struct Gm2L1Nd2NzParams {
    __cbuf__ void* dst = nullptr;
    __gm__ void* src = nullptr;
    uint32_t dataTypeLen = 2;
    uint32_t subBlockID = 0;
    uint32_t ndNum = 0;
    uint32_t nValue = 0;
    uint32_t dValue = 0;
    uint32_t srcNdMatrixStride = 0;
    uint32_t srcDValue = 0;
    uint32_t dstNzC0Stride = 0;
    uint32_t dstNzNStride = 0;
    uint32_t dstNzMatrixStride = 0;
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

    MSG_POS struct Gm2L1Params* p = (MSG_POS struct Gm2L1Params*)&(msg->buffer);
    p->dst = dst;
    p->src = src;
    p->subBlockID = GetSubBlockIdxImpl();
    p->blockCount = intriParams.blockCount;
    p->blockLen = intriParams.blockLen;
    p->srcStride = intriParams.srcStride;
    p->dstStride = intriParams.dstStride;
    msg->head = KfcMsgMakeFlag(KFC_Enum::SCMFUN_GM2L1, 0);
#if KFC_C310_SSBUF == 1
    set_intra_block(PIPE_MTE3, static_cast<uint8_t>(CUBE_WAIT_INTRA_Enum::GM_L1_UB_GM));
#endif
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

    auto p = (MSG_POS struct Gm2L1Nd2NzParams*)&(msg->buffer);
    p->dst = dst;
    p->src = src;
    p->subBlockID = GetSubBlockIdxImpl();
    p->dataTypeLen = dataTypeSize;
    p->ndNum = intriParams.ndNum;
    p->nValue = intriParams.nValue;
    p->dValue = intriParams.dValue;
    p->srcNdMatrixStride = intriParams.srcNdMatrixStride;
    p->dstNzC0Stride = intriParams.dstNzC0Stride;
    p->dstNzNStride = intriParams.dstNzNStride;
    p->dstNzMatrixStride = intriParams.dstNzMatrixStride;
    p->srcDValue = intriParams.srcDValue;
    msg->head = KfcMsgMakeFlag(KFC_Enum::SCMFUN_GM2L1ND2NZ, 0);
#if KFC_C310_SSBUF == 1
    set_intra_block(PIPE_MTE3, static_cast<uint8_t>(CUBE_WAIT_INTRA_Enum::GM_L1_UB_GM));
#endif
    GetKfcClient()->PostMessage<false>(msg);
}
} // namespace AscendC
#endif
