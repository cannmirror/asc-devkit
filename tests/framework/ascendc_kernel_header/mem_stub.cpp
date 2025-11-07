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
#include "mem.h"
#include "runtime/context.h"
#include "runtime/base.h"
#include "runtime/kernel.h"
#include "register/stream_manage_func_registry.h"
#include "acl/acl_base.h"
#include "acl/acl_rt.h"

rtError_t rtMalloc(void **devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId)
{
    if (devPtr == nullptr) {
        return 1;
    }
    return RT_ERROR_NONE;
}

aclError aclrtMallocWithCfg(void **devPtr, size_t size, aclrtMemMallocPolicy policy,
    aclrtMallocConfig *cfg)
{
    if (devPtr == nullptr) {
        return 1;
    }
    return ACL_ERROR_NONE;
}

rtError_t rtFree(void *devPtr)
{
    if (devPtr == nullptr) {
        return 1;
    }
    return RT_ERROR_NONE;
}

aclError aclrtFree(void *devPtr)
{
    if (devPtr == nullptr) {
        return 1;
    }
    return ACL_ERROR_NONE;
}

rtError_t rtStreamSynchronizeWithTimeout(rtStream_t stm, int32_t timeout)
{
    return RT_ERROR_NONE;
}

rtError_t rtStreamCreateWithFlags(rtStream_t *stream, int32_t priority, uint32_t flags)
{
    return RT_ERROR_NONE;
}

aclError aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag)
{
    return ACL_ERROR_NONE;
}

aclError aclrtCreateEventExWithFlag(aclrtEvent *event, uint32_t flag)
{
    return ACL_ERROR_NONE;
}

rtError_t rtEventCreateWithFlag(rtEvent_t *event_, uint32_t flag)
{
    return RT_ERROR_NONE;
}

aclError aclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag)
{
    return ACL_ERROR_NONE;
}

rtError_t rtStreamDestroy(rtStream_t stream)
{
    return RT_ERROR_NONE;
}

aclError aclrtDestroyStream(aclrtStream stream)
{
    return ACL_ERROR_NONE;
}

rtError_t rtKernelLaunchWithHandleV2(void *hdl, const uint64_t tilingKey, uint32_t blockDim, rtArgsEx_t *argsInfo,
    rtSmDesc_t *smDesc, rtStream_t stm, const rtTaskCfgInfo_t *cfgInfo)
{
    return RT_ERROR_NONE;
}

rtError_t rtVectorCoreKernelLaunchWithHandle(void *hdl, const uint64_t tilingKey, uint32_t blockDim,
    rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stm, const rtTaskCfgInfo_t *cfgInfo)
{
    return RT_ERROR_NONE;
}

rtError_t rtCtxGetCurrent(rtContext_t *ctx)
{
    return RT_ERROR_NONE;
}

rtError_t rtGetDevice(int32_t *device)
{
    return RT_ERROR_NONE;
}

aclError aclrtGetDevice(int32_t *deviceId)
{
    return ACL_ERROR_NONE;
}

rtError_t rtEventRecord(rtEvent_t event, rtStream_t stream)
{
    return RT_ERROR_NONE;
}

aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream)
{
    return ACL_ERROR_NONE;
}

rtError_t rtStreamWaitEvent(rtStream_t stream, rtEvent_t event)
{
    return RT_ERROR_NONE;
}

aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event)
{
    return ACL_ERROR_NONE;
}

rtError_t rtEventReset(rtEvent_t event, rtStream_t stream)
{
    return RT_ERROR_NONE;
}

aclError aclrtResetEvent(aclrtEvent event, aclrtStream stream)
{
    return ACL_ERROR_NONE;
}

rtError_t rtGetDeviceInfo(uint32_t deviceId, int32_t moduleType, int32_t infoType, int64_t *val)
{
    return RT_ERROR_NONE;
}

aclError aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t *value)
{
    return ACL_ERROR_NONE;
}

rtError_t rtEventDestroy(rtEvent_t event)
{
    return RT_ERROR_NONE;
}

aclError aclrtDestroyEvent(aclrtEvent event)
{
    return ACL_ERROR_NONE;
}

rtError_t rtGetSocVersion(char *version, const uint32_t maxLen)
{
    return 0;
}

const char *aclrtGetSocName()
{
    return " ";
}

namespace ge {
StreamMngFuncRegister::StreamMngFuncRegister(const StreamMngFuncType func_type, StreamMngFunc const manage_func) {}
}