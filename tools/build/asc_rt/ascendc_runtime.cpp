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
 * \file ascendc_runtime.cpp
 * \brief
 */
#include "ascendc_runtime.h"

#include <utility>
#include <cstdint>
#include <mutex>
#include <functional>
#include <unordered_set>
#include <iostream>
#include <unordered_map>

#include <string.h>
#include <sys/types.h>
#include <stdlib.h>
#include <stdio.h>

#include "runtime/context.h"
#include "runtime/base.h"
#include "runtime/kernel.h"
#include "runtime/stream.h"
#include "rt_ffts.h"
#include "kernel.h"
#include "toolchain/prof_api.h"
#include "mmpa/mmpa_api.h"
#include "acl/acl_rt.h"
#include "mem.h"
// #include "register/stream_manage_func_registry.h"
#include "ascendc_tool_log.h"
#include "acl_rt.h"
#include "acl/acl_base.h"

#ifdef __cplusplus
extern "C" {
#endif

enum class ElfType { ELF_TYPE_ELF = 0, ELF_TYPE_AIVEC, ELF_TYPE_AICUBE, ELF_TYPE_MAX };

static const std::unordered_set<char> ASCEND_SUPPORT_FORMAT = {'d', 'i', 'f', 'F', 'u', 'p', 'x', 'X', 's'};

static bool g_profEnable;

namespace {
constexpr int16_t SIZE_VAL_TWO = 2;
constexpr size_t ASCENDC_KERNEL_ID = 69;
}

#define PROF_TASK_TIME 0x00000002ULL  // dynamic profiling hwts log

bool AscendCheckSoCVersion(const char *socVersion, char *errMsg)
{
    static const std::unordered_map<std::string, std::string> ascendcSocVersionMap {
        {"ascend910b1", "ascend910b"},
        {"ascend910b2", "ascend910b"},
        {"ascend910b2c", "ascend910b"},
        {"ascend910b3", "ascend910b"},
        {"ascend910b4", "ascend910b"},
        {"ascend910b4-1", "ascend910b"},
        {"ascend910_9391", "ascend910b"},
        {"ascend910_9381", "ascend910b"},
        {"ascend910_9372", "ascend910b"},
        {"ascend910_9392", "ascend910b"},
        {"ascend910_9382", "ascend910b"},
        {"ascend910_9362", "ascend910b"},
        {"ascend910_9599", "ascend910_95"},
        {"ascend910_9589", "ascend910_95"},
        {"ascend910_9579", "ascend910_95"},
        {"ascend910_958b", "ascend910_95"},
        {"ascend910_957b", "ascend910_95"},
        {"ascend910_957d", "ascend910_95"},
        {"ascend910_950z", "ascend910_95"},
        {"ascend910_958a", "ascend910_95"},

        {"ascend910a", "ascend910"},
        {"ascend910proa", "ascend910"},
        {"ascend910b", "ascend910"},
        {"ascend910prob", "ascend910"},
        {"ascend910premiuma", "ascend910"},

        {"ascend310p1", "ascend310p"},
        {"ascend310p3", "ascend310p"},
        {"ascend310p5", "ascend310p"},
        {"ascend310p7", "ascend310p"},
        {"ascend310p3vir01", "ascend310p"},
        {"ascend310p3vir02", "ascend310p"},
        {"ascend310p3vir04", "ascend310p"},
        {"ascend310p3vir08", "ascend310p"},

        {"ascend310b1", "ascend310b"},
        {"ascend310b2", "ascend310b"},
        {"ascend310b3", "ascend310b"},
        {"ascend310b4", "ascend310b"}
    };

    static const std::unordered_map<std::string, std::string> ascendcOriSocVersionMap {
        {"ascend910b1", "Ascend910B1"},
        {"ascend910b2", "Ascend910B2"},
        {"ascend910b2c", "Ascend910B2C"},
        {"ascend910b3", "Ascend910B3"},
        {"ascend910b4", "Ascend910B4"},
        {"ascend910b4-1", "Ascend910B4-1"},
        {"ascend910_9391", "Ascend910_9391"},
        {"ascend910_9381", "Ascend910_9381"},
        {"ascend910_9372", "Ascend910_9372"},
        {"ascend910_9392", "Ascend910_9392"},
        {"ascend910_9382", "Ascend910_9382"},
        {"ascend910_9362", "Ascend910_9362"},
        {"ascend910_9599", "Ascend910_9599"},
        {"ascend910_9589", "Ascend910_9589"},
        {"ascend910_9579", "Ascend910_9579"},
        {"ascend910_958b", "Ascend910_958b"},
        {"ascend910_957b", "Ascend910_957b"},
        {"ascend910_957d", "Ascend910_957d"},
        {"ascend910_950z", "Ascend910_950z"},
        {"ascend910_958a", "Ascend910_958a"},

        {"ascend910a", "Ascend910A"},
        {"ascend910proa", "Ascend910ProA"},
        {"ascend910b", "Ascend910B"},
        {"ascend910prob", "Ascend910ProB"},
        {"ascend910premiuma", "Ascend910PremiumA"},

        {"ascend310p1", "Ascend310P1"},
        {"ascend310p3", "Ascend310P3"},
        {"ascend310p5", "Ascend310P5"},
        {"ascend310p7", "Ascend310P7"},
        {"ascend310p3vir01", "Ascend310P3Vir01"},
        {"ascend310p3vir02", "Ascend310P3Vir02"},
        {"ascend310p3vir04", "Ascend310P3Vir04"},
        {"ascend310p3vir08", "Ascend310P3Vir04"},

        {"ascend310b1", "Ascend310B1"},
        {"ascend310b2", "Ascend310B2"},
        {"ascend310b3", "Ascend310B3"},
        {"ascend310b4", "Ascend310B4"}
    };

    std::string compileSocVersion = std::string(socVersion);

    std::string curSocVersion = std::string(aclrtGetSocName());
    std::string lowerCurSocVersion;
    for (char c : curSocVersion) {
        lowerCurSocVersion += std::tolower(c);
    }

    const auto &it = ascendcSocVersionMap.find(compileSocVersion);
    if (it == ascendcSocVersionMap.end()) {
        ASCENDLOGE("bin soc version %s is incorrected.", compileSocVersion.c_str());
        return false;
    }
    const auto &it1 = ascendcSocVersionMap.find(lowerCurSocVersion);
    if (it1 == ascendcSocVersionMap.end()) {
        ASCENDLOGE("cur soc version %s not found.", lowerCurSocVersion.c_str());
        return false;
    }

    if (it->second != it1->second) {
        std::string tmpMsg = "the socversion " + ascendcOriSocVersionMap.at(compileSocVersion) +
            " of bin package does not match the current device socverison " +
            ascendcOriSocVersionMap.at(lowerCurSocVersion) +
            ". Please modify default socversion in run.sh or execute run.sh with socversion parameter.";
        errno_t err = strcpy_s(errMsg, 1024, tmpMsg.c_str()); // 1024 is errMsg length
        if (err != EOK) {
            ASCENDLOGE("strcpy_s failed in AscendCheckSoCVersion!");
        }
        return false;
    }
    return true;
}
uint32_t RegisterAscendBinary(const char *fileBuf, size_t fileSize, uint32_t type, void **handle)
{
    rtDevBinary_t binary;

    switch (static_cast<ElfType>(type)) {
        case ElfType::ELF_TYPE_AIVEC:
            binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
            break;
        case ElfType::ELF_TYPE_AICUBE:
            binary.magic = RT_DEV_BINARY_MAGIC_ELF_AICUBE;
            break;
        case ElfType::ELF_TYPE_ELF:
        default:
            binary.magic = RT_DEV_BINARY_MAGIC_ELF;
    }
    binary.version = 0;
    binary.data = fileBuf;
    binary.length = fileSize;
    return rtRegisterAllKernel(&binary, handle);
}

int32_t AscendDevBinaryRegister(const void *fileBuf, size_t fileSize, void **handle)
{
    rtDevBinary_t binary;
    binary.magic = RT_DEV_BINARY_MAGIC_ELF;
    binary.version = 0;
    binary.data = fileBuf;
    binary.length = fileSize;
    return rtDevBinaryRegister(&binary, handle);
}

int32_t AscendFunctionRegister(void *handle, const char *stubFunc)
{
    return rtFunctionRegister(handle, stubFunc, stubFunc, stubFunc, 0);
}

uint32_t LaunchAscendKernel(void *handle, const uint64_t key, const uint32_t blockDim, void **args, uint32_t size,
    const rtStream_t stream)
{
    rtArgsEx_t argsInfo = {
        .args = nullptr,
        .hostInputInfoPtr = nullptr,
        .argsSize = 0,
        .tilingAddrOffset = 0,
        .tilingDataOffset = 0,
        .hostInputInfoNum = 0,
        .hasTiling = 0,
        .isNoNeedH2DCopy = 0,
        .reserved = {0, 0, 0, 0}};
    argsInfo.args = (void *)args;
    argsInfo.argsSize = size;
    return rtKernelLaunchWithHandle(handle, key, blockDim, &argsInfo, NULL, stream, NULL);
}

int32_t AscendKernelLaunchWithFlagV2(const char *stubFunc, const uint32_t blockDim, void **args, uint32_t size,
    const rtStream_t stream)
{
    rtArgsEx_t argsInfo = {
        .args = nullptr,
        .hostInputInfoPtr = nullptr,
        .argsSize = 0,
        .tilingAddrOffset = 0,
        .tilingDataOffset = 0,
        .hostInputInfoNum = 0,
        .hasTiling = 0,
        .isNoNeedH2DCopy = 0,
        .reserved = {0, 0, 0, 0}};
    argsInfo.args = static_cast<void*>(args);
    argsInfo.argsSize = size;
    return rtKernelLaunchWithFlagV2(stubFunc, blockDim, &argsInfo, nullptr, stream, 0, nullptr);
}

uint32_t GetAscendCoreSyncAddr(void **addr)
{
    uint32_t len;
    return rtGetC2cCtrlAddr((uint64_t *)addr, &len);
}

int UnregisterAscendBinary(void *hdl)
{
    return rtDevBinaryUnRegister(hdl);
}

static void MsprofRc(const char *name, const uint64_t timeStamp)
{
    MsprofAdditionalInfo info{};

    info.type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
    info.level = MSPROF_REPORT_NODE_LEVEL;
    info.timeStamp = timeStamp;
    info.threadId = static_cast<uint32_t>(mmGetTid());
    info.dataLen = static_cast<uint32_t>(sizeof(uint32_t));
    auto contextIdInfo = reinterpret_cast<MsprofContextIdInfo *>(info.data);
    contextIdInfo->ctxIdNum = 1U;
    contextIdInfo->ctxIds[0] = 0U;

    const size_t typeLen = strlen(name);
    const uint64_t typeHash = MsprofGetHashId(name, typeLen);
    contextIdInfo->opName = typeHash;
    MsprofReportAdditionalInfo(static_cast<uint32_t>(true), &info, static_cast<uint32_t>(sizeof(MsprofAdditionalInfo)));
}

static void AscendBuildNodeBasicInfo(uint32_t blockDim, const std::pair<uint64_t, uint64_t> &opNameAndTypeHash,
    uint32_t taskType, uint64_t timeStamp, MsprofCompactInfo &nodeBasicInfo)
{
    auto &profNodeBasicInfo = nodeBasicInfo.data.nodeBasicInfo;
    profNodeBasicInfo.opName = opNameAndTypeHash.first;
    profNodeBasicInfo.opType = opNameAndTypeHash.second;
    profNodeBasicInfo.taskType = taskType;
    profNodeBasicInfo.blockDim = blockDim;
    nodeBasicInfo.level = static_cast<uint16_t>(MSPROF_REPORT_NODE_LEVEL);
    nodeBasicInfo.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
    nodeBasicInfo.timeStamp = timeStamp;
    nodeBasicInfo.threadId = static_cast<uint32_t>(mmGetTid());
}

static void MsprofRn(const char *name, uint32_t blockDim, const uint64_t time, uint32_t taskType)
{
    const uint64_t typeHash = MsprofGetHashId(name, strlen(name));
    MsprofCompactInfo nodeBasicInfo{};
    AscendBuildNodeBasicInfo(blockDim, { typeHash, typeHash }, static_cast<uint32_t>(taskType), time, nodeBasicInfo);
    MsprofReportCompactInfo(static_cast<uint32_t>(true), &nodeBasicInfo,
        static_cast<uint32_t>(sizeof(MsprofCompactInfo)));
}

inline void AscendMsprofReportApi(const uint64_t beginTime, MsprofApi &info)
{
    const uint64_t endTime = MsprofSysCycleTime();
    info.threadId = static_cast<uint32_t>(mmGetTid());
    info.beginTime = beginTime;
    info.endTime = endTime;
    info.magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    info.reserve = 0U;
    const int32_t res = MsprofReportApi(true, &info);
    if (res != 0) {
        ASCENDLOGE("Call MsprofReportApi res = %d\n", res);
    }
}

static void AscendReportLaunchInfo(const uint64_t beginTime, const char *const opType)
{
    MsprofApi info{};
    info.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
    const size_t typeLen = strlen(opType);
    info.itemId = MsprofGetHashId(opType, typeLen);
    info.level = MSPROF_REPORT_NODE_LEVEL;
    AscendMsprofReportApi(beginTime, info);
}

static int32_t AscendProfilingCallBack(uint32_t type, void *data, uint32_t len)
{
    if (data == nullptr) {
        ASCENDLOGE("data is nullptr\n");
        return -1;
    }
    if (len != sizeof(MsprofCommandHandle)) {
        ASCENDLOGE("len(%u) != sizeof MsprofCommandHandle(%zu)\n", len, sizeof(MsprofCommandHandle));
        return -1;
    }

    if (type != 1) {
        ASCENDLOGE("ProfilingCallBack, type = %u, discard this type\n", type);
        return 0;
    }
    MsprofCommandHandle *handle = (MsprofCommandHandle *)data;
    (handle->profSwitch & PROF_TASK_TIME) != 0 ? g_profEnable = true : g_profEnable = false;
    return 0;
}

bool GetAscendProfStatus()
{
    return g_profEnable;
}

void AscendProfRegister()
{
    MsprofRegisterCallback(ASCENDC_KERNEL_ID, AscendProfilingCallBack);
}

void StartAscendProf(const char *name, uint64_t *startTime)
{
    (void)name;
    *startTime = MsprofSysCycleTime();
}

void ReportAscendProf(const char *name, uint32_t blockDim, uint32_t taskType, const uint64_t startTime)
{
    static const std::vector<uint32_t> taskTypeMap = {
        static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_MIX_AIC), // MIX
        static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_AIV),     // AIV
        static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_AI_CORE), // AIC
        static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_AI_CORE), // NORMAL
        static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_AI_CORE), // MIX_VECTOR_CORE
        static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_AIV),     // KERNEL_TYPE_AIV_ONLY
        static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_AI_CORE), // KERNEL_TYPE_AIC_ONLY
        static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_MIX_AIV), // KERNEL_TYPE_MIX_AIV_1_0
        static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_MIX_AIC), // KERNEL_TYPE_MIX_AIC_1_0
        static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_MIX_AIC), // KERNEL_TYPE_MIX_AIC_1_1
        static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_MIX_AIC), // KERNEL_TYPE_MIX_AIC_1_2
        static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_AI_CPU)   // AICPU
    };
    static const std::vector<uint32_t> taskRationMap = {
        SIZE_VAL_TWO, 0, 0, 0, 0, 0, 0, 0, 0, 1, SIZE_VAL_TWO, 0
    };
    uint32_t taskRation = taskRationMap.at(taskType);
    taskType = taskTypeMap.at(taskType);
    AscendReportLaunchInfo(startTime, name);
    if (taskType == MSPROF_GE_TASK_TYPE_MIX_AIC || taskType == MSPROF_GE_TASK_TYPE_MIX_AIV) {
        blockDim = ((blockDim & 0xFFFFU) | (taskRation << 16U));
        MsprofRc(name, startTime + 1);
    }
    MsprofRn(name, blockDim, startTime + 1, taskType);
}

uint32_t AllocAscendMemDevice(void **devMem, uint64_t size)
{
    const rtError_t rtErr = rtMalloc(devMem, size, RT_MEMORYINFO_HBM_HUGE, 0);
    if (rtErr != 0) {
        ASCENDLOGE(" alloc device memory failed, runtime result = %d\n", rtErr);
        return rtErr;
    }
    return 0;
}

uint32_t FreeAscendMemDevice(void *devMem)
{
    const rtError_t rtErr = aclrtFree(devMem);
    if (rtErr != 0) {
        ASCENDLOGE(" free device memory failed, runtime result = %d\n", rtErr);
        return rtErr;
    }
    return 0;
}

typedef struct {
    rtStream_t stream;
    rtEvent_t eventA;
    rtEvent_t eventB;
} AscendCStreamForVectorCore;

static bool g_ascendCRegistedCallBack = false;
static std::mutex g_ascStreamMtx;
std::unordered_map<const void *, AscendCStreamForVectorCore> g_ascStreamMap;
std::unordered_map<void *, std::shared_ptr<std::mutex>> g_ascStreamMtxMap;

static uint32_t AscendCReportAdditionInfo(const char *const opType, uint32_t blockDim,
    uint32_t taskType, const uint64_t timeStamp, const uint64_t itemId)
{
    ASCENDLOGI("[Cann Profiling] node type is %s, taskType is %u\n", opType, taskType);
    const uint64_t typeHash = itemId;
    MsprofCompactInfo nodeBasicInfo{};
    AscendBuildNodeBasicInfo(blockDim, {typeHash, typeHash}, taskType, timeStamp, nodeBasicInfo);
    ASCENDC_ASSERT_RTOK_RETVAL(MsprofReportCompactInfo(
        static_cast<uint32_t>(true), &nodeBasicInfo, static_cast<uint32_t>(sizeof(MsprofCompactInfo))));
    return ASCENDC_SUCCESS;
}

static void AscendCInnerReportLaunchInfo(const uint64_t beginTime, const uint64_t itemId)
{
    ASCENDLOGI("Report LaunchInfo, itemId is %lu\n", itemId);
    MsprofApi info{};
    info.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
    info.itemId = itemId;
    info.level = MSPROF_REPORT_NODE_LEVEL;
    AscendMsprofReportApi(beginTime, info);
}

static inline uint32_t AscendCExecutorPreportProfiling(
    const char *const opType, uint32_t blockDim, const uint32_t taskType, const uint64_t launchBeginTime)
{
    const size_t typeLen = strlen(opType);
    const uint64_t itemId = MsprofGetHashId(opType, typeLen);
    AscendCInnerReportLaunchInfo(launchBeginTime, itemId);
    ASCENDC_ASSERT_RTOK_RETVAL(AscendCReportAdditionInfo(opType, blockDim, taskType,
        launchBeginTime + 1U, itemId));
    return ASCENDC_SUCCESS;
}

#define OP_CHECK_NO_RETURN(cond, log_func)    \
    do {                                      \
        if (!(cond)) {                        \
            log_func;                         \
        }                                     \
    } while (false)

void AscendCDestroyStreamCallBack(rtStream_t stream, const bool isCreate)
{
    if (isCreate) {
        return;
    }
    if (g_ascStreamMap.find(stream) != g_ascStreamMap.end()) {
        ASCENDLOGI("start callback main stream is %p, subStream %p, eventA %p, eventB %p",
            stream,
            g_ascStreamMap[stream].stream,
            g_ascStreamMap[stream].eventA,
            g_ascStreamMap[stream].eventB);
        OP_CHECK_NO_RETURN(aclrtDestroyStream(g_ascStreamMap[stream].stream) == RT_ERROR_NONE,
            ASCENDLOGE("Destroy stream %p failed.", g_ascStreamMap[stream].stream));
        OP_CHECK_NO_RETURN(aclrtDestroyEvent(g_ascStreamMap[stream].eventA) == RT_ERROR_NONE,
            ASCENDLOGE("Destroy event %p failed.", g_ascStreamMap[stream].eventA));
        OP_CHECK_NO_RETURN(aclrtDestroyEvent(g_ascStreamMap[stream].eventB) == RT_ERROR_NONE,
            ASCENDLOGE("Destroy event %p failed.", g_ascStreamMap[stream].eventB));
        g_ascStreamMap.erase(stream);
        ASCENDLOGI("after g_ascStreamMap.size() is %zu.", g_ascStreamMap.size());
    }
    return;
}


static rtArgsEx_t InitializeArgsInfo(void **args, uint32_t size)
{
    rtArgsEx_t argsInfo = {.args = nullptr,
        .hostInputInfoPtr = nullptr,
        .argsSize = 0,
        .tilingAddrOffset = 0,
        .tilingDataOffset = 0,
        .hostInputInfoNum = 0,
        .hasTiling = 0,
        .isNoNeedH2DCopy = 0,
        .reserved = {0, 0, 0, 0}};
    argsInfo.args = reinterpret_cast<void *>(args);
    argsInfo.argsSize = size;
    return argsInfo;
}

static uint32_t AscendCExecutorLaunchKernel(void* binHandle, const uint64_t tilingKey, const uint32_t blockDim,
    void** args, uint32_t size, const rtStream_t stream)
{
    // scheMode new is not need;
    const uint8_t scheMode = 0;
    const rtTaskCfgInfo_t cfgInfo = { 0U, 0U, scheMode, false, 0U, 0U, 0U, {0U, 0U}, 0U };
    ASCENDLOGI("tilingKey is %lu, scheMode is %hhu, blockDim is %u, stream is %p\n", tilingKey, scheMode, blockDim,
        stream);
    rtArgsEx_t argsInfo = InitializeArgsInfo(args, size);
    ASCENDC_ASSERT_RTOK_RETVAL(
        rtKernelLaunchWithHandleV2(binHandle, tilingKey, blockDim, &argsInfo, nullptr, stream, &cfgInfo));
    return ASCENDC_SUCCESS;
}

static uint32_t AscendCExecutorVectorCoreLaunchKernel(void* binHandle, const uint64_t tilingKey,
    const uint32_t blockDim, void** args, uint32_t size, const rtStream_t stream, uint32_t aivBlockDimOffset)
{
    ASCENDLOGI("tilingKey is %lu, aiv blockDim1 is %u\n", tilingKey, blockDim);
    const uint8_t scheMode = 0;
    const rtTaskCfgInfo_t cfgInfo = { 0U, 0U, scheMode, false, aivBlockDimOffset, 0U, 0U, {0U, 0U}, 0U };
    rtArgsEx_t argsInfo = InitializeArgsInfo(args, size);
    ASCENDC_ASSERT_RTOK_RETVAL(
        rtVectorCoreKernelLaunchWithHandle(binHandle, tilingKey, blockDim, &argsInfo, nullptr, stream, &cfgInfo));
    return ASCENDC_SUCCESS;
}

static uint32_t AscendCExecutorGetStreamAndEvent(
    const rtStream_t stream, rtStream_t *subStream, rtEvent_t *evtA, rtEvent_t *evtB,
    std::shared_ptr<std::mutex> &streamLckPtr)
{
    const std::lock_guard<std::mutex> lock(g_ascStreamMtx);
    rtStream_t mainStream = stream;
    if (stream == nullptr) {
        ASCENDLOGI("main stream is nullptr.");
        ASCENDC_ASSERT_RTOK_RETVAL(aclrtCtxGetCurrentDefaultStream(&mainStream));
    }
    if (g_ascStreamMtxMap.find(mainStream) == g_ascStreamMtxMap.cend()) {
        g_ascStreamMtxMap[mainStream] = std::make_shared<std::mutex>();
        ASCENDC_ASSERT_NOTNULL_RETVAL(g_ascStreamMtxMap[mainStream]);
    }
    streamLckPtr = g_ascStreamMtxMap[mainStream];
    if (g_ascStreamMap.find(mainStream) != g_ascStreamMap.end()) {
        *subStream = g_ascStreamMap[mainStream].stream;
        *evtA = g_ascStreamMap[mainStream].eventA;
        *evtB = g_ascStreamMap[mainStream].eventB;
        ASCENDLOGI("find main stream is %p, subStream %p, eventA %p, eventB %p", mainStream, *subStream, *evtA, *evtB);
    } else {
        CHECK_COND(aclrtCreateStreamWithConfig(subStream, RT_STREAM_PRIORITY_DEFAULT,
                                           RT_STREAM_FAST_LAUNCH | RT_STREAM_FAST_SYNC) == RT_ERROR_NONE,
                   ASCENDC_ERR_RUNTIME_ERROR, "create stream %p failed.", subStream);
        CHECK_COND(aclrtCreateEventExWithFlag(evtA, RT_EVENT_WITH_FLAG) == RT_ERROR_NONE,
                   ASCENDC_ERR_RUNTIME_ERROR, "create event %p failed.", evtA);
        CHECK_COND(aclrtCreateEventExWithFlag(evtB, RT_EVENT_WITH_FLAG) == RT_ERROR_NONE,
                   ASCENDC_ERR_RUNTIME_ERROR, "create event %p failed.", evtB);
        g_ascStreamMap[mainStream] = {*subStream, *evtA, *evtB};
    }
    ASCENDLOGI("main stream is %p, subStream %p, eventA %p, eventB %p.", mainStream, *subStream, *evtA, *evtB);

    if (g_ascendCRegistedCallBack) {
        return ASCENDC_SUCCESS;
    }
    ASCENDC_ASSERT_RTOK_RETVAL(rtRegStreamStateCallback("AscendCDestroySteam", AscendCDestroyStreamCallBack));

    g_ascendCRegistedCallBack = true;
    return ASCENDC_SUCCESS;
}

uint32_t LaunchAscendKernelForVectorCore(const char* opType, void* handle, const uint64_t key, void** args,
    uint32_t size, const rtStream_t stream, bool enbaleProf, uint32_t aicBlockDim, uint32_t aivBlockDim,
    uint32_t aivBlockDimOffset)
{
    ASCENDLOGI("aicBlockDim is %u, aivBlockDim is %u, aivBlockDimOffset is %u.\n", aicBlockDim, aivBlockDim,
        aivBlockDimOffset);
    AscendCStreamForVectorCore ascBaseStream = {};
    std::shared_ptr<std::mutex> streamLckPtr;
    ASCENDC_ASSERT_RTOK_RETVAL(
        AscendCExecutorGetStreamAndEvent(
            stream, &ascBaseStream.stream, &ascBaseStream.eventA, &ascBaseStream.eventB, streamLckPtr));
    ASCENDC_ASSERT_NOTNULL_RETVAL(streamLckPtr);
    std::lock_guard<std::mutex> lock(*streamLckPtr);
    ASCENDC_ASSERT_RTOK_RETVAL(aclrtRecordEvent(ascBaseStream.eventA, stream));
    ASCENDC_ASSERT_RTOK_RETVAL(aclrtStreamWaitEvent(ascBaseStream.stream, ascBaseStream.eventA));

    uint64_t launchMainBeginTime = 0;
    uint64_t launchSubBeginTime = 0;
    if (enbaleProf) {
        launchMainBeginTime = MsprofSysCycleTime();
    }

    // aicore kernel launch
    ASCENDC_ASSERT_RTOK_RETVAL(AscendCExecutorLaunchKernel(handle, key, aicBlockDim, args, size, stream));

    if (enbaleProf) {
        ASCENDC_ASSERT_RTOK_RETVAL(AscendCExecutorPreportProfiling(
            opType, aicBlockDim, MSPROF_GE_TASK_TYPE_AI_CORE, launchMainBeginTime));
    }
    ASCENDLOGI("Main stream launch sucess.\n");

    if (enbaleProf) {
        launchSubBeginTime = MsprofSysCycleTime();
    }
    // vector core kernel launch
    ASCENDC_ASSERT_RTOK_RETVAL(AscendCExecutorVectorCoreLaunchKernel(handle, key, aivBlockDim,
        args, size, ascBaseStream.stream, aivBlockDimOffset));
    if (enbaleProf) {
        ASCENDC_ASSERT_RTOK_RETVAL(AscendCExecutorPreportProfiling(
            opType, aivBlockDim, MSPROF_GE_TASK_TYPE_AIV, launchSubBeginTime));
    }
    ASCENDC_ASSERT_RTOK_RETVAL(aclrtRecordEvent(ascBaseStream.eventB, ascBaseStream.stream));
    ASCENDC_ASSERT_RTOK_RETVAL(aclrtStreamWaitEvent(stream, ascBaseStream.eventB));
    ASCENDLOGI("Sub stream launch sucess.\n");

    return ASCENDC_SUCCESS;
}

uint32_t GetCoreNumForMixVectorCore(uint32_t *aiCoreNum, uint32_t *vectorCoreNum)
{
    int32_t deviceId = 0;
    int64_t aicoreNum64 = 0;
    int64_t vectorCoreNum64 = 0;
    ASCENDC_ASSERT_RTOK_RETVAL(aclrtGetDevice(&deviceId));
    ASCENDC_ASSERT_RTOK_RETVAL(
        aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_AICORE_CORE_NUM, &aicoreNum64)
    );
     ASCENDC_ASSERT_RTOK_RETVAL(
        aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &vectorCoreNum64)
    );
    *aiCoreNum = static_cast<uint32_t>(aicoreNum64);
    *vectorCoreNum = static_cast<uint32_t>(vectorCoreNum64);
    ASCENDLOGI("aicore num: %u, vector core num %u\n", *aiCoreNum, *vectorCoreNum);
    return 0;
}

#ifdef __cplusplus
}
#endif