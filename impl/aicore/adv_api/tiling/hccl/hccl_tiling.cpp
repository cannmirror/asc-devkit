/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hccl_tiling.cpp
 * \brief
 */
#include "hccl/hccl_tiling.h"
#include "securec.h"
#include "detail/host_log.h"
#include "detail/hccl/hccl_tiling_msg.h"
#include "hccl/hccl_common.h"

using namespace std;

namespace AscendC {

void PrintMc2InitTiling(const Mc2InitTilingInner& tiling)
{
    TILING_LOG_DEBUG("Mc2InitTiling msg begin.");
    TILING_LOG_DEBUG("Mc2InitTiling msg version:%u", tiling.version);
    TILING_LOG_DEBUG("Mc2InitTiling msg mc2HcommCnt:%u", tiling.mc2HcommCnt);
    for (uint32_t i = 0; i < tiling.mc2HcommCnt; i++) {
        TILING_LOG_DEBUG("Mc2InitTiling msg offset%u:%u", i, tiling.offset[i]);
    }
    TILING_LOG_DEBUG("Mc2InitTiling msg debugMode:%u", tiling.debugMode);
    TILING_LOG_DEBUG("Mc2InitTiling msg preparePosition:%u", tiling.preparePosition);
    TILING_LOG_DEBUG("Mc2InitTiling msg queueNum:%u", tiling.queueNum);
    TILING_LOG_DEBUG("Mc2InitTiling msg commBlockNum:%u", tiling.commBlockNum);
    TILING_LOG_DEBUG("Mc2InitTiling msg end.");
}

void PrintMc2CcTiling(const Mc2CcTilingInner& tiling)
{
    TILING_LOG_DEBUG("Mc2CcTiling msg begin.");
    TILING_LOG_DEBUG("Mc2CcTiling msg skipLocalRankCopy:%u", tiling.skipLocalRankCopy);
    TILING_LOG_DEBUG("Mc2CcTiling msg skipBufferWindowCopy:%u", tiling.skipBufferWindowCopy);
    TILING_LOG_DEBUG("Mc2CcTiling msg stepSize:%u", tiling.stepSize);
    TILING_LOG_DEBUG("Mc2CcTiling msg version:%u", tiling.version);
    TILING_LOG_DEBUG("Mc2CcTiling msg groupName:%s", tiling.groupName);
    TILING_LOG_DEBUG("Mc2CcTiling msg algConfig:%s", tiling.algConfig);
    TILING_LOG_DEBUG("Mc2CcTiling msg opType:%u", tiling.opType);
    TILING_LOG_DEBUG("Mc2CcTiling msg reduceType:%u", tiling.reduceType);
    TILING_LOG_DEBUG("Mc2CcTiling msg end.");
}

uint32_t UpdateMc2InitTiling(uint64_t initTilingAddr, uint64_t ccTilingAddr)
{
    Mc2InitTilingInner* tilingInner = reinterpret_cast<Mc2InitTilingInner*>(static_cast<uintptr_t>(initTilingAddr));
    tilingInner->offset[tilingInner->mc2HcommCnt] = ccTilingAddr - initTilingAddr;
    tilingInner->mc2HcommCnt += 1;
    ASCENDC_HOST_ASSERT(tilingInner->mc2HcommCnt <= INIT_TILING_OFFSET, return TILING_FAILED,
        "mc2HcommCnt(%u) must be less than or equal %u.", tilingInner->mc2HcommCnt, INIT_TILING_OFFSET);
    TILING_LOG_INFO("Update Mc2InitTiling, mc2HcommCnt:%u, offset:%u, initTilingAddr:%#lx, ccTilingAddr:%#lx.",
        tilingInner->mc2HcommCnt, tilingInner->offset[tilingInner->mc2HcommCnt], initTilingAddr, ccTilingAddr);
    PrintMc2InitTiling(*tilingInner);
    return TILING_SUCCESS;
}

Mc2CcTilingConfig::Mc2CcTilingConfig(
    const std::string& groupName, uint32_t opType, const std::string& algConfig, uint32_t reduceType)
{
    impl_.groupName_ = groupName;
    impl_.opType_ = opType;
    impl_.algConfig_ = algConfig;
    impl_.reduceType_ = reduceType;
    TILING_LOG_INFO("Init groupName_:%s, opType_:%u, algConfig_:%s, reduceType_:%u.", impl_.groupName_.c_str(),
        impl_.opType_, impl_.algConfig_.c_str(), impl_.reduceType_);
}

Mc2CcTilingConfig::~Mc2CcTilingConfig() = default;

uint32_t Mc2CcTilingConfig::GetTiling(::Mc2InitTiling& tiling)
{
    // It is not a reduce type or a reduce type, and the reduce type is valid
    const bool reduceFlag = (impl_.opType_ == static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLREDUCE)
                             || impl_.opType_ == static_cast<uint8_t>(HcclCMDType::HCCL_CMD_REDUCE)
                             || impl_.opType_ == static_cast<uint8_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER));
    ASCENDC_HOST_ASSERT(!reduceFlag || impl_.reduceType_ < HCCL_REDUCE_RESERVED, return TILING_FAILED,
        "when opType(%u) is reduce, reduceType must be less than %u.", impl_.opType_,
        static_cast<uint8_t>(HCCL_REDUCE_RESERVED));

    impl_.initTilingAddr_ = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&tiling));
    Mc2InitTilingInner* tilingInner = reinterpret_cast<Mc2InitTilingInner*>(&tiling);
    tilingInner->version = INIT_TILING_VERSION + MC2_INIT_TILING_VERSION;
    tilingInner->mc2HcommCnt = 0;
    tilingInner->debugMode = impl_.debugMode_;
    tilingInner->preparePosition = 0;
    tilingInner->queueNum = impl_.queueNum_;
    tilingInner->commBlockNum = impl_.commBlockNum_;
    for (uint32_t i = 0; i < INIT_TILING_OFFSET; i++) { tilingInner->offset[i] = 0; }
    TILING_LOG_DEBUG("Mc2InitTiling addr %#lx", impl_.initTilingAddr_);
    PrintMc2InitTiling(*tilingInner);
    return TILING_SUCCESS;
}

uint32_t Mc2CcTilingConfig::GetTiling(::Mc2CcTiling& tiling)
{
    ASCENDC_HOST_ASSERT(impl_.initTilingAddr_ != 0, return TILING_FAILED, "must be set Mc2InitTiling first.");
    ASCENDC_HOST_ASSERT(impl_.groupName_.length() <= GROUP_NAME_SIZE, return TILING_FAILED,
        "groupName(%s) must be less than or equal %u.", impl_.groupName_.c_str(), GROUP_NAME_SIZE);
    ASCENDC_HOST_ASSERT(impl_.algConfig_.length() <= ALG_CONFIG_SIZE, return TILING_FAILED,
        "algConfig(%s) must be less than or equal %u.", impl_.algConfig_.c_str(), ALG_CONFIG_SIZE);
    bool isValid = (0 < impl_.opType_) && (impl_.opType_ < static_cast<uint8_t>(HcclCMDType::HCCL_CMD_MAX));
    ASCENDC_HOST_ASSERT(isValid, return TILING_FAILED, "opType(%u) must be less than %u, and not 0.", impl_.opType_,
        static_cast<uint8_t>(HcclCMDType::HCCL_CMD_MAX));
    ASCENDC_HOST_ASSERT(impl_.reduceType_ < HCCL_REDUCE_RESERVED, return TILING_FAILED,
        "reduceType(%u) must be less than %u.", impl_.reduceType_, static_cast<uint8_t>(HCCL_REDUCE_RESERVED));

    Mc2CcTilingInner* tilingInner = reinterpret_cast<Mc2CcTilingInner*>(&tiling);
    tilingInner->skipLocalRankCopy = impl_.skipLocalRankCopy_;
    tilingInner->skipBufferWindowCopy = impl_.skipBufferWindowCopy_;
    tilingInner->stepSize = impl_.stepSize_;
    tilingInner->version = MC2_CC_TILING_VERSION;
    (void)memset_s(tilingInner->reserved, sizeof(tilingInner->reserved), 0, sizeof(tilingInner->reserved));
    auto ret = strcpy_s(tilingInner->groupName, sizeof(tilingInner->groupName), impl_.groupName_.c_str());
    ASCENDC_HOST_ASSERT(ret == EOK, return TILING_FAILED, "groupName(%s) copy failed.", impl_.groupName_.c_str());
    ret = strcpy_s(tilingInner->algConfig, sizeof(tilingInner->algConfig), impl_.algConfig_.c_str());
    ASCENDC_HOST_ASSERT(ret == EOK, return TILING_FAILED, "algConfig(%s) copy failed.", impl_.algConfig_.c_str());
    tilingInner->opType = impl_.opType_;
    tilingInner->reduceType = impl_.reduceType_;
    PrintMc2CcTiling(*tilingInner);

    uint64_t ccTilingAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&tiling));
    return UpdateMc2InitTiling(impl_.initTilingAddr_, ccTilingAddr);
}

uint32_t Mc2CcTilingConfig::SetOpType(uint32_t opType)
{
    bool isValid = (0 < opType) && (opType < static_cast<uint8_t>(HcclCMDType::HCCL_CMD_MAX));
    ASCENDC_HOST_ASSERT(isValid, return TILING_FAILED, "opType(%u) must be less than %u, and not 0.", opType,
        static_cast<uint8_t>(HcclCMDType::HCCL_CMD_MAX));
    impl_.opType_ = opType;
    return TILING_SUCCESS;
}
uint32_t Mc2CcTilingConfig::SetGroupName(const std::string& groupName)
{
    ASCENDC_HOST_ASSERT(groupName.length() <= GROUP_NAME_SIZE, return TILING_FAILED,
        "groupName(%s) must be less than or equal %u.", groupName.c_str(), GROUP_NAME_SIZE);
    impl_.groupName_ = groupName;
    return TILING_SUCCESS;
}
uint32_t Mc2CcTilingConfig::SetAlgConfig(const std::string& algConfig)
{
    ASCENDC_HOST_ASSERT(algConfig.length() <= ALG_CONFIG_SIZE, return TILING_FAILED,
        "algConfig(%s) must be less than or equal %u.", algConfig.c_str(), ALG_CONFIG_SIZE);
    impl_.algConfig_ = algConfig;
    return TILING_SUCCESS;
}
uint32_t Mc2CcTilingConfig::SetReduceType(uint32_t reduceType)
{
    ASCENDC_HOST_ASSERT(reduceType < HCCL_REDUCE_RESERVED, return TILING_FAILED, "reduceType(%u) must be less than %u.",
        reduceType, static_cast<uint8_t>(HCCL_REDUCE_RESERVED));
    impl_.reduceType_ = reduceType;
    return TILING_SUCCESS;
}
uint32_t Mc2CcTilingConfig::SetStepSize(uint8_t stepSize)
{
    impl_.stepSize_ = stepSize;
    return TILING_SUCCESS;
}
uint32_t Mc2CcTilingConfig::SetSkipLocalRankCopy(uint8_t skipLocalRankCopy)
{
    ASCENDC_HOST_ASSERT(skipLocalRankCopy <= 1, return TILING_FAILED,
        "skipLocalRankCopy(%u) must be less than or equal 1.", skipLocalRankCopy);
    impl_.skipLocalRankCopy_ = skipLocalRankCopy;
    return TILING_SUCCESS;
}
uint32_t Mc2CcTilingConfig::SetSkipBufferWindowCopy(uint8_t skipBufferWindowCopy)
{
    ASCENDC_HOST_ASSERT(skipBufferWindowCopy <= 2, return TILING_FAILED,
        "skipBufferWindowCopy(%u) must be less than or equal 2.", skipBufferWindowCopy);
    impl_.skipBufferWindowCopy_ = skipBufferWindowCopy;
    return TILING_SUCCESS;
}
uint32_t Mc2CcTilingConfig::SetDebugMode(uint8_t debugMode)
{
    bool ret = (1 <= debugMode && debugMode <= 4) || (debugMode >= 250);
    ASCENDC_HOST_ASSERT(ret, return TILING_FAILED, "debugMode(%u) only support [1,4] or [250,255].", debugMode);
    impl_.debugMode_ = debugMode;
    return TILING_SUCCESS;
}

uint32_t Mc2CcTilingConfig::SetQueueNum(uint16_t num)
{
    impl_.queueNum_ = num;
    return TILING_SUCCESS;
}

uint32_t Mc2CcTilingConfig::SetCommBlockNum(uint16_t num)
{
    impl_.commBlockNum_ = num;
    return TILING_SUCCESS;
}
} // namespace AscendC
