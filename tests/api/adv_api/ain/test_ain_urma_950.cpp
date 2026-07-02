/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <gtest/gtest.h>
#include <vector>
#define private public
#include "kernel_operator.h"
#include "ain/ain.h"

using namespace AscendC;

namespace {

constexpr uint32_t URMA_SQ_DEPTH = 10;
constexpr uint32_t URMA_WQE_SIZE = 64;
constexpr uint32_t URMA_CQE_SIZE = 64;
constexpr uint32_t URMA_BUFFER_NUM = 2;
constexpr uint32_t AIN_RANK_SIZE = 2;
constexpr uint32_t AIN_PEER = 1;
constexpr uint64_t REMOTE_BASE = 0x3000;
constexpr uint64_t LOCAL_BASE = 0x5000;
constexpr uint32_t DESCRIPTOR_SIZE = AscendC::HCOMM_URMA_TMP_BUF_SIZE + 32;

class UrmaChannelResource {
public:
    UrmaChannelResource() : sqBuffer_(URMA_SQ_DEPTH * URMA_WQE_SIZE, 0), cqBuffer_(URMA_SQ_DEPTH * URMA_CQE_SIZE, 0)
    {
        channel_.engine = COMM_ENGINE_AIV;
        channel_.protocol = COMM_PROTOCOL_UB_MEM;
        channel_.sqNum = 1;
        channel_.cqNum = 1;
        channel_.remoteBufferNum = URMA_BUFFER_NUM;
        channel_.localBufferNum = URMA_BUFFER_NUM;
        channel_.sqContextAddr = &sqCtx_;
        channel_.cqContextAddr = &cqCtx_;
        channel_.remoteBufferAddr = remoteBuffers_.data();
        channel_.localBufferAddr = localBuffers_.data();

        sqCtx_.type = AscendC::SQ_CONTEXT_TYPE_UB_JFS;
        sqCtx_.contextInfo.ubJfs.sqVa = reinterpret_cast<uint64_t>(sqBuffer_.data());
        sqCtx_.contextInfo.ubJfs.headAddr = reinterpret_cast<uint64_t>(&sqHead_);
        sqCtx_.contextInfo.ubJfs.tailAddr = reinterpret_cast<uint64_t>(&sqTail_);
        sqCtx_.contextInfo.ubJfs.dbVa = reinterpret_cast<uint64_t>(&sqDoorbell_);
        sqCtx_.contextInfo.ubJfs.jfsID = 1;
        sqCtx_.contextInfo.ubJfs.wqeSize = URMA_WQE_SIZE;
        sqCtx_.contextInfo.ubJfs.sqDepth = URMA_SQ_DEPTH;
        sqCtx_.contextInfo.ubJfs.tpID = 1;

        cqCtx_.type = AscendC::CQ_CONTEXT_TYPE_UB_JFC;
        cqCtx_.contextInfo.ubJfc.scqVa = reinterpret_cast<uint64_t>(cqBuffer_.data());
        cqCtx_.contextInfo.ubJfc.headAddr = reinterpret_cast<uint64_t>(&cqHead_);
        cqCtx_.contextInfo.ubJfc.tailAddr = reinterpret_cast<uint64_t>(&cqTail_);
        cqCtx_.contextInfo.ubJfc.dbVa = reinterpret_cast<uint64_t>(&cqDoorbell_);
        cqCtx_.contextInfo.ubJfc.jfcID = 1;
        cqCtx_.contextInfo.ubJfc.cqeSize = URMA_CQE_SIZE;
        cqCtx_.contextInfo.ubJfc.cqDepth = URMA_SQ_DEPTH;

        InitBuffer(remoteBuffers_[0], REMOTE_BASE, 0x1000, 0x123456, 0x654321);
        InitBuffer(remoteBuffers_[1], 0x8000, 0x1000, 0x223456, 0x754321);
        InitBuffer(localBuffers_[0], LOCAL_BASE, 0x1000, 0x111111, 0x222222);
        InitBuffer(localBuffers_[1], 0xA000, 0x1000, 0x333333, 0x444444);
    }

    AscendC::ChannelHandle GetHandle() { return reinterpret_cast<AscendC::ChannelHandle>(&channel_); }

    uint32_t GetSqHead() const { return sqHead_; }

    uint32_t GetSqDoorbell() const { return sqDoorbell_; }

    uint32_t GetSqTail() const { return sqTail_; }

    void CompleteCurrentSq()
    {
        cqTail_ = sqHead_;
        sqTail_ = sqHead_;
    }

private:
    void InitBuffer(
        AscendC::RegedBufferEntity& buffer, uint64_t addr, uint64_t size, uint32_t tokenId, uint32_t tokenValue)
    {
        buffer.type = AscendC::REGED_BUFFER_RMA;
        buffer.bufferInfo.rma.addr = addr;
        buffer.bufferInfo.rma.size = size;
        buffer.bufferInfo.rma.protectionInfo.type = AscendC::PROTECTION_TYPE_UB;
        buffer.bufferInfo.rma.protectionInfo.memInfo.ub.tokenId = tokenId;
        buffer.bufferInfo.rma.protectionInfo.memInfo.ub.tokenValue = tokenValue;
    }

private:
    AscendC::ChannelEntity channel_ = {};
    AscendC::SqContext sqCtx_ = {};
    AscendC::CqContext cqCtx_ = {};
    std::array<AscendC::RegedBufferEntity, URMA_BUFFER_NUM> remoteBuffers_ = {};
    std::array<AscendC::RegedBufferEntity, URMA_BUFFER_NUM> localBuffers_ = {};
    std::vector<uint8_t> sqBuffer_;
    std::vector<uint8_t> cqBuffer_;
    uint32_t sqHead_ = 0;
    uint32_t sqTail_ = 0;
    uint32_t cqHead_ = 0;
    uint32_t cqTail_ = 0;
    uint32_t sqDoorbell_ = 0;
    uint32_t cqDoorbell_ = 0;
};

class AinResource {
public:
    explicit AinResource(AscendC::ChannelHandle channel)
    {
        channelHandles_[AIN_PEER] = channel;
        for (uint32_t i = 0; i < AIN_RANK_SIZE; ++i) {
            channelHandlePtrs_[i] = &channelHandles_[i];
            entityNumPerRank_[i] = 1;
        }
        aivRes_.entity = channelHandlePtrs_.data();
        aivRes_.entityNumPerRank = entityNumPerRank_.data();

        devComm_.rankId = 0;
        devComm_.rankSize = AIN_RANK_SIZE;
        devComm_.AivRes = &aivRes_;

        remoteMems_[0].type = COMM_MEM_TYPE_DEVICE;
        remoteMems_[0].addr = reinterpret_cast<void*>(0x1000);
        remoteMems_[0].size = 0x1000;
        remoteMems_[AIN_PEER].type = COMM_MEM_TYPE_DEVICE;
        remoteMems_[AIN_PEER].addr = reinterpret_cast<void*>(REMOTE_BASE);
        remoteMems_[AIN_PEER].size = 0x1000;

        dstWin_.userVa = reinterpret_cast<void*>(0x7000);
        dstWin_.rankSize = AIN_RANK_SIZE;
        dstWin_.remoteMems = remoteMems_.data();
        dstWin_.remoteMemNum = AIN_RANK_SIZE;

        srcWin_.userVa = reinterpret_cast<void*>(LOCAL_BASE);
        srcWin_.rankSize = AIN_RANK_SIZE;
        srcWin_.remoteMems = remoteMems_.data();
        srcWin_.remoteMemNum = AIN_RANK_SIZE;
    }

    AscendC::AinDevComm GetDevComm() { return reinterpret_cast<AscendC::AinDevComm>(&devComm_); }

    AscendC::AinCommSymWindow GetDstWin() { return reinterpret_cast<AscendC::AinCommSymWindow>(&dstWin_); }

    AscendC::AinCommSymWindow GetSrcWin() { return reinterpret_cast<AscendC::AinCommSymWindow>(&srcWin_); }

private:
    AivRes aivRes_ = {};
    HcclDevComm devComm_ = {};
    std::array<AscendC::ChannelHandle, AIN_RANK_SIZE> channelHandles_ = {};
    std::array<AscendC::ChannelHandle*, AIN_RANK_SIZE> channelHandlePtrs_ = {};
    std::array<uint32_t, AIN_RANK_SIZE> entityNumPerRank_ = {};
    std::array<CommMem, AIN_RANK_SIZE> remoteMems_ = {};
    AscendC::SymmetricWindow dstWin_ = {};
    AscendC::SymmetricWindow srcWin_ = {};
};

} // namespace

class AinUrmaTestSuite : public testing::Test {
protected:
    void SetUp() override
    {
        blockIdxBak_ = block_idx;
        pipe_.InitBuffer(descriptorBuf_, DESCRIPTOR_SIZE);
        descriptor_ = descriptorBuf_.Get<uint8_t>();
        descriptorUbuf_.addr = reinterpret_cast<__ubuf__ uint8_t*>(descriptor_.GetPhyAddr());
        descriptorUbuf_.bytes = DESCRIPTOR_SIZE;
        descriptorUbuf_.eventId = 0;
    }

    void TearDown() override { block_idx = blockIdxBak_; }

    AscendC::AinDescriptorUbuf GetDescriptor() const { return descriptorUbuf_; }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECOUT> descriptorBuf_;
    AscendC::LocalTensor<uint8_t> descriptor_;
    AscendC::AinDescriptorUbuf descriptorUbuf_ = {};
    int64_t blockIdxBak_ = 0;
};

TEST_F(AinUrmaTestSuite, PutImmediateCommitsWrite)
{
    UrmaChannelResource channel;
    AinResource ainResource(channel.GetHandle());
    AscendC::Ain ain(ainResource.GetDevComm(), 0);

    ain.put(
        AscendC::AinTeam{}, AIN_PEER, ainResource.GetDstWin(), 0x20, ainResource.GetSrcWin(), 0x40, 8,
        AscendC::AinRemoteNone{}, AscendC::AinLocalNone{}, GetDescriptor());

    EXPECT_EQ(channel.GetSqHead(), 1U);
    EXPECT_EQ(channel.GetSqDoorbell(), 1U);
}

TEST_F(AinUrmaTestSuite, GetImmediateCommitsRead)
{
    UrmaChannelResource channel;
    AinResource ainResource(channel.GetHandle());
    AscendC::Ain ain(ainResource.GetDevComm(), 0);

    ain.get(
        AscendC::AinTeam{}, AIN_PEER, ainResource.GetDstWin(), 0x40, ainResource.GetSrcWin(), 0x20, 8, GetDescriptor());

    EXPECT_EQ(channel.GetSqHead(), 1U);
    EXPECT_EQ(channel.GetSqDoorbell(), 1U);
}

TEST_F(AinUrmaTestSuite, DelayedPutFlushCommitsAndDrains)
{
    UrmaChannelResource channel;
    AinResource ainResource(channel.GetHandle());
    AscendC::Ain ain(ainResource.GetDevComm(), 0);

    ain.put(
        AscendC::AinTeam{}, AIN_PEER, ainResource.GetDstWin(), 0x20, ainResource.GetSrcWin(), 0x40, 8,
        AscendC::AinRemoteNone{}, AscendC::AinLocalNone{}, GetDescriptor(), AscendC::AIN_COMMIT_DELAYED);

    EXPECT_EQ(channel.GetSqHead(), 1U);
    EXPECT_EQ(channel.GetSqDoorbell(), 0U);

    channel.CompleteCurrentSq();
    ain.flush();

    EXPECT_EQ(channel.GetSqDoorbell(), 0U);
    EXPECT_EQ(channel.GetSqTail(), 1U);
}

TEST_F(AinUrmaTestSuite, DelayedGetFlushCommitsAndDrains)
{
    UrmaChannelResource channel;
    AinResource ainResource(channel.GetHandle());
    AscendC::Ain ain(ainResource.GetDevComm(), 0);

    ain.get(
        AscendC::AinTeam{}, AIN_PEER, ainResource.GetDstWin(), 0x40, ainResource.GetSrcWin(), 0x20, 8, GetDescriptor(),
        AscendC::AIN_COMMIT_DELAYED);

    EXPECT_EQ(channel.GetSqHead(), 1U);
    EXPECT_EQ(channel.GetSqDoorbell(), 0U);

    channel.CompleteCurrentSq();
    ain.flush();

    EXPECT_EQ(channel.GetSqDoorbell(), 0U);
    EXPECT_EQ(channel.GetSqTail(), 1U);
}

TEST_F(AinUrmaTestSuite, DelayedOperationsFlushPeerChannel)
{
    UrmaChannelResource channel;
    AinResource ainResource(channel.GetHandle());
    AscendC::Ain ain(ainResource.GetDevComm(), 0);

    ain.put(
        AscendC::AinTeam{}, AIN_PEER, ainResource.GetDstWin(), 0x20, ainResource.GetSrcWin(), 0x40, 8,
        AscendC::AinRemoteNone{}, AscendC::AinLocalNone{}, GetDescriptor(), AscendC::AIN_COMMIT_DELAYED);
    ain.get(
        AscendC::AinTeam{}, AIN_PEER, ainResource.GetDstWin(), 0x60, ainResource.GetSrcWin(), 0x80, 8, GetDescriptor(),
        AscendC::AIN_COMMIT_DELAYED);

    EXPECT_EQ(channel.GetSqHead(), 2U);
    EXPECT_EQ(channel.GetSqDoorbell(), 0U);

    channel.CompleteCurrentSq();
    ain.flush();

    EXPECT_EQ(channel.GetSqDoorbell(), 0U);
    EXPECT_EQ(channel.GetSqTail(), 2U);
}
