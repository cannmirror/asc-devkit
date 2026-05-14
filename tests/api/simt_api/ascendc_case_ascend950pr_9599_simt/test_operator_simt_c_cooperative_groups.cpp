/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>
#include <type_traits>
#include "simt_compiler_stub.h"
#include "simt_api/cooperative_groups.h"

using namespace std;
using namespace AscendC;
using namespace cooperative_groups;

class CooperativeGroupsTestsuite : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(CooperativeGroupsTestsuite, GroupTypeTest)
{
    EXPECT_EQ(static_cast<unsigned int>(group_type::thread_block_type), 0u);
    EXPECT_EQ(static_cast<unsigned int>(group_type::tiled_group_type), 1u);
    EXPECT_EQ(static_cast<unsigned int>(group_type::coalesced_group_type), 2u);
}

TEST_F(CooperativeGroupsTestsuite, ThreadGroupGetTypeTest)
{
    thread_group tg(group_type::thread_block_type);
    EXPECT_EQ(tg.get_type(), group_type::thread_block_type);

    thread_group tg2(group_type::coalesced_group_type);
    EXPECT_EQ(tg2.get_type(), group_type::coalesced_group_type);

    thread_group tg3(group_type::tiled_group_type);
    EXPECT_EQ(tg3.get_type(), group_type::tiled_group_type);
}

TEST_F(CooperativeGroupsTestsuite, ThreadBlockThisThreadBlockTest)
{
    thread_block tb = this_thread_block();
    EXPECT_EQ(tb.get_type(), group_type::thread_block_type);
}

TEST_F(CooperativeGroupsTestsuite, ThreadBlockSyncTest)
{
    thread_block::sync();
}

TEST_F(CooperativeGroupsTestsuite, ThreadBlockThreadRankTest)
{
    unsigned int rank = thread_block::thread_rank();
    EXPECT_GE(rank, 0u);
}

TEST_F(CooperativeGroupsTestsuite, ThreadBlockGroupIndexTest)
{
    dim3 gi = thread_block::group_index();
    EXPECT_GE(gi.x, 0u);
}

TEST_F(CooperativeGroupsTestsuite, ThreadBlockThreadIndexTest)
{
    dim3 ti = thread_block::thread_index();
    EXPECT_GE(ti.x, 0u);
}

TEST_F(CooperativeGroupsTestsuite, ThreadBlockDimThreadsTest)
{
    dim3 dt = thread_block::dim_threads();
    EXPECT_GT(dt.x, 0u);
}

TEST_F(CooperativeGroupsTestsuite, ThreadBlockNumThreadsTest)
{
    unsigned int nt = thread_block::num_threads();
    EXPECT_GT(nt, 0u);
}

TEST_F(CooperativeGroupsTestsuite, ThreadBlockSizeTest)
{
    unsigned int s = thread_block::size();
    EXPECT_EQ(s, thread_block::num_threads());
}

TEST_F(CooperativeGroupsTestsuite, ThreadBlockGroupDimTest)
{
    dim3 gd = thread_block::group_dim();
    EXPECT_EQ(gd.x, thread_block::dim_threads().x);
}

TEST_F(CooperativeGroupsTestsuite, CoalescedGroupConstructorTest)
{
    coalesced_group cg(0xFFFFFFFF);
    EXPECT_EQ(cg.get_type(), group_type::coalesced_group_type);
    EXPECT_EQ(cg.num_threads(), 32ull);
    EXPECT_EQ(cg.size(), 32ull);
}

TEST_F(CooperativeGroupsTestsuite, CoalescedGroupConstructorPartialMaskTest)
{
    coalesced_group cg(0x0000FFFF);
    EXPECT_EQ(cg.num_threads(), 16ull);
    EXPECT_EQ(cg.size(), 16ull);
    EXPECT_EQ(cg.meta_group_size(), 1ull);
    EXPECT_EQ(cg.meta_group_rank(), 0ull);
}

TEST_F(CooperativeGroupsTestsuite, CoalescedGroupSyncTest)
{
    coalesced_group cg(0xFFFFFFFF);
    cg.sync();
}

TEST_F(CooperativeGroupsTestsuite, CoalescedGroupThreadRankTest)
{
    coalesced_group cg(0xFFFFFFFF);
    unsigned long long rank = cg.thread_rank();
    EXPECT_GE(rank, 0ull);
}

TEST_F(CooperativeGroupsTestsuite, CoalescedGroupMetaGroupSizeRankTest)
{
    coalesced_group cg(0x0000FFFF);
    EXPECT_EQ(cg.meta_group_size(), 1ull);
    EXPECT_EQ(cg.meta_group_rank(), 0ull);
}

TEST_F(CooperativeGroupsTestsuite, CoalescedThreadsTypeTest)
{
    EXPECT_EQ(coalesced_group(0xFFFFFFFF).get_type(), group_type::coalesced_group_type);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_CoalescedGroupShflInt32Test)
{
    coalesced_group cg(0xFFFFFFFF);
    int32_t var = 42;
    int32_t result = cg.shfl(var, 0);
    EXPECT_EQ(result, 42);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_CoalescedGroupShflUint32Test)
{
    coalesced_group cg(0xFFFFFFFF);
    uint32_t var = 100u;
    uint32_t result = cg.shfl(var, 0);
    EXPECT_EQ(result, 100u);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_CoalescedGroupShflInt64Test)
{
    coalesced_group cg(0xFFFFFFFF);
    int64_t var = 12345678901234ll;
    int64_t result = cg.shfl(var, 0);
    EXPECT_EQ(result, 12345678901234ll);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_CoalescedGroupShflUint64Test)
{
    coalesced_group cg(0xFFFFFFFF);
    uint64_t var = 9876543210ull;
    uint64_t result = cg.shfl(var, 0);
    EXPECT_EQ(result, 9876543210ull);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_CoalescedGroupShflFloatTest)
{
    coalesced_group cg(0xFFFFFFFF);
    float var = 3.14f;
    float result = cg.shfl(var, 0);
    EXPECT_FLOAT_EQ(result, 3.14f);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_CoalescedGroupShflHalfTest)
{
    coalesced_group cg(0xFFFFFFFF);
    half var = static_cast<half>(1.5);
    half result = cg.shfl(var, 0);
    EXPECT_EQ(result, static_cast<half>(1.5));
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_CoalescedGroupShflHalf2Test)
{
    coalesced_group cg(0xFFFFFFFF);
    half2 var;
    var.x = static_cast<half>(2.5);
    var.y = static_cast<half>(2.5);
    half2 result = cg.shfl(var, 0);
    half2 expected;
    expected.x = static_cast<half>(2.5);
    expected.y = static_cast<half>(2.5);
    EXPECT_EQ(result.x, expected.x);
    EXPECT_EQ(result.y, expected.y);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_CoalescedGroupShflUpInt32Test)
{
    coalesced_group cg(0xFFFFFFFF);
    int32_t var = 10;
    int32_t result = cg.shfl_up(var, 1);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_CoalescedGroupShflUpFloatTest)
{
    coalesced_group cg(0xFFFFFFFF);
    float var = 1.0f;
    float result = cg.shfl_up(var, 1);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_CoalescedGroupShflDownInt32Test)
{
    coalesced_group cg(0xFFFFFFFF);
    int32_t var = 10;
    int32_t result = cg.shfl_down(var, 1);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_CoalescedGroupShflDownFloatTest)
{
    coalesced_group cg(0xFFFFFFFF);
    float var = 1.0f;
    float result = cg.shfl_down(var, 1);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_CoalescedGroupAnyTest)
{
    coalesced_group cg(0xFFFFFFFF);
    int result = cg.any(1);
    EXPECT_EQ(result, 1);
    result = cg.any(0);
    EXPECT_EQ(result, 0);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_CoalescedGroupAllTest)
{
    coalesced_group cg(0xFFFFFFFF);
    int result = cg.all(1);
    EXPECT_EQ(result, 1);
    result = cg.all(0);
    EXPECT_EQ(result, 0);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_CoalescedGroupBallotTest)
{
    coalesced_group cg(0xFFFFFFFF);
    unsigned int result = cg.ballot(1);
    EXPECT_EQ(result, 0xFFFFFFFFu);
}

TEST_F(CooperativeGroupsTestsuite, TileHelpersTest)
{
    EXPECT_EQ(tile_helpers<32>::tile_count, 1u);
    EXPECT_EQ(tile_helpers<32>::tile_mask, 0xFFFFFFFFu);
    EXPECT_EQ(tile_helpers<32>::lane_mask, 0x1Fu);
    EXPECT_EQ(tile_helpers<32>::shift_count, 5u);

    EXPECT_EQ(tile_helpers<16>::tile_count, 2u);
    EXPECT_EQ(tile_helpers<16>::tile_mask, 0x0000FFFFu);
    EXPECT_EQ(tile_helpers<16>::lane_mask, 0x0Fu);
    EXPECT_EQ(tile_helpers<16>::shift_count, 4u);

    EXPECT_EQ(tile_helpers<8>::tile_count, 4u);
    EXPECT_EQ(tile_helpers<8>::tile_mask, 0x000000FFu);
    EXPECT_EQ(tile_helpers<8>::lane_mask, 0x07u);
    EXPECT_EQ(tile_helpers<8>::shift_count, 3u);

    EXPECT_EQ(tile_helpers<4>::tile_count, 8u);
    EXPECT_EQ(tile_helpers<4>::tile_mask, 0x0000000Fu);
    EXPECT_EQ(tile_helpers<4>::lane_mask, 0x03u);
    EXPECT_EQ(tile_helpers<4>::shift_count, 2u);

    EXPECT_EQ(tile_helpers<2>::tile_count, 16u);
    EXPECT_EQ(tile_helpers<2>::tile_mask, 0x00000003u);
    EXPECT_EQ(tile_helpers<2>::lane_mask, 0x01u);
    EXPECT_EQ(tile_helpers<2>::shift_count, 1u);

    EXPECT_EQ(tile_helpers<1>::tile_count, 32u);
    EXPECT_EQ(tile_helpers<1>::tile_mask, 0x00000001u);
    EXPECT_EQ(tile_helpers<1>::lane_mask, 0x00u);
    EXPECT_EQ(tile_helpers<1>::shift_count, 0u);
}

TEST_F(CooperativeGroupsTestsuite, ThreadBlockTileBaseNumThreadsSizeTest)
{
    EXPECT_EQ(thread_block_tile_base<32>::num_threads(), 32ull);
    EXPECT_EQ(thread_block_tile_base<32>::size(), 32ull);
    EXPECT_EQ(thread_block_tile_base<16>::num_threads(), 16ull);
    EXPECT_EQ(thread_block_tile_base<16>::size(), 16ull);
    EXPECT_EQ(thread_block_tile_base<8>::num_threads(), 8ull);
    EXPECT_EQ(thread_block_tile_base<8>::size(), 8ull);
    EXPECT_EQ(thread_block_tile_base<4>::num_threads(), 4ull);
    EXPECT_EQ(thread_block_tile_base<4>::size(), 4ull);
    EXPECT_EQ(thread_block_tile_base<2>::num_threads(), 2ull);
    EXPECT_EQ(thread_block_tile_base<2>::size(), 2ull);
}

TEST_F(CooperativeGroupsTestsuite, ThreadBlockTileBaseThreadRankTest)
{
    unsigned long long rank = thread_block_tile_base<32>::thread_rank();
    EXPECT_GE(rank, 0ull);
}

TEST_F(CooperativeGroupsTestsuite, ThreadBlockTileBaseSyncTest)
{
    thread_block_tile_base<32> tbtb;
    tbtb.sync();
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_ThreadBlockTileBaseShflInt32Test)
{
    thread_block_tile_base<32> tbtb;
    int32_t var = 42;
    int32_t result = tbtb.shfl(var, 0);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_ThreadBlockTileBaseShflFloatTest)
{
    thread_block_tile_base<32> tbtb;
    float var = 3.14f;
    float result = tbtb.shfl(var, 0);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_ThreadBlockTileBaseShflUpInt32Test)
{
    thread_block_tile_base<32> tbtb;
    int32_t var = 10;
    int32_t result = tbtb.shfl_up(var, 1);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_ThreadBlockTileBaseShflDownInt32Test)
{
    thread_block_tile_base<32> tbtb;
    int32_t var = 10;
    int32_t result = tbtb.shfl_down(var, 1);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_ThreadBlockTileBaseShflXorInt32Test)
{
    thread_block_tile_base<32> tbtb;
    int32_t var = 10;
    int32_t result = tbtb.shfl_xor(var, 1);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_ThreadBlockTileBaseAnyTest)
{
    thread_block_tile_base<32> tbtb;
    int result = tbtb.any(1);
    EXPECT_EQ(result, 1);
    result = tbtb.any(0);
    EXPECT_EQ(result, 0);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_ThreadBlockTileBaseAllTest)
{
    thread_block_tile_base<32> tbtb;
    int result = tbtb.all(1);
    EXPECT_EQ(result, 1);
    result = tbtb.all(0);
    EXPECT_EQ(result, 0);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_ThreadBlockTileBaseBallotTest)
{
    thread_block_tile_base<32> tbtb;
    unsigned int result = tbtb.ballot(1);
    EXPECT_NE(result, 0u);
}

TEST_F(CooperativeGroupsTestsuite, TiledPartitionThreadBlockSize4Test)
{
    thread_block tb = this_thread_block();
    auto tile = tiled_partition<4>(tb);
    EXPECT_EQ(tile.size(), 4ull);
    EXPECT_EQ(tile.num_threads(), 4ull);
}

TEST_F(CooperativeGroupsTestsuite, TiledPartitionThreadBlockSize8Test)
{
    thread_block tb = this_thread_block();
    auto tile = tiled_partition<8>(tb);
    EXPECT_EQ(tile.size(), 8ull);
    EXPECT_EQ(tile.num_threads(), 8ull);
}

TEST_F(CooperativeGroupsTestsuite, TiledPartitionThreadBlockSize16Test)
{
    thread_block tb = this_thread_block();
    auto tile = tiled_partition<16>(tb);
    EXPECT_EQ(tile.size(), 16ull);
    EXPECT_EQ(tile.num_threads(), 16ull);
}

TEST_F(CooperativeGroupsTestsuite, TiledPartitionThreadBlockSize32Test)
{
    thread_block tb = this_thread_block();
    auto tile = tiled_partition<32>(tb);
    EXPECT_EQ(tile.size(), 32ull);
    EXPECT_EQ(tile.num_threads(), 32ull);
}

TEST_F(CooperativeGroupsTestsuite, TiledPartitionMetaGroupInfoTest)
{
    thread_block tb = this_thread_block();
    auto tile = tiled_partition<4>(tb);
    unsigned int meta_rank = tile.meta_group_rank();
    unsigned int meta_size = tile.meta_group_size();
    EXPECT_GE(meta_rank, 0u);
    EXPECT_GT(meta_size, 0u);
}

TEST_F(CooperativeGroupsTestsuite, TiledPartitionThreadGroupTest)
{
    thread_block tb = this_thread_block();
    thread_group tg = tiled_partition(tb, 4);
    EXPECT_EQ(tg.get_type(), group_type::coalesced_group_type);
}

TEST_F(CooperativeGroupsTestsuite, TiledPartitionCoalescedGroupTest)
{
    coalesced_group cg(0x0000FFFF);
    coalesced_group result = tiled_partition(cg, 4);
    EXPECT_EQ(result.get_type(), group_type::coalesced_group_type);
}

TEST_F(CooperativeGroupsTestsuite, SupportTypeSimtInternelTest)
{
    EXPECT_TRUE((SupportTypeSimtInternel<int32_t, int32_t>));
    EXPECT_TRUE((SupportTypeSimtInternel<uint32_t, int32_t, uint32_t>));
    EXPECT_FALSE((SupportTypeSimtInternel<double, int32_t, uint32_t>));
    EXPECT_TRUE((SupportTypeSimtInternel<float, int32_t, float>));
    EXPECT_TRUE((SupportTypeSimtInternel<half, int32_t, half>));
    EXPECT_TRUE((SupportTypeSimtInternel<half2, int32_t, half2>));
    EXPECT_TRUE((SupportTypeSimtInternel<int64_t, int32_t, int64_t>));
    EXPECT_TRUE((SupportTypeSimtInternel<uint64_t, int32_t, uint64_t>));
}

TEST_F(CooperativeGroupsTestsuite, FnsInternalOffsetZeroTest)
{
    unsigned int mask = 0xFFFFFFFF;
    unsigned int result = __fns_internal(mask, 0, 0);
    EXPECT_EQ(result, 0u);
}

TEST_F(CooperativeGroupsTestsuite, FnsInternalOffsetPositiveTest)
{
    unsigned int mask = 0xFFFFFFFF;
    unsigned int result = __fns_internal(mask, 0, 1);
    EXPECT_EQ(result, 0u);
}

TEST_F(CooperativeGroupsTestsuite, FnsInternalOffsetNegativeTest)
{
    unsigned int mask = 0xFFFFFFFF;
    unsigned int result = __fns_internal(mask, 31, -1);
    EXPECT_EQ(result, 31u);
}

TEST_F(CooperativeGroupsTestsuite, FnsInternalPartialMaskTest)
{
    unsigned int mask = 0x0000FFFF;
    unsigned int result = __fns_internal(mask, 0, 1);
    EXPECT_EQ(result, 0u);
}

TEST_F(CooperativeGroupsTestsuite, FnsInternalNotFoundTest)
{
    unsigned int mask = 0x00000001;
    unsigned int result = __fns_internal(mask, 0, 2);
    EXPECT_EQ(result, static_cast<unsigned int>(-1));
}

TEST_F(CooperativeGroupsTestsuite, ThreadBlockTileImplMetaGroupRankSizeTest)
{
    thread_block tb = this_thread_block();
    auto tile = tiled_partition<4>(tb);
    unsigned int meta_rank = tile.meta_group_rank();
    unsigned int meta_size = tile.meta_group_size();
    EXPECT_GE(meta_rank, 0u);
    EXPECT_GT(meta_size, 0u);
    EXPECT_EQ(meta_size, (thread_block::size() + 4 - 1) / 4);
}

TEST_F(CooperativeGroupsTestsuite, ThreadBlockTileConversionTest)
{
    thread_block tb = this_thread_block();
    auto tile4 = tiled_partition<4>(tb);
    thread_block_tile<4, void> tile_void = tile4;
    EXPECT_EQ(tile_void.size(), 4ull);
}

TEST_F(CooperativeGroupsTestsuite, ThreadGroupBaseClassThreadBlockTest)
{
    thread_block tb = this_thread_block();
    thread_group& tg = tb;
    EXPECT_EQ(tg.size(), thread_block::size());
    EXPECT_GT(tg.size(), 0ull);
    EXPECT_EQ(tg.num_threads(), thread_block::num_threads());
    EXPECT_EQ(tg.num_threads(), tg.size());
    EXPECT_EQ(tg.thread_rank(), thread_block::thread_rank());
    EXPECT_GE(tg.thread_rank(), 0ull);
    EXPECT_LT(tg.thread_rank(), tg.size());
    EXPECT_EQ(tg.get_type(), group_type::thread_block_type);
}

TEST_F(CooperativeGroupsTestsuite, ThreadGroupBaseClassSyncFromThreadBlockTest)
{
    thread_block tb = this_thread_block();
    thread_group& tg = tb;
    tg.sync();
}

TEST_F(CooperativeGroupsTestsuite, ThreadGroupBaseClassCoalescedGroupTest)
{
    coalesced_group cg(0xFFFFFFFF);
    thread_group& tg = cg;
    EXPECT_EQ(tg.size(), cg.size());
    EXPECT_EQ(tg.size(), 32ull);
    EXPECT_EQ(tg.num_threads(), cg.num_threads());
    EXPECT_EQ(tg.num_threads(), 32ull);
    EXPECT_EQ(tg.thread_rank(), cg.thread_rank());
    EXPECT_EQ(tg.get_type(), group_type::coalesced_group_type);
}

TEST_F(CooperativeGroupsTestsuite, ThreadGroupBaseClassSyncFromCoalescedGroupTest)
{
    coalesced_group cg(0xFFFFFFFF);
    thread_group& tg = cg;
    tg.sync();
}

TEST_F(CooperativeGroupsTestsuite, ThreadGroupBaseClassPartialCoalescedGroupTest)
{
    coalesced_group cg(0x0000FFFF);
    thread_group& tg = cg;
    EXPECT_EQ(tg.size(), 16ull);
    EXPECT_EQ(tg.num_threads(), 16ull);
}

TEST_F(CooperativeGroupsTestsuite, ThreadGroupFromTiledPartitionSizeTest)
{
    thread_block tb = this_thread_block();
    thread_group tg = tiled_partition(tb, 4);
    EXPECT_GT(tg.size(), 0ull);
    EXPECT_LE(tg.size(), 32ull);
    EXPECT_EQ(tg.num_threads(), tg.size());
}

TEST_F(CooperativeGroupsTestsuite, ThreadGroupFromTiledPartitionThreadRankTest)
{
    thread_block tb = this_thread_block();
    thread_group tg = tiled_partition(tb, 4);
    unsigned long long rank = tg.thread_rank();
    EXPECT_EQ(rank, static_cast<unsigned long long>(__popc(tg._tiled_info.mask & lanemask_lt())));
}

TEST_F(CooperativeGroupsTestsuite, ThreadGroupFromTiledPartitionSyncTest)
{
    thread_block tb = this_thread_block();
    thread_group tg = tiled_partition(tb, 4);
    tg.sync();
}

TEST_F(CooperativeGroupsTestsuite, ThreadGroupFromTiledPartitionGetTypeTest)
{
    thread_block tb = this_thread_block();
    thread_group tg = tiled_partition(tb, 4);
    EXPECT_EQ(tg.get_type(), group_type::coalesced_group_type);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_CoalescedThreadsTest)
{
    coalesced_group cg = coalesced_threads();
    EXPECT_EQ(cg.get_type(), group_type::coalesced_group_type);
    EXPECT_GT(cg.num_threads(), 0ull);
    EXPECT_EQ(cg.num_threads(), cg.size());
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_BinaryPartitionCoalescedGroupAllTrueTest)
{
    coalesced_group cg(0xFFFFFFFF);
    coalesced_group result = binary_partition(cg, true);
    EXPECT_EQ(result.get_type(), group_type::coalesced_group_type);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_BinaryPartitionCoalescedGroupAllFalseTest)
{
    coalesced_group cg(0xFFFFFFFF);
    coalesced_group result = binary_partition(cg, false);
    EXPECT_EQ(result.get_type(), group_type::coalesced_group_type);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_BinaryPartitionCoalescedGroupMixedTest)
{
    coalesced_group cg(0xFFFFFFFF);
    bool pred = (thread_block::thread_rank() % 2 == 0);
    coalesced_group result = binary_partition(cg, pred);
    EXPECT_EQ(result.get_type(), group_type::coalesced_group_type);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_BinaryPartitionThreadBlockTileTest)
{
    thread_block tb = this_thread_block();
    auto tile4 = tiled_partition<4>(tb);
    bool pred = (tile4.thread_rank() % 2 == 0);
    coalesced_group result = binary_partition(tile4, pred);
    EXPECT_EQ(result.get_type(), group_type::coalesced_group_type);
}

TEST_F(CooperativeGroupsTestsuite, ThreadBlockTileConstructorTest)
{
    thread_block tb = this_thread_block();
    thread_block_tile<4, thread_block> tile4(tb);
    EXPECT_EQ(tile4.size(), 4ull);
    EXPECT_EQ(tile4.num_threads(), 4ull);
}

TEST_F(CooperativeGroupsTestsuite, ThreadBlockTileFromThreadBlockTest)
{
    thread_block tb = this_thread_block();
    auto tile8 = tiled_partition<8>(tb);
    EXPECT_EQ(tile8.size(), 8ull);
    EXPECT_EQ(tile8.num_threads(), 8ull);
    EXPECT_GE(tile8.thread_rank(), 0ull);
    EXPECT_LT(tile8.thread_rank(), tile8.size());
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_ThreadBlockTileShflInt32Test)
{
    thread_block tb = this_thread_block();
    auto tile4 = tiled_partition<4>(tb);
    int32_t var = 42;
    int32_t result = tile4.shfl(var, 0);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_ThreadBlockTileShflUpInt32Test)
{
    thread_block tb = this_thread_block();
    auto tile4 = tiled_partition<4>(tb);
    int32_t var = 10;
    int32_t result = tile4.shfl_up(var, 1);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_ThreadBlockTileShflDownInt32Test)
{
    thread_block tb = this_thread_block();
    auto tile4 = tiled_partition<4>(tb);
    int32_t var = 10;
    int32_t result = tile4.shfl_down(var, 1);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_ThreadBlockTileShflXorInt32Test)
{
    thread_block tb = this_thread_block();
    auto tile4 = tiled_partition<4>(tb);
    int32_t var = 10;
    int32_t result = tile4.shfl_xor(var, 1);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_ThreadBlockTileAnyTest)
{
    thread_block tb = this_thread_block();
    auto tile4 = tiled_partition<4>(tb);
    int result = tile4.any(1);
    EXPECT_EQ(result, 1);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_ThreadBlockTileAllTest)
{
    thread_block tb = this_thread_block();
    auto tile4 = tiled_partition<4>(tb);
    int result = tile4.all(1);
    EXPECT_EQ(result, 1);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_ThreadBlockTileBallotTest)
{
    thread_block tb = this_thread_block();
    auto tile4 = tiled_partition<4>(tb);
    unsigned int result = tile4.ballot(1);
    EXPECT_NE(result, 0u);
}

TEST_F(CooperativeGroupsTestsuite, TiledPartitionFromThreadBlockTileTest)
{
    thread_block tb = this_thread_block();
    auto tile8 = tiled_partition<8>(tb);
    auto tile4 = tiled_partition<4>(tile8);
    EXPECT_EQ(tile4.size(), 4ull);
    EXPECT_EQ(tile4.num_threads(), 4ull);
    EXPECT_GE(tile4.thread_rank(), 0ull);
}

TEST_F(CooperativeGroupsTestsuite, MultiLevelTiledPartitionMetaGroupInfoTest)
{
    thread_block tb = this_thread_block();
    auto tile8 = tiled_partition<8>(tb);
    auto tile4 = tiled_partition<4>(tile8);
    unsigned int meta_rank = tile4.meta_group_rank();
    unsigned int meta_size = tile4.meta_group_size();
    EXPECT_GE(meta_rank, 0u);
    EXPECT_GT(meta_size, 0u);
}

TEST_F(CooperativeGroupsTestsuite, ThreadBlockTileSyncTest)
{
    thread_block tb = this_thread_block();
    auto tile4 = tiled_partition<4>(tb);
    tile4.sync();
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_CoalescedGroupTiledPartitionPartialMaskTest)
{
    coalesced_group cg(0x0000FFFF);
    coalesced_group tile = tiled_partition(cg, 4);
    EXPECT_GT(tile.num_threads(), 0ull);
    EXPECT_LE(tile.num_threads(), 16ull);
}

TEST_F(CooperativeGroupsTestsuite, ThreadBlockTileMetaGroupRankSizeFromBlockTest)
{
    thread_block tb = this_thread_block();
    auto tile4 = tiled_partition<4>(tb);
    unsigned int meta_rank = tile4.meta_group_rank();
    unsigned int meta_size = tile4.meta_group_size();
    EXPECT_EQ(meta_rank, thread_block::thread_rank() / 4);
    EXPECT_EQ(meta_size, (thread_block::size() + 3) / 4);
}

TEST_F(CooperativeGroupsTestsuite, DISABLED_ThreadGroupTiledPartitionFromCoalescedGroupTest)
{
    coalesced_group cg(0xFFFFFFFF);
    thread_group tg = tiled_partition(static_cast<const thread_group&>(cg), 4);
    EXPECT_EQ(tg.get_type(), group_type::coalesced_group_type);
    EXPECT_GT(tg.num_threads(), 0ull);
}