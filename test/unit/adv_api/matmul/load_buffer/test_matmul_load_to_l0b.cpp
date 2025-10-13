/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * @brief load data instruction ut for ascend910B1
 *
 */
#include <gtest/gtest.h>
#include "kernel_operator.h"
#include "include/adv_api/matmul/tiling.h"
#include "impl/adv_api/detail/matmul/utils/matmul_param.h"
#include "impl/adv_api/detail/matmul/policy/matmul_policy.h"
#define private public
#define protected public
#include "impl/adv_api/detail/matmul/policy/matmul_private_modules.h"
#include "impl/adv_api/detail/matmul/stage/split/load_to_l0b/load_to_l0b.h"

using namespace std;
using namespace AscendC;


namespace {

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using LoadToB2 = Impl::Detail::LoadToL0B<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG, class MM_CB,
MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE(LoadToB2) {
    MATMUL_ALLOW_USING(LoadToB2);

public:
    MatmulImpl() {};
    TPipe pipe;
};
}

class test_matmul_load_to_l0b : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using L0cT = float;

    using A_TYPE = MatmulType<AscendC::TPosition::TSCM, CubeFormat::ND, half>;
    using B_TYPE = MatmulType<AscendC::TPosition::TSCM, CubeFormat::ND, half>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, L0cT>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM, void, CustomMatmulPolicy> mm;
};

TEST_F(test_matmul_load_to_l0b, case0) {
    // input: N : 16, K : 16, isTrans: false
    TBuf<TPosition::B2> l0bBuf;
    mm.pipe.InitBuffer(l0bBuf, 65536);
    auto l0b = l0bBuf.Get<half>();
    TBuf<TPosition::TSCM> l1bBuf;
    mm.pipe.InitBuffer(l1bBuf, 65536);
    auto l1b = l1bBuf.Get<half>();
    mm.Prepare(false, 16);
    mm.Load(l0b, l1b, 0, 0, 16, 16, 0, 0, false);
}
