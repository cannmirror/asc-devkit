/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @brief mock modules for kfc
 *
 */

#include <gtest/gtest.h>
#include "kernel_operator.h"
#include "include/adv_api/matmul/tiling.h"
#include "impl/adv_api/detail/matmul/utils/matmul_param.h"
using namespace std;

namespace AscendC {
template <typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG,
    PolicyType POLICY_TYPE = PolicyType::MATMUL_DEFAULT, typename = void>
class CustomMatmulScheduler
{
    using DstT = typename C_TYPE::T;
public:
    __aicore__ inline void Init(const TCubeTiling *__restrict cubeTiling, TPipe *tpipe) {}
    __aicore__ inline bool ScheduleOnce(bool enPartialSum) { return true; }
    __aicore__ inline void GetResult(const GlobalTensor<DstT> &gm, uint8_t enAtomic = 0,
                                     bool enSequentialWrite = false) {}
    __aicore__ inline void End() {}
    __aicore__ inline void Reset() {}
};
 
template <typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG>
class CustomBatchScheduler
{
public:
    __aicore__ inline void Init(const TCubeTiling *__restrict cubeTiling, TPipe *tpipe) {}
    template<class T>
    void Schedule(const T& dst, bool enPartialSum, uint8_t enAtomic, bool enSequentialWrite,
        const uint32_t matrixStrideA, const uint32_t matrixStrideB, const uint32_t matrixStrideC) {};
    __aicore__ inline void End() {}
};
}
