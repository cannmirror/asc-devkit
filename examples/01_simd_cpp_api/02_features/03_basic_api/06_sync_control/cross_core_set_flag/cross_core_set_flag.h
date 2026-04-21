/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file cross_core_set_flag.h
 * \brief
 */

#include "acl/acl.h"
#include "kernel_operator.h"
#include "data_utils.h"
#include <iostream>
#include <vector>
#include <iterator>
#include "acl/acl.h"
#include "kernel_operator.h"

constexpr uint32_t BUFFER_NUM = 1;

constexpr uint32_t M = 32;
constexpr uint32_t N = 64;
constexpr uint32_t K = 32;
constexpr uint32_t NUM_BLOCKS = 8;

constexpr uint32_t A_BLOCKS_LENGTH = M * K / NUM_BLOCKS;
constexpr uint32_t B_BLOCKS_LENGTH = K / NUM_BLOCKS * N;
constexpr uint32_t C_AIC_BLOCKS_LENGTH = M * N;
constexpr uint32_t C_AIV_BLOCKS_LENGTH = M / (NUM_BLOCKS * 2) * N;

// 模式2的flagId,AIC等AIV
constexpr uint16_t SYNC_AIV_AIC_FLAG = 12;
// 模式0的flagId
constexpr uint16_t SYNC_AIC_FLAG = 11;
// 模式2的flagId,AIV等AIC
constexpr uint16_t SYNC_AIC_AIV_FLAG = 13;

class KernelMmad {
public:
    __aicore__ inline KernelMmad()
    {
        //左矩阵分形的shape
        fractalShape[0] = 16;
        fractalShape[1] = 32 / sizeof(half);
        fractalSize = 16 * fractalShape[1];
        fractalNum = 1;
        blockIdx = AscendC::GetBlockIdx(); // 获取当前工作的核ID
    }
    __aicore__ inline void InitAIC(GM_ADDR A, GM_ADDR B, GM_ADDR C)
    {
        ACUBEGM.SetGlobalBuffer((__gm__ half *)A + A_BLOCKS_LENGTH * AscendC::GetBlockIdx(), A_BLOCKS_LENGTH);
        BCUBEGM.SetGlobalBuffer((__gm__ half *)B + B_BLOCKS_LENGTH * AscendC::GetBlockIdx(), B_BLOCKS_LENGTH);
        CCUBEGM.SetGlobalBuffer((__gm__ float *)C, C_AIC_BLOCKS_LENGTH);
    }

    __aicore__ inline void InitAIV(GM_ADDR a, GM_ADDR b, GM_ADDR A, GM_ADDR B, GM_ADDR C)
    {
        aGM.SetGlobalBuffer((__gm__ uint8_t *)a + A_BLOCKS_LENGTH * (AscendC::GetBlockIdx() / 2), A_BLOCKS_LENGTH);
        bGM.SetGlobalBuffer((__gm__ uint8_t *)b + B_BLOCKS_LENGTH * (AscendC::GetBlockIdx() / 2), B_BLOCKS_LENGTH);
        AVectorGM.SetGlobalBuffer((__gm__ half *)A + A_BLOCKS_LENGTH * (AscendC::GetBlockIdx() / 2), A_BLOCKS_LENGTH);
        BVectorGM.SetGlobalBuffer((__gm__ half *)B + B_BLOCKS_LENGTH * (AscendC::GetBlockIdx() / 2), B_BLOCKS_LENGTH);
        CVectorGM.SetGlobalBuffer((__gm__ float *)C + C_AIV_BLOCKS_LENGTH * AscendC::GetBlockIdx(), C_AIV_BLOCKS_LENGTH);
    }
    __aicore__ inline void ProcessAIC()
    {
        // 构造静态tensor
        AscendC::LocalTensor<half> a1Local(AscendC::TPosition::A1, a1LocalAddr, aSizeAlignL1);
        AscendC::LocalTensor<half> b1Local(AscendC::TPosition::B1, b1LocalAddr, bSizeAlignL1);
        AscendC::LocalTensor<half> a2Local(AscendC::TPosition::A2, a2LocalAddr, aSizeAlignL0);
        AscendC::LocalTensor<half> b2Local(AscendC::TPosition::B2, b2LocalAddr, bSizeAlignL0);
        AscendC::LocalTensor<float> c1Local(AscendC::TPosition::CO1, c1LocalAddr, cSizeAlignL0);


        // 模式2,每一个AICore内部，AIC等2个AIV
        AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG);

        CopyIn(a1Local, b1Local);
        SplitA(a1Local, a2Local);
        SplitBTranspose(b1Local, b2Local);
        Compute(a2Local, b2Local, c1Local);
        CopyOut(c1Local);
        // 模式0,8个AICore 包含的8个AIC同步
        AscendC::CrossCoreSetFlag<0, PIPE_FIX>(SYNC_AIC_FLAG);  
        AscendC::CrossCoreWaitFlag(SYNC_AIC_FLAG);  

        // 模式2,每一个AICore内部，2个AIV等AIC
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_AIV_FLAG);
    }

    __aicore__ inline void ProcessAIV()
    {
        // 构造静态tensor
        AscendC::LocalTensor<uint8_t> aLocal(AscendC::TPosition::VECIN, aAddr, A_BLOCKS_LENGTH);
        AscendC::LocalTensor<uint8_t> bLocal(AscendC::TPosition::VECIN, bAddr, B_BLOCKS_LENGTH);
        AscendC::LocalTensor<float> cLocal(AscendC::TPosition::VECIN, cAddr, C_AIV_BLOCKS_LENGTH);
        AscendC::LocalTensor<half> castALocal(AscendC::TPosition::VECOUT, castAAddr, A_BLOCKS_LENGTH);
        AscendC::LocalTensor<half> castBLocal(AscendC::TPosition::VECOUT, castBAddr, B_BLOCKS_LENGTH);
        AscendC::LocalTensor<float> reluCLocal(AscendC::TPosition::VECOUT, reluCAddr, C_AIV_BLOCKS_LENGTH);

        if (blockIdx % 2 == 0) {
            AscendC::DataCopy(aLocal, aGM, A_BLOCKS_LENGTH);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Cast(castALocal, aLocal, AscendC::RoundMode::CAST_NONE, A_BLOCKS_LENGTH);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopy(AVectorGM, castALocal, A_BLOCKS_LENGTH);
        } else {
            AscendC::DataCopy(bLocal, bGM, B_BLOCKS_LENGTH);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Cast(castBLocal, bLocal, AscendC::RoundMode::CAST_NONE, B_BLOCKS_LENGTH);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopy(BVectorGM, castBLocal, B_BLOCKS_LENGTH);
        }

        // 模式2,每一个AICore内部，AIC等2个AIV
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);

        // 模式2,每一个AICore内部，2个AIV等AIC
        AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);

        // 进行LeakyRelu运算
        float alpha = 0.001;
        AscendC::DataCopy(cLocal, CVectorGM, C_AIV_BLOCKS_LENGTH); 
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::LeakyRelu(reluCLocal, cLocal, alpha, C_AIV_BLOCKS_LENGTH);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(CVectorGM, reluCLocal, C_AIV_BLOCKS_LENGTH);
    }

private:
    __aicore__ inline uint16_t CeilDivision(uint16_t numerator, uint16_t denominator) 
    {
        return (numerator + denominator - 1) / denominator;
    }

    __aicore__ inline uint16_t CeilAlign(uint16_t numerator, uint16_t denominator) 
    {
        return (numerator + denominator - 1) / denominator * denominator;
    }
    __aicore__ inline void CopyIn(AscendC::LocalTensor<half>& a1Local, AscendC::LocalTensor<half>& b1Local)
    {
        AscendC::Nd2NzParams nd2nzA1Params;
        nd2nzA1Params.ndNum = 1;
        nd2nzA1Params.nValue = m;
        nd2nzA1Params.dValue = k;
        nd2nzA1Params.srcNdMatrixStride = 0;
        nd2nzA1Params.srcDValue = k;
        nd2nzA1Params.dstNzC0Stride = CeilAlign(m, fractalShape[0]);

        nd2nzA1Params.dstNzNStride = 1;
        nd2nzA1Params.dstNzMatrixStride = 0;
        AscendC::DataCopy(a1Local, ACUBEGM, nd2nzA1Params);
        AscendC::Nd2NzParams nd2nzB1Params;
        nd2nzB1Params.ndNum = 1;
        nd2nzB1Params.nValue = k;
        nd2nzB1Params.dValue = n;
        nd2nzB1Params.srcNdMatrixStride = 0;
        nd2nzB1Params.srcDValue = n;
        nd2nzB1Params.dstNzC0Stride = CeilAlign(k, fractalShape[0]);

        nd2nzB1Params.dstNzNStride = 1;
        nd2nzB1Params.dstNzMatrixStride = 0;
        AscendC::DataCopy(b1Local, BCUBEGM, nd2nzB1Params);
    }
    __aicore__ inline void SplitA(AscendC::LocalTensor<half>& a1Local, AscendC::LocalTensor<half>& a2Local)
    {
        // MTE2与MTE1流水在L1上存在写后读依赖
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);

        uint32_t dstOffset = CeilDivision(k, fractalShape[1]) * fractalSize;
        uint32_t srcOffset = fractalSize;
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = CeilDivision(k, fractalShape[1]);
        loadDataParams.srcStride = CeilDivision(m, fractalShape[0]);
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;
        for (int i = 0; i < CeilDivision(m, fractalShape[0]); ++i) {
            AscendC::LoadData(a2Local[i * dstOffset], a1Local[i * srcOffset], loadDataParams);
        }
    }
    __aicore__ inline void SplitBTranspose(AscendC::LocalTensor<half>& b1Local, AscendC::LocalTensor<half>& b2Local)
    {
        uint32_t dstOffset = CeilDivision(n, fractalShape[0] * fractalNum) * fractalSize * fractalNum;
        uint32_t srcOffset = fractalSize * fractalNum;
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = CeilDivision(n, fractalShape[0] * fractalNum);
        loadDataParams.srcStride = CeilDivision(k, fractalShape[0] * fractalNum);
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;
        for (int i = 0; i < CeilDivision(k, fractalShape[0] * fractalNum); ++i) {
            AscendC::LoadData(b2Local[i * dstOffset], b1Local[i * srcOffset], loadDataParams);
        }
    }
    __aicore__ inline void Compute(AscendC::LocalTensor<half>& a2Local, AscendC::LocalTensor<half>& b2Local, AscendC::LocalTensor<float>& c1Local)
    {
        // M与MTE1流水在L0A、L0B上存在写后读依赖
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams);
    }
    __aicore__ inline void CopyOut(AscendC::LocalTensor<float>& c1Local)
    {
         // M与FIX流水在L0C存在写后读依赖
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);

        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = n;
        fixpipeParams.mSize = m;
        fixpipeParams.srcStride = CeilAlign(m, fractalShape[0]);
        fixpipeParams.dstStride = n;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        // 对L0C-->GM搬出的数据，启用原子累加（分块矩阵乘结果累加得到完整矩阵乘结果）
        AscendC::SetAtomicAdd<float>();
        AscendC::Fixpipe(CCUBEGM, c1Local, fixpipeParams);
        // 清空原子操作
        AscendC::DisableDmaAtomic();
    }

private:
    AscendC::GlobalTensor<uint8_t> aGM;
    AscendC::GlobalTensor<uint8_t> bGM;
    AscendC::GlobalTensor<half> AVectorGM;
    AscendC::GlobalTensor<half> BVectorGM;
    AscendC::GlobalTensor<float> CVectorGM;
    AscendC::GlobalTensor<float> CCUBEGM;
    AscendC::GlobalTensor<half> ACUBEGM;
    AscendC::GlobalTensor<half> BCUBEGM;

    uint16_t m = M, k = K / NUM_BLOCKS, n = N;

    // 对齐后的shape
    uint16_t aSizeAlignL1 = CeilAlign((uint16_t)m, fractalShape[0]) * CeilAlign((uint16_t)k, fractalShape[1]);
    uint16_t aSizeAlignL0 = CeilAlign((uint16_t)m, fractalShape[0]) * CeilAlign((uint16_t)k, fractalShape[1]);

    uint16_t bSizeAlignL1 = CeilAlign((uint16_t)k, fractalShape[1]) * CeilAlign((uint16_t)n, fractalShape[1]);
    uint16_t bSizeAlignL0 = CeilAlign((uint16_t)k, fractalShape[1]) * CeilAlign((uint16_t)n, fractalShape[1]);

    uint16_t cSizeAlignL0 = CeilAlign((uint16_t)m, fractalShape[0]) * CeilAlign((uint16_t)n, fractalShape[0]);

    uint16_t fractalShape[2] = {0, 0};
    uint16_t fractalSize = 0;
    uint16_t fractalNum = 0;
    int32_t blockIdx = 0;
    uint32_t  a1LocalAddr = 0;
    uint32_t  b1LocalAddr = aSizeAlignL1;
    uint32_t  a2LocalAddr = 0;
    uint32_t  b2LocalAddr = 0;
    uint32_t  c1LocalAddr = 0;

    uint32_t  aAddr = 0;
    uint32_t  bAddr = A_BLOCKS_LENGTH;
    uint32_t  cAddr = A_BLOCKS_LENGTH + B_BLOCKS_LENGTH;
    uint32_t  castAAddr = 0;
    uint32_t  castBAddr = A_BLOCKS_LENGTH;
    uint32_t  reluCAddr = A_BLOCKS_LENGTH + B_BLOCKS_LENGTH;
};

class KernelCrossCoreSetFlag {
public:
    __aicore__ inline KernelCrossCoreSetFlag() {}
    __aicore__ inline void Init(GM_ADDR initialData, GM_ADDR atomicResult, uint32_t totalLength)
    {
        this->blockLength = totalLength;
        initialDataGm.SetGlobalBuffer((__gm__ float *)initialData, this->blockLength);
        atomicResultGm.SetGlobalBuffer((__gm__ float *)atomicResult, this->blockLength);
    }

    __aicore__ inline void Process()
    {
        if constexpr (SCENARIO_NUM == 0) {
            // 模式0
            ProcessMode0();
        } else if constexpr (SCENARIO_NUM == 1) {
            // 模式1
            ProcessMode1();
        }
    }
    __aicore__ inline void ProcessMode0()
    {
        const uint32_t xAddr = 0;                                   
        const uint32_t yAddr = this->blockLength * sizeof(float);   

        AscendC::LocalTensor<float> xLocal(AscendC::TPosition::VECCALC, xAddr, this->blockLength);
        AscendC::LocalTensor<float> yLocal(AscendC::TPosition::VECCALC, yAddr, this->blockLength);
                     
        AscendC::DataCopy(xLocal, initialDataGm, this->blockLength); 
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        AscendC::Muls(xLocal, xLocal, float(AscendC::GetBlockIdx()), this->blockLength);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        // UB 到 GM 搬运启用原子累加：搬运至 atomicResult 的数据与原值累加后覆盖原值
        AscendC::SetAtomicAdd<float>(); 
        // DataCopy属于PIPE_MTE3流水操作
        AscendC::DataCopy(atomicResultGm, xLocal, this->blockLength);   
        // 当本AIV完成前置PIPE_MTE3(DataCopy)流水操作后，通知其他AIV核，本AIV已经完成
        AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(0);  
        // 阻塞本AIV继续往下执行指令，直到其他AIV全部都完成PIPE_MTE3流水操作，才解除阻塞往下执行。
        AscendC::CrossCoreWaitFlag(0);               
        // 关闭原子累加
        AscendC::SetAtomicNone();  

        if (AscendC::GetBlockIdx() == 0) {
            AscendC::DataCopy(yLocal, atomicResultGm, this->blockLength);   // PIPE_MTE2
            AscendC::printf("============== In PrintTensor Process AIV %d ==============", AscendC::GetBlockIdx());
            AscendC::DumpTensor(yLocal, AscendC::GetBlockIdx(), this->blockLength);
            AscendC::DataCopy(atomicResultGm, yLocal, this->blockLength);
            return;
        }
    }
    __aicore__ inline void ProcessMode1()
    {
        // 16个aiv，GetBlockIdx取值为0-15。
         if ((AscendC::GetBlockIdx() == 2)||(AscendC::GetBlockIdx() == 3)) {
            const uint32_t xAddr = 0;                                   
            const uint32_t yAddr = this->blockLength * sizeof(float);  

            AscendC::LocalTensor<float> xLocal(AscendC::TPosition::VECCALC, xAddr, this->blockLength);
            AscendC::LocalTensor<float> yLocal(AscendC::TPosition::VECCALC, yAddr, this->blockLength);

            AscendC::DataCopy(xLocal, initialDataGm, this->blockLength);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            AscendC::Muls(xLocal, xLocal, float(AscendC::GetBlockIdx()), this->blockLength);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            AscendC::SetAtomicAdd<float>();
            AscendC::DataCopy(atomicResultGm, xLocal, this->blockLength);
            AscendC::CrossCoreSetFlag<1, PIPE_MTE3>(0);  
            AscendC::CrossCoreWaitFlag(0);             
            AscendC::SetAtomicNone();

            if (AscendC::GetBlockIdx() == 2) {
                AscendC::DataCopy(yLocal, atomicResultGm, this->blockLength); 
                AscendC::printf("============== In PrintTensor Process AIV %d ==============", AscendC::GetBlockIdx());
                AscendC::DumpTensor(yLocal, AscendC::GetBlockIdx(), this->blockLength);
                AscendC::DataCopy(atomicResultGm, yLocal, this->blockLength);
                return;
            }
        }
    }

private:
    AscendC::GlobalTensor<float> initialDataGm;
    AscendC::GlobalTensor<float> atomicResultGm;
    uint32_t blockLength;
};

uint32_t VerifyResult(std::vector<float> &output, std::vector<float> &golden)
{
    auto printTensor = [](std::vector<float> &tensor, const char *name) {
        constexpr size_t maxPrintSize = 20;
        std::cout << name << ": ";
        std::copy(tensor.begin(), tensor.begin() + std::min(tensor.size(), maxPrintSize),
            std::ostream_iterator<float>(std::cout, " "));
        if (tensor.size() > maxPrintSize) {
            std::cout << "...";
        }
        std::cout << std::endl;
    };
    if (std::equal(output.begin(), output.end(), golden.begin())) {
        std::cout << "test pass!" << std::endl;
        return 0;
    } else {
        std::cout << "test failed!" << std::endl;
        printTensor(output, "Output");
        printTensor(golden, "Golden");
        return 1;
    }
    return 0;
}