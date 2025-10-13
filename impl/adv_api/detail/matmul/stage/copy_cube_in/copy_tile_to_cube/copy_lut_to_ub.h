/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file copy_lut_to_ub.h
 * \brief
 */

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_COPY_LUT_TO_UB_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_COPY_LUT_TO_UB_H

namespace AscendC {
namespace Impl {
namespace Detail {
template <typename IMPL, const auto &MM_CFG, typename = void>
class QtableProcessor {
public:
    __aicore__ inline QtableProcessor() = default;
    __aicore__ inline ~QtableProcessor() = default;

    /**
     * @description: Init UB buffer
     * @return: void
     */
    __aicore__ inline void Init(uint32_t qtableNum)
    {}

    /**
     * @description: Set qtable tensor params
     * @param: qtable tensor params
     * @return: void
     */
    __aicore__ inline void SetLookupTable(const GlobalTensor<uint64_t> &qtableTensor)
    {}

    /**
     * @description: Get qtable buf
     * @param: index: the index of qtable buf
     * @return: qtable
     */
    __aicore__ inline uint64_t GetQtable(uint16_t index) const
    {
        return 0;
    }
};


template <typename IMPL, const auto &MM_CFG>
class QtableProcessor<IMPL, MM_CFG, AscendC::enable_if_t<IsDecompMode<MM_CFG>(), void>> {
public:
    __aicore__ inline QtableProcessor() = default;
    __aicore__ inline ~QtableProcessor() = default;

    /**
     * @description: Init UB buffer
     * @return: void
     */
    __aicore__ inline void Init(uint32_t qtableNum)
    {
        byteSize_ = CeilAlign(qtableNum * sizeof(uint64_t), ONE_BLK_SIZE);
        GetTPipePtr()->InitBuffer(qidQtable_, byteSize_);
    }
    /**
     * @description: Set qtable tensor params
     * @param: qtable tensor params
     * @return: void
     */
    __aicore__ inline void SetLookupTable(const GlobalTensor<uint64_t> &qtableTensor)
    {
        qtableTensor_ = qidQtable_.Get<uint64_t>();
        DataCopy(qtableTensor_, qtableTensor, byteSize_ / sizeof(uint64_t));
        TEventID enQueEvtID = GetTPipePtr()->FetchEventID(HardEvent::MTE2_S);
        SetFlag<HardEvent::MTE2_S>(enQueEvtID);
        WaitFlag<HardEvent::MTE2_S>(enQueEvtID);
    }

    /**
     * @description: Get qtable buf
     * @param: index: the index of qtable buf
     * @return: qtable
     */
    __aicore__ inline uint64_t GetQtable(uint16_t index) const
    {
        return qtableTensor_.GetValue(static_cast<uint32_t>(index));
    }

private:
    TBuf<TPosition::VECIN> qidQtable_;
    LocalTensor<uint64_t> qtableTensor_;
    uint32_t byteSize_ = 0;
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif  // IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_COPY_LUT_TO_UB_H