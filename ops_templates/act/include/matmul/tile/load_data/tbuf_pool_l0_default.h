/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tbuf_pool_l0_default.h
 * \brief
 */
#ifndef ACT_INCLUDE_MATMUL_TILE_LOAD_DATA_TBUF_POOL_L0_DEFAULT_H
#define ACT_INCLUDE_MATMUL_TILE_LOAD_DATA_TBUF_POOL_L0_DEFAULT_H

namespace Act {
namespace Gemm {
namespace Tile {
class TBufPoolL0 {
public:
    __aicore__ inline TBufPoolL0(){};
    __aicore__ inline ~TBufPoolL0(){};

    __aicore__ inline void Init(bool isL0Db = true)
    {
        useL0PingPong_ = static_cast<uint16_t>(isL0Db);
        GetTPipePtr()->InitBuffer(l0aBuf_, L0A_SIZE);
        GetTPipePtr()->InitBuffer(l0bBuf_, L0B_SIZE);
    }

    __aicore__ inline void SetDBFlag(bool isL0Db = true)
    {
        useL0PingPong_ = static_cast<uint16_t>(isL0Db);
    }

    __aicore__ inline TBufPoolL0& Allocate()
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPongFlag_);
        return *this;
    }

    template <AscendC::TPosition Pos, typename T>
    __aicore__ inline AscendC::LocalTensor<T> GetBuffer(uint8_t subIdx = 0)
    {
        AscendC::LocalTensor<typename T::LiteType> tempTensor;
        if constexpr (Pos == AscendC::TPosition::A2) {
            tempTensor = l0aBuf_.Get<typename T::LiteType>();
            if (l0PingPongFlag_ != 0) {
                tempTensor = tempTensor[L0A_SIZE / DOUBLE_BUFFER_COUNT / sizeof(typename T::LiteType)];
            }
        } else {
            tempTensor = l0bBuf_.Get<typename T::LiteType>();
            if (l0PingPongFlag_ != 0) {
                tempTensor = tempTensor[L0B_SIZE / DOUBLE_BUFFER_COUNT / sizeof(typename T::LiteType)];
            }
        }
        AscendC::LocalTensor<T> retTensor;
        retTensor.SetAddr(tempTensor.address_);
        return retTensor;
    }

    template <AscendC::TPosition Pos>
    __aicore__ inline bool Hit(uint32_t pos = 0)
    {
        return false;
    }

    __aicore__ inline void ResetCache() {}

    __aicore__ inline void EnQue()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPongFlag_);
    }

    __aicore__ inline void DeQue()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPongFlag_);
    }

    __aicore__ inline void Free()
    {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPongFlag_);
        l0PingPongFlag_ = useL0PingPong_ - l0PingPongFlag_;
    }

private:
    AscendC::TBuf<AscendC::TPosition::A2> l0aBuf_;
    AscendC::TBuf<AscendC::TPosition::B2> l0bBuf_;
    uint16_t l0PingPongFlag_{0};
    uint16_t useL0PingPong_{1};
};

} // namespace Tile
} // namespace Gemm
} // namespace Act
#endif // ACT_INCLUDE_MATMUL_TILE_LOAD_DATA_TBUF_POOL_L0_DEFAULT_H
