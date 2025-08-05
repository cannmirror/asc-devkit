/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_ELEWISE_COMMON_H
#define ATVC_ELEWISE_COMMON_H

namespace ATVC {
struct EleWiseTilingData {
    uint32_t tailBlockCnt; // 需要额外执行一次循环的核的数量
    uint32_t tailElemCnt; // 尾块元素数量
    uint32_t numPerBlock; // 每个核需计算的总元素数量
    uint32_t tiledCnt;    // 每次搬入搬出的元素个数
    uint32_t blockNum;    // 执行核数
};

struct EleWiseParam {
    EleWiseTilingData tilingData;   // 影响数据搬运的相关参数
    uint32_t totalCnt = 0;          // 单Tensor的元素个数
    uint32_t nBufferNum = 2;        // 每个Queue中的Tensor数量
};
}

#endif
