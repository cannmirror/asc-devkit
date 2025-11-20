/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef C_API_INTERF_UTIL_H
#define C_API_INTERF_UTIL_H

union DeqScaleConfig
{
    uint64_t val;
    struct {
        uint64_t reserved1 : 5;
        uint64_t offset : 9;
        uint64_t signMode : 1;
        uint64_t reserved2 : 17;
        float scale;
    };
};

#endif