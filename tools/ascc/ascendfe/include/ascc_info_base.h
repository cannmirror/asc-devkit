/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ascc_info_base.h
 * \brief
 */

#ifndef __INCLUDE_ASCC_INFO_BASE_H__
#define __INCLUDE_ASCC_INFO_BASE_H__

namespace Ascc {

class AsccInfoBase {
public:
    AsccInfoBase() = default;
    virtual ~AsccInfoBase() = default;
    virtual bool IsEmpty() const = 0;
};

} // Ascc
#endif // __INCLUDE_ASCC_INFO_BASE_H__