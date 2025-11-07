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
 * \file kernel_operator_set_spr_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_SET_SPR_IMPL_H
#define ASCENDC_MODULE_OPERATOR_SET_SPR_IMPL_H
namespace AscendC {
__aicore__ inline void SetQuantPreImpl(uint64_t config)
{
    set_quant_pre(config);
}

__aicore__ inline void SetNdParaImpl(uint64_t config) {
}

__aicore__ inline void SetFpcImpl(uint64_t config)
{
    set_fpc(config);
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_SET_SPR_IMPL_H