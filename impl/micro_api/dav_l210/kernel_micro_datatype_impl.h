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
 * \file kernel_micro_datatype_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_DATATYPE_IMPL_H
#define ASCENDC_MODULE_MICRO_DATATYPE_IMPL_H

namespace AscendC {
template <typename T> struct TypeGet;
template <> struct TypeGet<uint32_t> {
    using T = vector_u32;
};
template <> struct TypeGet<int32_t> {
    using T = vector_s32;
};
template <> struct TypeGet<float> {
    using T = vector_f32;
};
template <> struct TypeGet<uint16_t> {
    using T = vector_u16;
};
template <> struct TypeGet<half> {
    using T = vector_f16;
};
template <> struct TypeGet<int16_t> {
    using T = vector_s16;
};
template <> struct TypeGet<uint8_t> {
    using T = vector_u8;
};
template <> struct TypeGet<int8_t> {
    using T = vector_s8;
};
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_DATATYPE_IMPL_H