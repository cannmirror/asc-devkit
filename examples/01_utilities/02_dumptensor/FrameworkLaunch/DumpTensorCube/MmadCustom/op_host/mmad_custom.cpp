/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file mmad_custom.cpp
 * \brief
 */
#include "mmad_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace matmul_tiling;

namespace optiling {
/**
  * @brief  Generate mmad tiling.
  * @param  context: Tiling kernel context.
  * @retval Status of GetTiling (GRAPH_SUCCESS or GRAPH_FAILED).
  */
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    context->SetBlockDim(1);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class MmadCustom : public OpDef {
public:
    explicit MmadCustom(const char *name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Output("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend310p")
            .AddConfig("ascend910b");
    }
};

OP_ADD(MmadCustom);
} // namespace ops
