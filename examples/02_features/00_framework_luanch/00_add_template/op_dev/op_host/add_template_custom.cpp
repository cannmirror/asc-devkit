/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "register/op_def_registry.h"
#include "../op_kernel/add_template_custom_tiling.h"
#include "../op_kernel/tiling_key_add_template_custom.h"

namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t DEFAULT_TILE_NUM = 8;
constexpr int MIN_LENGTH_FOR_SPLIT = 2048;
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    ge::DataType dtype_x = context->GetInputDesc(0)->GetDataType();
    ge::DataType dtype_y = context->GetInputDesc(1)->GetDataType();
    ge::DataType dtype_z = context->GetOutputDesc(0)->GetDataType();
    uint32_t D_T_X = C_DT_FLOAT;
    uint32_t D_T_Y = C_DT_FLOAT;
    uint32_t D_T_Z = C_DT_FLOAT;
    uint32_t TILE_NUM = 1;
    uint32_t IS_SPLIT = 0;
    if (dtype_x == ge::DataType::DT_FLOAT) {
        D_T_X = C_DT_FLOAT;
    } else if (dtype_x == ge::DataType::DT_FLOAT16) {
        D_T_X = C_DT_FLOAT16;
    }
    if (dtype_y == ge::DataType::DT_FLOAT) {
        D_T_Y = C_DT_FLOAT;
    } else if (dtype_y == ge::DataType::DT_FLOAT16) {
        D_T_Y = C_DT_FLOAT16;
    }
    if (dtype_z == ge::DataType::DT_FLOAT) {
        D_T_Z = C_DT_FLOAT;
    } else if (dtype_z == ge::DataType::DT_FLOAT16) {
        D_T_Z = C_DT_FLOAT16;
    }
    if (totalLength < MIN_LENGTH_FOR_SPLIT) {
        IS_SPLIT = 0;
        TILE_NUM = 1;
    } else {
        IS_SPLIT = 1;
        TILE_NUM = DEFAULT_TILE_NUM;
    }
    context->SetBlockDim(BLOCK_DIM);
    TilingDataTemplate *tiling = context->GetTilingData<TilingDataTemplate>();
    tiling->totalLength = totalLength;
    context->GetRawTilingData()->SetDataSize(sizeof(TilingDataTemplate));
    const uint64_t tilingKey = GET_TPL_TILING_KEY(D_T_X, D_T_Y, D_T_Z, TILE_NUM, IS_SPLIT);  // 模板参数tilingkey配置
    context->SetTilingKey(tilingKey);
    ASCENDC_TPL_SEL_PARAM(context, D_T_X, D_T_Y, D_T_Z, TILE_NUM, IS_SPLIT);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x1_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class AddTemplateCustom : public OpDef {
public:
    explicit AddTemplateCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(AddTemplateCustom);
}  // namespace ops
