/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*
* The code snippet comes from Huawei's open-source Ascend project.
* Copyright 2020 Huawei Technologies Co., Ltd.
* Licensed under the Apache License, Version 2.0 (the "License");
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*/
#ifndef OPERATOR_DESC_H
#define OPERATOR_DESC_H

#include <string>
#include <vector>

#include "acl/acl.h"

/**
 * Op description
 */
struct OperatorDesc {
    /**
     * Constructor
     * @param [in] opType: op type
     */
    explicit OperatorDesc(std::string opType);

    /**
     * Destructor
     */
    virtual ~OperatorDesc();

    /**
     * Add an input tensor description
     * @param [in] dataType: data type
     * @param [in] numDims: number of dims
     * @param [in] dims: dims
     * @param [in] format: format
     * @return OperatorDesc
     */
    OperatorDesc &AddInputTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format);

    /**
     * Add an output tensor description
     * @param [in] dataType: data type
     * @param [in] numDims: number of dims
     * @param [in] dims: dims
     * @param [in] format: format
     * @return OperatorDesc
     */
    OperatorDesc &AddOutputTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format);

    std::string opType;
    std::vector<aclTensorDesc *> inputDesc;
    std::vector<aclTensorDesc *> outputDesc;
    aclopAttr *opAttr;
};

#endif // OPERATOR_DESC_H
