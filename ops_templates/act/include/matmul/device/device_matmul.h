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
 * \file device_matmul.h
 * \brief
 */
#ifndef MATMUL_DEVICE_DEVICE_MATMUL_H
#define MATMUL_DEVICE_DEVICE_MATMUL_H

#include "kernel_operator.h"
#include "../kernel/kernel_matmul_mix_workspace.h"
#include "../kernel/kernel_matmul.h"
#include "../kernel/kernel_sparse_matmul.h"

namespace Act {
namespace Gemm {
namespace Device {
template <class MatmulKernel>
__global__ __aicore__ void KernelFunc(typename MatmulKernel::Params params)
{
    // For quantized types, switch to mixed-precision operators
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::TPipe tPipe;
    using Kernel = MatmulKernel;
    Kernel op;
    op(params);
}

template <class MatmulKernel>
class DeviceMatmul {
public:
    using Arguments = typename MatmulKernel::Arguments;
    using Params = typename MatmulKernel::Params;
    Params params_{};

    static size_t GetWorkspaceSize(Arguments& args)
    {
        size_t workspaceSize = GetSysWorkspaceSize();
        int64_t blockNum = GetBlockNum(args);
        workspaceSize += MatmulKernel::GetWorkspaceSize(args.problemShape, blockNum);
        return workspaceSize;
    }

    Status CanImplement(Arguments& args)
    {
        return MatmulKernel::CanImplement(args);
    }

    void InitParams(Arguments& args, GM_ADDR workspace)
    {
        params_ = MatmulKernel::InitParams(args, workspace);
        return;
    }

    static int64_t GetBlockNum(Arguments& args)
    {
        int64_t blockNum = MatmulKernel::GetBlockNum(args.problemShape);
        return blockNum;
    }

    void operator()(bool isQuant = false, void* stream = nullptr)
    {
        int64_t blockNum = MatmulKernel::GetBlockNum(params_.problemShape);
        KernelFunc<MatmulKernel><<<blockNum, nullptr, stream>>>(params_);
    }
};
} // namespace Device
} // namespace Gemm
} // namespace Act
#endif