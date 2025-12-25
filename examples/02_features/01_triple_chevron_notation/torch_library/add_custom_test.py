#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import os
import torch
import torch_npu
import torchair
from torch_npu.testing.testcase import TestCase, run_tests
torch.ops.load_library("libcustom_ops.so")


class SingleOpModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        return torch.ops.ascendc_ops.ascendc_add(x, y)


class TestCustomAdd(TestCase):
    def test_add_custom_ops(self):
        config = torchair.CompilerConfig()
        config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(SingleOpModel().npu(), backend=npu_backend)

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization
        )

        profiler = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.NPU,
                torch_npu.profiler.ProfilerActivity.CPU,
            ],
            with_stack=False,
            record_shapes=False,
            profile_memory=False,
            experimental_config=experimental_config,
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=10, repeat=1, skip_first=0),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./res")
        )

        length = [8, 2048]
        x = torch.rand(length, device='cpu', dtype=torch.float)
        y = torch.rand(length, device='cpu', dtype=torch.float)
        profiler.start()
        output = model(x.npu(), y.npu()).cpu()
        profiler.stop()

        cpuout = torch.add(x, y)
        self.assertRtolEqual(output, cpuout)


if __name__ == "__main__":
    run_tests()
