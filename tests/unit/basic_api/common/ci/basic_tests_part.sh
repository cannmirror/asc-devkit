#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set -e

test_one_targets=(
  tikcpp_utest_ascend910
  # tikcpp_utest_compile_trace_log
  # tikcpp_utest_ascend910
  # tikcpp_utest_ascend610
  # tikcpp_utest_ascend310p
  # tikcpp_utest_ascend310B1
  # tikcpp_utest_ascend610Lite
  # tikcpp_tpl_tiling_key_utest_debug
  # tikcpp_tpl_tiling_key_utest_release
)

test_two_targets=(
  tikcpp_utest_ascend910
  # ascendc_utest_kernel
  # ascendc_utest_aclrtc
  # tikcpp_utest_ascend910B1_AIC
  # tikcpp_utest_ascend910B1_AIV
  # tikcpp_tiling_utest_ascend910
  # tikcpp_tiling_utest_ascend610
  # tikcpp_tiling_utest_ascend310p
)

test_three_targets=(
  tikcpp_utest_ascend910
  # tikcpp_tiling_utest_ascend910B1_AIC
  # tikcpp_tiling_utest_ascend910B1_AIV
  # tikcpp_tiling_utest_ascend310B1
  # tikcpp_tiling_utest_ascend610Lite
)