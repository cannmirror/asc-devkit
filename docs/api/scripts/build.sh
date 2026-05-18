#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set -euo pipefail
cd "$(dirname "$0")/.."

echo -e "\033[36m==> 1/3 生成 TOC ...\033[0m"
python3 scripts/_gen_toc.py

echo -e "\033[36m==> 2/3 清理缓存 ...\033[0m"
rm -rf _build

echo -e "\033[36m==> 3/3 构建 HTML（自动后处理） ...\033[0m"
sphinx-build -b html -c scripts . _build/html

echo -e "\033[32mDone! 打开 _build/html/index.html 查看。\033[0m"
