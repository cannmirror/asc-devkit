#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent
with open(BASE / "_toc.yml", encoding="utf-8") as f:
    data = yaml.safe_load(f)
entries = data["entries"]

for e in entries:
    title = e.get("title", "")
    file = e.get("file", "")
    has_entries = bool(e.get("entries"))
    if has_entries:
        for sub in e["entries"][:5]:
            stitle = sub.get("title", "")
            sfile = sub.get("file", "")
            shentries = bool(sub.get("entries"))
