#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
class get_buffer_manager:
    def __init__(self):
        self.__l1_fusion_type = "l1_fusion_type"
        self.__rbs = "rbs"
        self.__tensor_list_index = "tensor_list_index"

    def set_l1_fusion_type(self, l1_fusion_type):
        return l1_fusion_type

    def set_remapped_buffers(self, rbs):
        return rbs

    def set_tensor_index(self, tensor_list_index):
        return tensor_list_index

class RemappedBuffer:
    pass
