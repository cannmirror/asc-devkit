# Copyright (c) 2024 Huawei Technologies Co., Ltd. This file is a part of the
# CANN Open Software. Licensed under CANN Open Software License Agreement
# Version 1.0 (the "License"). Please refer to the License for details. You may
# not use this file except in compliance with the License. THIS SOFTWARE IS
# PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR
# FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software
# repository for the full text of the License.
# ===============================================================================

add_library(intf_llt_pub INTERFACE)

target_include_directories(intf_llt_pub INTERFACE)

target_compile_definitions(intf_llt_pub INTERFACE CFG_BUILD_DEBUG
                                                  _GLIBCXX_USE_CXX11_ABI=0)

target_compile_options(
  intf_llt_pub
  INTERFACE -g
            -w
            $<$<BOOL:${ENABLE_GCOV}>:-fprofile-arcs
            -ftest-coverage>
            $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address
            -fno-omit-frame-pointer
            -static-libasan
            -fsanitize=undefined
            -static-libubsan
            -fsanitize=leak
            -static-libtsan>
            -fPIC
            -pipe)

target_link_options(
  intf_llt_pub
  INTERFACE
  $<$<BOOL:${ENABLE_GCOV}>:-fprofile-arcs
  -ftest-coverage>
  $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address
  -static-libasan
  -fsanitize=undefined
  -static-libubsan
  -fsanitize=leak
  -static-libtsan>)

target_link_directories(intf_llt_pub INTERFACE)

target_link_libraries(intf_llt_pub INTERFACE GTest::gtest -lpthread
                                             $<$<BOOL:${ENABLE_GCOV}>:-lgcov>)
