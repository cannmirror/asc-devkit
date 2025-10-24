# Copyright (c) 2024 Huawei Technologies Co., Ltd. This file is a part of the
# CANN Open Software. Licensed under CANN Open Software License Agreement
# Version 1.0 (the "License"). Please refer to the License for details. You may
# not use this file except in compliance with the License. THIS SOFTWARE IS
# PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR
# FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software
# repository for the full text of the License.
# ===============================================================================

add_subdirectory(adv_api)
add_subdirectory(utils)

if(NOT BUILD_OPEN_PROJECT)
  install_package(
    PACKAGE
    tiling_api
    TARGETS
    tiling_api
    tiling_api_headers
    DIRECTORY
    ${ASCENDC_DIR}/include
    DESTINATION
    ${INSTALL_INCLUDE_DIR}/asc
    DIRECTORY
    ${ASCENDC_DIR}/impl
    DESTINATION
    ${INSTALL_INCLUDE_DIR}/asc
    FILES
    ${ASCENDC_DIR}/include/adv_api/tiling_api.h
    DESTINATION
    ${INSTALL_INCLUDE_DIR}/asc/include/adv_api)

  install(SCRIPT ${ASCENDC_DIR}/impl/adv_api/cmake/tiling_directory.cmake)
endif()

if(BUILD_OPEN_PROJECT)
  set(install_script_dir ${CMAKE_CURRENT_BINARY_DIR}/install_scripts/)

  add_custom_target(
    generate_install_script ALL
    COMMAND rm -rf ${install_script_dir}
    COMMAND cp -rf ${ASCENDC_DIR}/cmake/install ${install_script_dir}
    COMMAND chmod -R u+w ${install_script_dir}
    COMMAND echo "base_package=toolkit" > ${install_script_dir}/version.info
    COMMAND echo "backup_dir=${CMAKE_PROJECT_NAME}" >>
            ${install_script_dir}/version.info
    COMMAND echo "Version=${CANN_VERSION}" >>
            ${install_script_dir}/version.info)

  install(
    DIRECTORY ${install_script_dir}
    DESTINATION .
    FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_READ
                      OWNER_WRITE GROUP_WRITE WORLD_WRITE)

  set(CPACK_PACKAGE_NAME ${CMAKE_PROJECT_NAME})
  set(CPACK_PACKAGE_VERSION ${CMAKE_PROJECT_VERSION})
  set(CPACK_PACKAGE_DESCRIPTION "CPack ascendc-api-adv project")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "CPack ascendc-api-adv project")
  set(CPACK_PACKAGE_DIRECTORY ${CMAKE_BINARY_DIR})
  set(CPACK_PACKAGE_FILE_NAME
      "CANN-ascendc_api_adv-${CANN_VERSION}-linux.${CMAKE_SYSTEM_PROCESSOR}.run"
  )
  set(CPACK_GENERATOR External)
  set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
  set(CPACK_EXTERNAL_ENABLE_STAGING TRUE)
  set(CPACK_EXTERNAL_PACKAGE_SCRIPT
      ${ASCEND_CANN_PACKAGE_PATH}/tools/op_project_templates/ascendc/customize/cmake/makeself.cmake
  )
  set(CPACK_EXTERNAL_BUILT_PACKAGES
      ${CPACK_PACKAGE_DIRECTORY}/_CPack_Packages/Linux/External/${CPACK_PACKAGE_FILE_NAME}/${CPACK_PACKAGE_FILE_NAME}
  )
  include(CPack)
endif()

