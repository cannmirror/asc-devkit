include(${CMAKE_CURRENT_LIST_DIR}/modules/ascend_modules/config.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/modules/ascend_modules/func.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/modules/ascend_modules/intf.cmake)

# plugin support ASC language
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/ASC_CMake")
include(${CMAKE_CURRENT_LIST_DIR}/ASC_CMake/FindASC.cmake)