if(CUSTOM_ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH  ${CUSTOM_ASCEND_CANN_PACKAGE_PATH})
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH  $ENV{ASCEND_HOME_PATH})
elseif(DEFINED ENV{ASCEND_OPP_PATH})
    get_filename_component(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_OPP_PATH}/.." ABSOLUTE)
else()
    set(ASCEND_CANN_PACKAGE_PATH  "/usr/local/Ascend/ascend-toolkit/latest")
endif()

if (NOT EXISTS "${ASCEND_CANN_PACKAGE_PATH}")
    message(FATAL_ERROR "${ASCEND_CANN_PACKAGE_PATH} does not exist, please install the cann package and set environment variables.")
endif()

# execute_process(COMMAND bash ${ASCENDC_ADV_API_CMAKE_DIR}/scripts/check_version_compatiable.sh
#                              ${ASCEND_CANN_PACKAGE_PATH}
#                              toolkit
#                              ${ASCENDC_DIR}/version.info
#     RESULT_VARIABLE result
#     OUTPUT_STRIP_TRAILING_WHITESPACE
#     OUTPUT_VARIABLE CANN_VERSION
#     )

# if (result)
#     message(FATAL_ERROR "${CANN_VERSION}")
# else()
#      string(TOLOWER ${CANN_VERSION} CANN_VERSION)
# endif()

if (CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX     "${CMAKE_CURRENT_BINARY_DIR}/_CPack_Packages/makeself_staging"  CACHE STRING "path for install()" FORCE)
endif ()

set(HI_PYTHON                     "python3"                       CACHE   STRING   "python executor")
set(PRODUCT_SIDE                  host)
set(COMPILE_BASE_ON_SUBGROUP OFF BOOL)

if (NOT COMPILE_BASE_ON_SUBGROUP)

set(TILING_API_LIB ${ASCEND_CANN_PACKAGE_PATH}/lib64/libtiling_api.a)
if (NOT EXISTS "${TILING_API_LIB}")
    message(FATAL_ERROR "${TILING_API_LIB} does not exist, please check whether the toolkit package is installed.")
endif()

set(ASCENDC_API_ADV_OBJ      ascendc_api_adv_obj)
set(ASCENDC_API_ADV_OBJ_PATH ${CMAKE_CURRENT_BINARY_DIR}/ascendc_api_adv_objs)

file(REMOVE_RECURSE ${ASCENDC_API_ADV_OBJ_PATH})
file(MAKE_DIRECTORY ${ASCENDC_API_ADV_OBJ_PATH})

execute_process(
    COMMAND ${CMAKE_AR} -x ${TILING_API_LIB}
    WORKING_DIRECTORY ${ASCENDC_API_ADV_OBJ_PATH}
    RESULT_VARIABLE result
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )

add_library(${ASCENDC_API_ADV_OBJ} OBJECT IMPORTED)
set_target_properties(${ASCENDC_API_ADV_OBJ} PROPERTIES
    IMPORTED_OBJECTS "${ASCENDC_API_ADV_OBJ_PATH}/platform_ascendc.cpp.o;${ASCENDC_API_ADV_OBJ_PATH}/context_builder.cpp.o;${ASCENDC_API_ADV_OBJ_PATH}/context_builder_impl.cpp.o;${ASCENDC_API_ADV_OBJ_PATH}/template_argument.cpp.o"
    )
endif()

if (ENABLE_TEST)
    set(CMAKE_SKIP_RPATH FALSE)
else ()
    set(CMAKE_SKIP_RPATH TRUE)
endif ()


get_filename_component(ASCENDC_API_ADV_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
# include(${ASCENDC_API_ADV_CMAKE_DIR}/intf_pub_linux.cmake)
