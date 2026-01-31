# function(create_opensource target_name suffix_name product_side install_prefix toolchain_file)

set(open_source_target_name mockcpp)
set(MOCKCPP_DOWNLOAD_PATH ${CANN_3RD_LIB_PATH}/pkg)
set(MOCKCPP_SRC_PATH ${CANN_3RD_LIB_PATH}/mockcpp)
set(mockcpp_SOURCE_DIR ${MOCKCPP_SRC_PATH}/mockcpp)

if (CMAKE_HOST_SYSTEM_PROCESSOR  STREQUAL "aarch64")
    set(mockcpp_CXXFLAGS "-fPIC")
else()
    set(mockcpp_CXXFLAGS "-fPIC -std=c++11")
endif()
set(mockcpp_FLAGS "-fPIC")
set(mockcpp_LINKER_FLAGS "")

if ((NOT DEFINED ABI_ZERO) OR (ABI_ZERO STREQUAL ""))
    set(ABI_ZERO "true")
endif()


if (ABI_ZERO STREQUAL true)
    set(mockcpp_CXXFLAGS "${mockcpp_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(mockcpp_FLAGS "${mockcpp_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

set(BOOST_INCLUDE_DIRS ${BOOST_SRC_PATH})
if (NOT EXISTS "${CMAKE_INSTALL_PREFIX}/mockcpp/lib/libmockcpp.a")
    set(PATCH_FILE ${MOCKCPP_DOWNLOAD_PATH}/mockcpp-2.7_py3.patch)
    set(REQ_URL "https://gitcode.com/cann-src-third-party/mockcpp/releases/download/v2.7-h1/mockcpp-2.7.tar.gz")
    set(MOCKCPP_LOCAL_SRC "${CANN_3RD_LIB_PATH}/../llt/third_party/mockcpp_src")
    set(MOCKCPP_OPTS
        -DCMAKE_CXX_FLAGS=${mockcpp_CXXFLAGS}
        -DCMAKE_C_FLAGS=${mockcpp_FLAGS}
        -DBOOST_INCLUDE_DIRS=${BOOST_INCLUDE_DIRS}
        -DCMAKE_SHARED_LINKER_FLAGS=${mockcpp_LINKER_FLAGS}
        -DCMAKE_EXE_LINKER_FLAGS=${mockcpp_LINKER_FLAGS}
        -DBUILD_32_BIT_TARGET_BY_64_BIT_COMPILER=OFF
        -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/mockcpp
    )
    include(ExternalProject)
    if(EXISTS ${MOCKCPP_LOCAL_SRC})
        message("Found local mockcpp source: ${MOCKCPP_LOCAL_SRC}")
        file(COPY ${MOCKCPP_LOCAL_SRC}/ DESTINATION "${MOCKCPP_SRC_PATH}/")
        ExternalProject_Add(mockcpp
            SOURCE_DIR ${MOCKCPP_SRC_PATH}
            CONFIGURE_COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR} ${MOCKCPP_OPTS} <SOURCE_DIR>
            BUILD_COMMAND make install -j 16
        )
    else()
        message("No local mockcpp source, downloading from ${REQ_URL}")
        if (NOT EXISTS ${PATCH_FILE})
            set(PATCH_URL "https://gitcode.com/cann-src-third-party/mockcpp/releases/download/v2.7-h1/mockcpp-2.7_py3.patch")
            file(DOWNLOAD
                ${PATCH_URL}
                ${PATCH_FILE}
                TIMEOUT 60
                EXPECTED_HASH SHA256=f1e9091992bf5c340af7d8c2f800b8d43d198fe4a8130f7bcd3f7cba1b0a324b
            )
        endif()
        ExternalProject_Add(mockcpp
            URL ${REQ_URL}
            URL_HASH SHA256=73ab0a8b6d1052361c2cebd85e022c0396f928d2e077bf132790ae3be766f603
            DOWNLOAD_DIR ${MOCKCPP_DOWNLOAD_PATH}
            SOURCE_DIR ${MOCKCPP_SRC_PATH}
            TLS_VERIFY OFF
            PATCH_COMMAND git init && git apply ${PATCH_FILE} && sed -i
            "1icmake_minimum_required(VERSION 3.16.0)" CMakeLists.txt && rm -rf .git
            CONFIGURE_COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR} ${MOCKCPP_OPTS} <SOURCE_DIR>
            BUILD_COMMAND make install -j 16
        )
    endif()
endif()

set(MOCKCPP_DIR ${CMAKE_INSTALL_PREFIX}/mockcpp)

set(MOCKCPP_INCLUDE_ONE ${MOCKCPP_DIR}/include)

set(MOCKCPP_INCLUDE_TWO ${BOOST_INCLUDE_DIRS})

set(MOCKCPP_STATIC_LIBRARY ${MOCKCPP_DIR}/lib/libmockcpp.a)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(mockcpp
    REQUIRED_VARS MOCKCPP_INCLUDE_ONE MOCKCPP_INCLUDE_TWO MOCKCPP_STATIC_LIBRARY
)

message("mockcpp_FOUND is ${mockcpp_FOUND}")

if(mockcpp_FOUND)
    set(MOCKCPP_INCLUDE_DIR ${MOCKCPP_INCLUDE_ONE} ${MOCKCPP_INCLUDE_TWO})
    get_filename_component(MOCKCPP_LIBRARY_DIR ${MOCKCPP_STATIC_LIBRARY} DIRECTORY)

    if(NOT TARGET mockcpp_static)
        add_library(mockcpp_static STATIC IMPORTED)
        set_target_properties(mockcpp_static PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${MOCKCPP_INCLUDE_DIR}"
            IMPORTED_LOCATION "${MOCKCPP_STATIC_LIBRARY}"
            )
        add_dependencies(mockcpp_static mockcpp)
    endif()
endif()