cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(hnswlib_nif C CXX)

if(NOT DEFINED PRIV_DIR)
    if(DEFINED MIX_APP_PATH AND NOT "${MIX_APP_PATH}" STREQUAL "")
        set(PRIV_DIR "${MIX_APP_PATH}/priv")
    else()
        set(PRIV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/priv")
    endif()
endif()
message(STATUS "Using PRIV_DIR: ${PRIV_DIR}")

if(DEFINED ERTS_INCLUDE_DIR AND NOT "${ERTS_INCLUDE_DIR}" STREQUAL "")
    set(ERTS_INCLUDE_DIR "${ERTS_INCLUDE_DIR}")
else()
    set(ERTS_INCLUDE_DIR_ONE_LINER "erl -noshell -s init stop -eval \"io:format('~ts/erts-~ts/include/', [code:root_dir(), erlang:system_info(version)]).\"")
    if(WIN32)
        execute_process(COMMAND powershell -command "${ERTS_INCLUDE_DIR_ONE_LINER}" OUTPUT_VARIABLE ERTS_INCLUDE_DIR)
    else()
        execute_process(COMMAND bash -c "${ERTS_INCLUDE_DIR_ONE_LINER}" OUTPUT_VARIABLE ERTS_INCLUDE_DIR)
    endif()
    set(ERTS_INCLUDE_DIR "${ERTS_INCLUDE_DIR}")
endif()
message(STATUS "Using ERTS_INCLUDE_DIR: ${ERTS_INCLUDE_DIR}")

if(DEFINED EI_INCLUDE_DIR AND NOT "${EI_INCLUDE_DIR}" STREQUAL "")
    set(EI_INCLUDE_DIR "${EI_INCLUDE_DIR}")
else()
    set(EI_DIR_ONE_LINER "erl -noshell -eval 'io:put_chars(code:lib_dir(erl_interface)), halt().'")
    if(WIN32)
        execute_process(COMMAND powershell -command "${EI_DIR_ONE_LINER}" OUTPUT_VARIABLE EI_DIR)
    else()
        execute_process(COMMAND bash -c "${EI_DIR_ONE_LINER}" OUTPUT_VARIABLE EI_DIR)
    endif()

    set(EI_INCLUDE_DIR "${EI_DIR}/include")
    set(EI_LIB_DIR "${EI_DIR}/lib")
endif()
message(STATUS "Using EI_INCLUDE_DIR: ${EI_INCLUDE_DIR}")
message(STATUS "Using EI_LIB_DIR: ${EI_LIB_DIR}")

if(UNIX AND APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "-flat_namespace -undefined suppress -undefined dynamic_lookup")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wno-comment")
if(DEFINED ENV{TARGET_GCC_FLAGS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} $ENV{TARGET_GCC_FLAGS}")
endif()

message(STATUS "CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")

set(SOURCE_FILES
    "${C_SRC}/nif_utils.cpp"
    "${C_SRC}/hnswlib_nif.cpp"
    "${C_SRC}/hnswlib_nif_resource.cpp"
)

include_directories("${ERTS_INCLUDE_DIR}")
include_directories("${EI_INCLUDE_DIR}")
include_directories("${HNSWLIB_SRC}")
link_directories("${EI_LIB_DIR}")
link_libraries(ei)

add_library(hnswlib_nif SHARED
    ${SOURCE_FILES}
)
install(
    TARGETS hnswlib_nif
    RUNTIME DESTINATION "${PRIV_DIR}"
)

set_target_properties(hnswlib_nif PROPERTIES PREFIX "")
set_target_properties(hnswlib_nif PROPERTIES SUFFIX ".so")
set_target_properties(hnswlib_nif PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
)

if(UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
elseif(UNIX AND APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "-flat_namespace -undefined suppress -undefined dynamic_lookup")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g3")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-reorder-ctor -Wno-unused-function -Wno-sign-compare -Wno-unused-parameter -Wno-missing-field-initializers -Wno-deprecated-declarations")
