#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.20)

# The project declaration must come after `find_package(CUPVA...)` because `find_package` sets the appropriate `CMAKE_TOOLCHAIN_FILE`.
# Suppress developer warnings due to this requirement.
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS TRUE CACHE INTERNAL "Developer warnings disabled")

# Set the minimum and default versions of the PVA SDK.
set(MINIMUM_PVA_SDK_VERSION "2.5.4")
set(DEFAULT_PVA_SDK_VERSION "2.10.0")

# Define a build option for FORCE_PVA_SDK_VERSION
option(FORCE_PVA_SDK_VERSION "Force a specific version of the PVA SDK" "")

# Define a build option for CUDA_DIR
option(CUDA_TOOLKIT_DIR "Path to CUDA Toolkit root directory" "")

# Define build options for NVSCI paths
option(NVSCI_INCLUDE_PATH "Path to NVSCI include directory" "")
option(NVSCI_LIBRARY_PATH "Path to NVSCI library directory" "")

# Define a build option for IMPORT_DEVICE_TARGETS
option(IMPORT_DEVICE_TARGETS "Import device targets" OFF)

# SDK 2.10 and later supply their own ASIP Programmer; an exported path would override it.
if(FORCE_PVA_SDK_VERSION)
    set(PVA_TARGET_SDK_VERSION "${FORCE_PVA_SDK_VERSION}")
else()
    set(PVA_TARGET_SDK_VERSION "${DEFAULT_PVA_SDK_VERSION}")
endif()
if(PVA_TARGET_SDK_VERSION VERSION_GREATER_EQUAL "2.10.0" AND DEFINED ENV{PVA_ASIP_PATH_FROM_ENVSETUP})
    message(STATUS "PVA SDK ${PVA_TARGET_SDK_VERSION} supplies its own ASIP Programmer; ignoring the ASIP paths from ci/envsetup.sh")
    unset(ENV{PVA_GEN2_ASIP_PATH})
    unset(ENV{PVA_GEN3_ASIP_PATH})
    unset(ENV{CUPVA_GEN2_ASIP_PATH})
    unset(ENV{CUPVA_GEN3_ASIP_PATH})
endif()

# Check if the FORCE_PVA_SDK_VERSION build flag is set and not empty
if(FORCE_PVA_SDK_VERSION)
    if(FORCE_PVA_SDK_VERSION VERSION_LESS MINIMUM_PVA_SDK_VERSION)
        message(WARNING "Forced PVA SDK version ${FORCE_PVA_SDK_VERSION} is less than the minimum required version ${MINIMUM_PVA_SDK_VERSION}. The build may fail due to incompatible features or APIs.")
    endif()
    find_package(CUPVA ${FORCE_PVA_SDK_VERSION} EXACT REQUIRED)
else()
    find_package(CUPVA ${DEFAULT_PVA_SDK_VERSION} REQUIRED)
endif()

message(STATUS "Found CUPVA version ${CUPVA_VERSION}")

if(${CUPVA_VERSION} VERSION_LESS "2.6.0")
    # Turn off LLVM front-end support flag for older PVA SDK versions
    set(CUPVA_LLVM_SUPPORTED OFF)
    add_compile_definitions(CUPVA_LLVM_SUPPORTED=0)
    # Only build for GEN2
    set(CUPVA_SUPPORTED_GENS "GEN2")
    set(CUPVA_ONLY_BUILD_GENS "GEN2")
else()
    set(CUPVA_LLVM_SUPPORTED ON)
    add_compile_definitions(CUPVA_LLVM_SUPPORTED=1)
endif()

# Set cuPVA runtime version
string(REGEX MATCH "^([0-9]+)\\.([0-9]+)" VERSION_MATCH ${CUPVA_VERSION})
math(EXPR CUPVA_RUNTIME_VERSION "(${CMAKE_MATCH_1} * 1000) + (${CMAKE_MATCH_2} * 1)")
add_compile_definitions(CUPVA_RUNTIME_VERSION=${CUPVA_RUNTIME_VERSION})

# Check for CUDA stream support (available in CUPVA 2.7.0 and later)
if(${CUPVA_VERSION} VERSION_GREATER_EQUAL "2.7.0" AND (CUPVA_BUILD_MODE STREQUAL QNX OR CUPVA_BUILD_MODE STREQUAL L4T))
    set(CUPVA_HAS_CUDA_STREAM_SUPPORT 1)
else()
    set(CUPVA_HAS_CUDA_STREAM_SUPPORT 0)
endif()
add_compile_definitions(CUPVA_HAS_CUDA_STREAM_SUPPORT=${CUPVA_HAS_CUDA_STREAM_SUPPORT})

# Set CMAKE_PREFIX_PATH and CMAKE_MODULE_PATH to ensure lower-level CMake files use the same versions
list(APPEND CMAKE_PREFIX_PATH ${CUPVA_DIR})
list(APPEND CMAKE_MODULE_PATH ${CUPVA_DIR})

# Enable ccache if available (must be before project() for compiler detection)
include("${CMAKE_CURRENT_LIST_DIR}/cmake/ccache_setup.cmake")

project(pva-solutions VERSION 0.6.0)

# If project toolchain leaves CMAKE_BUILD_TYPE unset, choose default now
if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PVA_SOLUTIONS_CMAKE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${PVA_SOLUTIONS_CMAKE_PATH}")

include("${PVA_SOLUTIONS_CMAKE_PATH}/pva_vpu_compile_flags.cmake")

# Add the linker flag to ignore unresolved NvSci symbols in shared libraries
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-unresolved-symbols=ignore-in-shared-libs")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-unresolved-symbols=ignore-in-shared-libs")

# Look in both host root paths and system root paths for the below packages...
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CUPVA_BUILD_MODE STREQUAL QNX)
    set(ENABLE_QNX ON)
endif()

# Setup CUDA Toolkit
include("cmake/cuda_setup.cmake")
setup_cuda_toolkit()

# Single-op build filter (defines pva_op_add_subdirectory macro used by operator subdirs)
include("${CMAKE_CURRENT_LIST_DIR}/cmake/pva_op_filter.cmake")

# Set the NVCV_ENABLE_INSTALL option to OFF
set(NVCV_ENABLE_INSTALL OFF CACHE BOOL "Enables creation of NVCV installers using cpack" FORCE)

# Check if cvcuda directory exists, if not clone it
set(CVCUDA_VERSION_TAG "v0.14.0-beta")
set(CVCUDA_COMMIT_SHA "d271c1c8788bf697431098b489060e85cf539943")
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cvcuda")
    message(STATUS "CV-CUDA not found, cloning repository with tag ${CVCUDA_VERSION_TAG} and commit ${CVCUDA_COMMIT_SHA}...")
    execute_process(
        COMMAND git clone https://github.com/CVCUDA/CV-CUDA.git cvcuda
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty"
        RESULT_VARIABLE GIT_CLONE_RESULT
    )
    if(NOT GIT_CLONE_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to clone CV-CUDA repository")
    endif()
    execute_process(
        COMMAND git -C cvcuda rev-list -n 1 ${CVCUDA_VERSION_TAG}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty"
        OUTPUT_VARIABLE CVCUDA_TAG_SHA
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE GIT_TAG_LOOKUP_RESULT
    )
    if(NOT GIT_TAG_LOOKUP_RESULT EQUAL 0 OR NOT CVCUDA_TAG_SHA STREQUAL CVCUDA_COMMIT_SHA)
        message(FATAL_ERROR "CV-CUDA tag verification failed: tag ${CVCUDA_VERSION_TAG} expected ${CVCUDA_COMMIT_SHA}, got ${CVCUDA_TAG_SHA}")
    endif()
    execute_process(
        COMMAND git -C cvcuda checkout --detach ${CVCUDA_COMMIT_SHA}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty"
        RESULT_VARIABLE GIT_CHECKOUT_RESULT
    )
    if(NOT GIT_CHECKOUT_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to check out CV-CUDA commit ${CVCUDA_COMMIT_SHA}")
    endif()
    message(STATUS "CV-CUDA repository cloned successfully")
endif()

file(REAL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cvcuda" CVCUDA_SOURCE_DIR)
execute_process(
    COMMAND git -c safe.directory=${CVCUDA_SOURCE_DIR} -C "${CVCUDA_SOURCE_DIR}" rev-parse --show-toplevel
    OUTPUT_VARIABLE CVCUDA_GIT_TOPLEVEL
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE GIT_TOPLEVEL_RESULT
)
file(REAL_PATH "${CVCUDA_GIT_TOPLEVEL}" CVCUDA_GIT_TOPLEVEL)
if(NOT GIT_TOPLEVEL_RESULT EQUAL 0 OR NOT CVCUDA_GIT_TOPLEVEL STREQUAL CVCUDA_SOURCE_DIR)
    message(FATAL_ERROR "CV-CUDA checkout verification failed: ${CVCUDA_SOURCE_DIR} is not a git checkout")
endif()
execute_process(
    COMMAND git -c safe.directory=${CVCUDA_SOURCE_DIR} -C "${CVCUDA_SOURCE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE CVCUDA_CHECKED_OUT_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE GIT_REV_PARSE_RESULT
)
if(NOT GIT_REV_PARSE_RESULT EQUAL 0 OR NOT CVCUDA_CHECKED_OUT_SHA STREQUAL CVCUDA_COMMIT_SHA)
    message(FATAL_ERROR "CV-CUDA checkout verification failed: expected ${CVCUDA_COMMIT_SHA}, got ${CVCUDA_CHECKED_OUT_SHA}")
endif()

# For L4T cross-compilation, use the C++ cross-compiler as the CUDA host compiler.
# This must be set before the CUDA language is enabled.
if(CUPVA_BUILD_MODE STREQUAL "L4T")
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "CUDA host compiler for L4T cross-compilation")
endif()

# Build only NVCV; PVA Solutions does not depend on CV-CUDA operators.
add_subdirectory(3rdparty/cvcuda/src/nvcv)

# Build lodePNG library
add_subdirectory(3rdparty/lodepng)

# nlohmann json lib
add_subdirectory(3rdparty/nlohmann_json)

# Any runtime packages we generate should have this prefix
string(TOLOWER "${CUPVA_BUILD_MODE}" PVASOL_RUNTIME_PACKAGE)

# For QNX target, we will generate 'cross' runtime package
if(CUPVA_BUILD_MODE STREQUAL QNX)
    set(PVASOL_RUNTIME_PACKAGE "${PVASOL_RUNTIME_PACKAGE}-cross")
endif()

set(PVASOL_SAMPLES_PACKAGE "${PVASOL_RUNTIME_PACKAGE}-samples")

# set install paths
set(PVASOL_LIB_INSTALL_PATH "lib/${CMAKE_LIBRARY_ARCHITECTURE}/")
set(PVASOL_BIN_INSTALL_PATH "bin/${CMAKE_LIBRARY_ARCHITECTURE}/")
set(PVASOL_INCLUDE_INSTALL_PATH "include")

# -fPIC for all the shared library
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include("cmake/pva_onnx_compiler.cmake")
include("cmake/pva_operator_utils.cmake")

# Set the device binaries root directory (shared by all operators)
set(DEVICE_BINARIES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src/operator/device_binaries")

# Fetch precompiled device binaries / pva_dl_lib / host tools
# for the pinned release. Must run before add_subdirectory(src)
# so the fetched files are in place before the operators are configured.
include("cmake/pva_fetch_prebuilt.cmake")

add_subdirectory(src)
# When filtering to a single op, exclude all samples/pipelines from the default 'all' target.
# The specific op's sample can still be built explicitly via --target <sample_name>.
if(PVA_BUILD_OP_FILTER)
    add_subdirectory(samples EXCLUDE_FROM_ALL)
    add_subdirectory(pipelines EXCLUDE_FROM_ALL)
else()
    add_subdirectory(samples)
    add_subdirectory(pipelines)
endif()

# --------------------------------------------------------------
# PACKAGING
# --------------------------------------------------------------
include("cmake/packaging.cmake")

# --------------------------------------------------------------
# Compilation options printout
# --------------------------------------------------------------
message(STATUS "")
message(STATUS "Platform")
message(STATUS "    Host             : ${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_VERSION} ${CMAKE_HOST_SYSTEM_PROCESSOR}")
message(STATUS "    Target           : ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "    CMake            : ${CMAKE_VERSION}")
message(STATUS "    CMake generator  : ${CMAKE_GENERATOR}")
message(STATUS "    CMake build tool : ${CMAKE_BUILD_TOOL}")
message(STATUS "    Configuration    : ${CMAKE_BUILD_TYPE}")
message(STATUS "    ccache           : ${CCACHE_CMD}")
message(STATUS "")

message(STATUS "Default compiler/linker config")
message(STATUS "    C++ Compiler : ${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_VERSION})")
message(STATUS "    C++ Standard : ${CMAKE_CXX_STANDARD}")

string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE)
if(BUILD_TYPE)
    message(STATUS "    C++ flags    : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${BUILD_TYPE}}")
else()
    message(STATUS "    C++ flags    : ${CMAKE_CXX_FLAGS}")
endif()

message(STATUS "")

message(STATUS "    C Compiler : ${CMAKE_C_COMPILER} (${CMAKE_C_COMPILER_VERSION})")
if(BUILD_TYPE)
    message(STATUS "    C flags    : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${BUILD_TYPE}}")
else()
    message(STATUS "    C flags    : ${CMAKE_C_FLAGS}")
endif()

message(STATUS "")

message(STATUS "    CUDA Compiler : ${CMAKE_CUDA_COMPILER} (${CMAKE_CUDA_COMPILER_VERSION})")
if(BUILD_TYPE)
    message(STATUS "    CUDA flags    : ${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${BUILD_TYPE}}")
else()
    message(STATUS "    CUDA flags    : ${CMAKE_CUDA_FLAGS}")
endif()

message(STATUS "")

message(STATUS "    CUDA toolkit target dir : ${CUDAToolkit_TARGET_DIR}")
message(STATUS "    CUDA toolkit include dir : ${CUDAToolkit_INCLUDE_DIR}")
message(STATUS "    CUDA toolkit path to runtime dynamic lib : ${CUDA_cudart_LIBRARY}")
message(STATUS "    CUDA toolkit path to runtime static lib : ${CUDA_cudart_static_LIBRARY}")

message(STATUS "")

# Print operator runtime compile flags/defs and link flags, if they exist
get_target_property(PVA_OPERATOR_C_OPT pva_operator COMPILE_OPTIONS)
if (PVA_OPERATOR_C_OPT)
    message(STATUS "    pva_operator Compiler Options : ${PVA_OPERATOR_C_OPT}")
endif()
get_target_property(_PVA_OPERATOR_DEFS pva_operator COMPILE_DEFINITIONS)
if (_PVA_OPERATOR_DEFS)
    set(PVA_OPERATOR_DEFS "")
    foreach(def ${_PVA_OPERATOR_DEFS})
        set(PVA_OPERATOR_DEFS "${PVA_OPERATOR_DEFS} -D${def}")
    endforeach()
    message(STATUS "    pva_operator Definitions      :${PVA_OPERATOR_DEFS}")
endif()
get_target_property(PVA_OPERATOR_L_OPT pva_operator LINK_OPTIONS)
if (PVA_OPERATOR_L_OPT)
    message(STATUS "    pva_operator Linker Options   : ${PVA_OPERATOR_L_OPT}")
endif()

message(STATUS "")

message(STATUS "CPACK_PACKAGE_VERSION: ${CPACK_PACKAGE_VERSION}")

