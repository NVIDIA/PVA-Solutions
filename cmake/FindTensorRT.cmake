#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#
# FindTensorRT.cmake
#
# Finds TensorRT libraries for cross-compilation (L4T) and native builds.
#
# This module defines:
#   TensorRT_FOUND        - True if TensorRT was found
#   TensorRT_INCLUDE_DIRS - TensorRT include directories
#   TensorRT_LIBRARIES    - TensorRT libraries to link against
#   TensorRT_LIBRARY_DIRS - TensorRT library directories (for link_directories)
#   TensorRT_VERSION      - TensorRT version string
#
# and the imported targets TensorRT::nvinfer, TensorRT::nvinfer_plugin and
# TensorRT::nvonnxparser. Set TENSORRT_ROOT to search a specific installation.
#

# Use the packaged installation by default. Callers may set TENSORRT_ROOT to a
# different installed TensorRT location explicitly.
if(NOT DEFINED TENSORRT_ROOT OR TENSORRT_ROOT STREQUAL "")
    if(EXISTS "/usr/include/aarch64-linux-gnu/NvInfer.h")
        set(TENSORRT_ROOT "/usr")
    endif()
endif()
set(TENSORRT_ROOT "${TENSORRT_ROOT}" CACHE PATH "TensorRT root directory")

if(NOT TensorRT_FIND_QUIETLY)
    message(STATUS "TensorRT: TENSORRT_ROOT = ${TENSORRT_ROOT}")
endif()

# Locate the headers
unset(TensorRT_INCLUDE_DIR)
if(EXISTS "${TENSORRT_ROOT}/include/aarch64-linux-gnu/NvInfer.h")
    set(TensorRT_INCLUDE_DIR "${TENSORRT_ROOT}/include/aarch64-linux-gnu")
elseif(EXISTS "${TENSORRT_ROOT}/include/NvInfer.h")
    set(TensorRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
endif()

# Determine which library directory to use
# For cross-compilation (L4T), use stub libraries
if(EXISTS "${TENSORRT_ROOT}/lib/stubs/libnvinfer.so" AND
   (CMAKE_CROSSCOMPILING OR CUPVA_BUILD_MODE STREQUAL "L4T"))
    set(TensorRT_LIBRARY_DIR "${TENSORRT_ROOT}/lib/stubs")
elseif(EXISTS "${TENSORRT_ROOT}/lib/aarch64-linux-gnu/libnvinfer.so")
    set(TensorRT_LIBRARY_DIR "${TENSORRT_ROOT}/lib/aarch64-linux-gnu")
else()
    set(TensorRT_LIBRARY_DIR "${TENSORRT_ROOT}/lib")
endif()

# Locate the libraries; a missing library leaves its variable empty
foreach(_trt_lib nvinfer nvinfer_plugin nvonnxparser)
    unset(TensorRT_${_trt_lib}_LIBRARY)
    if(EXISTS "${TensorRT_LIBRARY_DIR}/lib${_trt_lib}.so")
        set(TensorRT_${_trt_lib}_LIBRARY "${TensorRT_LIBRARY_DIR}/lib${_trt_lib}.so")
    endif()
endforeach()
unset(_trt_lib)

# Get version from header
unset(TensorRT_VERSION)
if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TRT_VERSION_STRINGS
        REGEX "#define NV_TENSORRT_(MAJOR|MINOR|PATCH)")

    foreach(_ver_str ${TRT_VERSION_STRINGS})
        if(_ver_str MATCHES "#define NV_TENSORRT_MAJOR[ ]+([0-9]+)")
            set(TensorRT_VERSION_MAJOR ${CMAKE_MATCH_1})
        elseif(_ver_str MATCHES "#define NV_TENSORRT_MINOR[ ]+([0-9]+)")
            set(TensorRT_VERSION_MINOR ${CMAKE_MATCH_1})
        elseif(_ver_str MATCHES "#define NV_TENSORRT_PATCH[ ]+([0-9]+)")
            set(TensorRT_VERSION_PATCH ${CMAKE_MATCH_1})
        endif()
    endforeach()

    set(TensorRT_VERSION "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
endif()

# Set TensorRT_FOUND and honour REQUIRED, QUIET and version requests
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS
        TensorRT_INCLUDE_DIR
        TensorRT_nvinfer_LIBRARY
        TensorRT_nvinfer_plugin_LIBRARY
        TensorRT_nvonnxparser_LIBRARY
    VERSION_VAR TensorRT_VERSION
)

if(NOT TensorRT_FOUND)
    return()
endif()

# Set output variables
set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})
set(TensorRT_LIBRARY_DIRS ${TensorRT_LIBRARY_DIR})
set(TensorRT_LIBRARIES
    ${TensorRT_nvinfer_LIBRARY}
    ${TensorRT_nvinfer_plugin_LIBRARY}
    ${TensorRT_nvonnxparser_LIBRARY}
)

if(NOT TensorRT_FIND_QUIETLY)
    message(STATUS "TensorRT: Include: ${TensorRT_INCLUDE_DIRS}")
    message(STATUS "TensorRT: Lib dir: ${TensorRT_LIBRARY_DIRS}")
endif()

# Create imported targets
if(NOT TARGET TensorRT::nvinfer)
    add_library(TensorRT::nvinfer SHARED IMPORTED)
    set_target_properties(TensorRT::nvinfer PROPERTIES
        IMPORTED_LOCATION "${TensorRT_nvinfer_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
    )

    add_library(TensorRT::nvinfer_plugin SHARED IMPORTED)
    set_target_properties(TensorRT::nvinfer_plugin PROPERTIES
        IMPORTED_LOCATION "${TensorRT_nvinfer_plugin_LIBRARY}"
    )

    add_library(TensorRT::nvonnxparser SHARED IMPORTED)
    set_target_properties(TensorRT::nvonnxparser PROPERTIES
        IMPORTED_LOCATION "${TensorRT_nvonnxparser_LIBRARY}"
    )
endif()
