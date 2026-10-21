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

# To guide the search if libraries/headers aren't in default system paths,
# ensure the CMake variables NVSCI_INCLUDE_PATH and NVSCI_LIBRARY_PATH are set.
# These can be set as cache variables (e.g., via -D options to CMake)
# or by a parent CMake script.
#
# CMake Variables (expected to be set):
#   NVSCI_INCLUDE_PATH - Path to the directory containing NvSci header files (e.g., nvscibuf.h).
#   NVSCI_LIBRARY_PATH - Path to the directory containing NvSci libraries (libnvscibuf.so, libnvscisync.so).

find_path(NVSCI_H "nvscibuf.h"
    HINTS
        ${NVSCI_INCLUDE_PATH})
find_library(NVSCIBUF_LIB "nvscibuf"
    HINTS
        ${NVSCI_LIBRARY_PATH}
    NO_CMAKE_FIND_ROOT_PATH)
find_library(NVSCISYNC_LIB "nvscisync"
    HINTS
        ${NVSCI_LIBRARY_PATH}
    NO_CMAKE_FIND_ROOT_PATH)

if (NOT NVSCI_H OR
    NOT NVSCIBUF_LIB OR
    NOT NVSCISYNC_LIB)
    if(NOT nvsci_FIND_QUIETLY)
        message(WARNING "One or more NvSci libraries/headers not found. Please ensure the CMake variables/options NVSCI_INCLUDE_PATH and NVSCI_LIBRARY_PATH are set correctly. For example:\n"
                        "  -DNVSCI_INCLUDE_PATH=/path/to/nvsci/include \n"
                        "  -DNVSCI_LIBRARY_PATH=/path/to/nvsci/lib")
    endif()
    set(nvsci_FOUND OFF)
    return()
endif()

# define targets using external libraries
add_library(nvscibuf SHARED IMPORTED)
set_target_properties(nvscibuf PROPERTIES IMPORTED_LOCATION ${NVSCIBUF_LIB})
target_include_directories(nvscibuf INTERFACE ${NVSCI_H})

add_library(nvscisync SHARED IMPORTED)
set_target_properties(nvscisync PROPERTIES IMPORTED_LOCATION ${NVSCISYNC_LIB})
target_include_directories(nvscisync INTERFACE ${NVSCI_H})

set(nvsci_FOUND ON)
