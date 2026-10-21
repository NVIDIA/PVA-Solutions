#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

find_package(cupva REQUIRED)

if(NOT TARGET pva_dl_lib)
    set(PVA_DL_LIB_NAME pva_dl_${CUPVA_RUNTIME_VERSION})
    set(_pva_dl_arch "${CMAKE_LIBRARY_ARCHITECTURE}")
    if(CUPVA_BUILD_MODE STREQUAL "EMU")
        string(APPEND _pva_dl_arch "-emu")
    endif()
    set(PVA_DL_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/${_pva_dl_arch}/lib${PVA_DL_LIB_NAME}.a)
    set(PVA_DL_ALLOWLIST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/lib/${_pva_dl_arch}/pva_allowlist_pva_dl_lib_all_${CUPVA_RUNTIME_VERSION})

    add_library(pva_dl_lib STATIC IMPORTED GLOBAL)
    set_target_properties(pva_dl_lib PROPERTIES
        IMPORTED_LOCATION ${PVA_DL_LIB_PATH}
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ALLOWLIST_FILE "${PVA_DL_ALLOWLIST_FILE}"
    )
endif()