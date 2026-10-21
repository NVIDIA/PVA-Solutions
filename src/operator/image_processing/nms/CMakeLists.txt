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

if(NOT CUPVA_LLVM_SUPPORTED)
    message(STATUS "LLVM is not supported on this system, OpNms will not be built")
    return()
endif()

find_package(cupva REQUIRED)

add_subdirectory(device)

add_library_cupva(pva_nms_op
    HOST
        OpNms.cpp
    DEVICE
        ${VPUBIN_NAMES})

target_link_libraries(pva_nms_op
    PRIVATE pva_operator_priv_includes
    PRIVATE pva_operator_includes
    PRIVATE pva_operator_utils
    PRIVATE nvcv_types
    PUBLIC pva_operator_priv_submit_definitions
)
