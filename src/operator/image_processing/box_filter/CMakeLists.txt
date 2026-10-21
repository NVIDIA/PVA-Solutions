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

if(NOT CUPVA_LLVM_SUPPORTED)
    message(STATUS "LLVM is not supported on this system, FAST Corner Detector will not be built")
    return()
endif()

find_package(cupva REQUIRED)

add_library_cupva(pva_box_filter_op
    HOST
        OpBoxFilter.cpp
    DEVICE
        ${VPUBIN_NAMES})

target_compile_options(pva_box_filter_op
    PRIVATE -DTW=128
    PRIVATE -DTH=128
    PRIVATE -DQBITS_8=8
    PRIVATE -DQBITS_16=16
)

target_link_libraries(pva_box_filter_op
    PRIVATE pva_operator_priv_includes
    PRIVATE pva_operator_includes
    PRIVATE pva_operator_utils
    PRIVATE pva_primitive_utils
    PRIVATE nvcv_types
    PUBLIC pva_operator_priv_submit_definitions
)
