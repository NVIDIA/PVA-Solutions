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
    message(STATUS "LLVM is not supported on this system, OpDoaCaponROI will not be built")
    return()
endif()

find_package(cupva REQUIRED)

# Build the combined two-pass operator.
add_subdirectory(device)

add_library_cupva(pva_capon_roi_op
    HOST
        OpDoaCaponROI.cpp
    DEVICE
        pva_capon_roi_combined_s32_interp_dev
        pva_capon_roi_combined_s32_no_interp_dev
        pva_capon_roi_combined_fp_interp_dev
        pva_capon_roi_combined_fp_no_interp_dev)

target_link_libraries(pva_capon_roi_op
    PRIVATE pva_operator_priv_includes
    PRIVATE pva_operator_includes
    PRIVATE pva_operator_utils
    PRIVATE nvcv_types
    PUBLIC pva_operator_priv_submit_definitions
)

target_include_directories(
    pva_capon_roi_op
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/device)
