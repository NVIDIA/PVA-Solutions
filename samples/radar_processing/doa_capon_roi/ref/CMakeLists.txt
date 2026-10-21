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

# Host utilities and scalar reference implementation for Capon ROI DoA.
add_library(doa_capon_roi_ref STATIC
    doa_capon_roi_ref.cpp
    doa_capon_roi_utils.c
    part1/cov_mat_ref.c
    part1/complex_cholesky_ref.c
    part2/find_local_max_ref.cpp
)

target_include_directories(doa_capon_roi_ref
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/part1
        ${CMAKE_CURRENT_SOURCE_DIR}/part2
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(doa_capon_roi_ref
    PRIVATE nvcv_types
    PRIVATE cupva_host
    PUBLIC  pva_operator
    PUBLIC  pva_operator_includes
    PRIVATE pva_operator_priv_includes
    PRIVATE pva_operator_utils
)
