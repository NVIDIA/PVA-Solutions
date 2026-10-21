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

add_subdirectory(device)

set(MEDIAN_FILTER_DEVICE_TARGETS)
foreach(io_type u8 s8 u16 s16 u32 s32)
    list(APPEND MEDIAN_FILTER_DEVICE_TARGETS
         pva_median_filter_${io_type}_3x3_dev)

    foreach(knl_type 5x5 5x7 7x7)
        list(APPEND MEDIAN_FILTER_DEVICE_TARGETS
             pva_median_filter_sepa_2x2_${io_type}_${knl_type}_dev)
    endforeach()
endforeach()

list(APPEND MEDIAN_FILTER_DEVICE_TARGETS
     pva_median_filter_i2s16_3x3_dev
     pva_median_filter_sepa_2x2_i2s16_5x5_dev
     pva_median_filter_sepa_2x2_i2s16_5x7_dev
     pva_median_filter_sepa_2x2_i2s16_7x7_dev)

add_library_cupva(pva_median_filter_op
    HOST
        OpMedianFilter.cpp
    DEVICE
        ${MEDIAN_FILTER_DEVICE_TARGETS})

target_link_libraries(pva_median_filter_op
    PRIVATE pva_operator_priv_includes
    PRIVATE pva_operator_includes
    PRIVATE pva_operator_utils
    PRIVATE nvcv_types
    PUBLIC pva_operator_priv_submit_definitions
)
