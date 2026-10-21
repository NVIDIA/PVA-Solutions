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

cmake_minimum_required(VERSION 3.20)
find_package(cupva REQUIRED)
find_package(nvcv_types REQUIRED)
find_package(pva_solutions REQUIRED)

project(nhwc_to_nchw_letterbox_normalize)

add_subdirectory(ref)

add_executable(nhwc_to_nchw_letterbox_normalize
    nhwc_to_nchw_letterbox_normalize.cpp
)

target_link_libraries(nhwc_to_nchw_letterbox_normalize
    PRIVATE cupva_host
    PRIVATE nvcv_types
    PRIVATE pva_operator
    PRIVATE pva_nhwc_to_nchw_letterbox_normalize_op
    PRIVATE nhwc_to_nchw_letterbox_normalize_ref
    PRIVATE CUDA::cudart_static
)

install(TARGETS nhwc_to_nchw_letterbox_normalize
        RUNTIME DESTINATION ${PVASOL_BIN_INSTALL_PATH}
        COMPONENT ${PVASOL_SAMPLES_PACKAGE})
