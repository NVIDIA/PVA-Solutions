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

project(radar_doa_capon_roi)

add_subdirectory(ref)

# Run the operator on capon_roi/16x16_ura and compare it with the fixed-point C reference.
add_executable(radar_doa_capon_roi
    doa_capon_roi.c
)

target_link_libraries(radar_doa_capon_roi
    PRIVATE cupva_host
    PRIVATE CUDA::cudart_static
    PRIVATE nvcv_types
    PRIVATE pva_operator
    PRIVATE doa_capon_roi_ref
    PRIVATE imageIO_utils
)

install(TARGETS radar_doa_capon_roi
        RUNTIME DESTINATION ${PVASOL_BIN_INSTALL_PATH}
        COMPONENT ${PVASOL_SAMPLES_PACKAGE})
