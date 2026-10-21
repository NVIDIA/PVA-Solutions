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

add_library_cupva(pva_bartlett_op
    HOST
        OpDoaBartlettBeamforming.cpp
    DEVICE
        pva_bartlett_separable_dev
        pva_bartlett_full2d_dev)

target_link_libraries(pva_bartlett_op
    PRIVATE pva_operator_priv_includes
    PRIVATE pva_operator_includes
    PRIVATE pva_operator_utils
    PRIVATE pva_steering_vector_generator
    PRIVATE nvcv_types
    PUBLIC pva_operator_priv_submit_definitions
)

add_library(pva_bartlett_op_includes INTERFACE)
target_include_directories(pva_bartlett_op_includes
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/device
)

target_include_directories(
    pva_bartlett_op
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/device)
