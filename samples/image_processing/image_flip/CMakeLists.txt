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

cmake_minimum_required(VERSION 3.20)
find_package(cupva REQUIRED)
find_package(nvcv_types REQUIRED)
find_package(pva_solutions REQUIRED)

project(image_flip)

add_subdirectory(ref)

add_executable(image_flip
    image_flip.c
)

target_link_libraries(image_flip
    PRIVATE cupva_host
    PRIVATE nvcv_types
    PRIVATE pva_operator
    PRIVATE image_flip_ref
    PRIVATE imageIO_utils
    PRIVATE CUDA::cudart_static
)

install(TARGETS image_flip
        RUNTIME DESTINATION ${PVASOL_BIN_INSTALL_PATH}
        COMPONENT ${PVASOL_SAMPLES_PACKAGE})
