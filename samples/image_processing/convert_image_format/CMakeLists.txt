#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(convert_image_format)

add_subdirectory(ref)

add_executable(convert_image_format
    convert_image_format.c
)

target_link_libraries(convert_image_format
    PRIVATE cupva_host
    PRIVATE nvcv_types
    PRIVATE pva_operator
    PRIVATE convert_image_format_ref
    PRIVATE CUDA::cudart_static
)

install(TARGETS convert_image_format
        RUNTIME DESTINATION ${PVASOL_BIN_INSTALL_PATH}
        COMPONENT ${PVASOL_SAMPLES_PACKAGE})

if(CUPVA_BUILD_MODE STREQUAL "L4T" OR CUPVA_BUILD_MODE STREQUAL "QNX")
    add_executable(convert_image_format_cuda_interop
        convert_image_format_cuda_interop.c
    )

    target_link_libraries(convert_image_format_cuda_interop
        PRIVATE cupva_host
        PRIVATE nvcv_types
        PRIVATE pva_operator
        PRIVATE convert_image_format_ref
        PRIVATE interop_utils
        PRIVATE CUDA::cudart_static
    )

    install(TARGETS convert_image_format_cuda_interop
            RUNTIME DESTINATION ${PVASOL_BIN_INSTALL_PATH}
            COMPONENT ${PVASOL_SAMPLES_PACKAGE})

    if(NVSCI_SUPPORTED)
        add_executable(convert_image_format_nvsci_raw_interop
            convert_image_format_nvsci_raw_interop.c
        )

        target_link_libraries(convert_image_format_nvsci_raw_interop
            PRIVATE cupva_host
            PRIVATE nvcv_types
            PRIVATE pva_operator
            PRIVATE convert_image_format_ref
            PRIVATE interop_utils
            PRIVATE nvscibuf
            PRIVATE nvscisync
            PRIVATE CUDA::cudart_static
        )

        install(TARGETS convert_image_format_nvsci_raw_interop
                RUNTIME DESTINATION ${PVASOL_BIN_INSTALL_PATH}
                COMPONENT ${PVASOL_SAMPLES_PACKAGE})
    endif()
endif()