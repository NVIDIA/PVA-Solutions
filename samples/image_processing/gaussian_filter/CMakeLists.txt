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

project(gaussian_filter)

add_subdirectory(ref)

add_executable(gaussian_filter
    gaussian_filter.c
)

target_link_libraries(gaussian_filter
    PRIVATE cupva_host
    PRIVATE nvcv_types
    PRIVATE pva_operator
    PRIVATE gaussian_filter_ref
    PRIVATE imageIO_utils
    PRIVATE CUDA::cudart_static
)

install(TARGETS gaussian_filter
        RUNTIME DESTINATION ${PVASOL_BIN_INSTALL_PATH}
        COMPONENT ${PVASOL_SAMPLES_PACKAGE})

if(CUPVA_BUILD_MODE STREQUAL "L4T" OR CUPVA_BUILD_MODE STREQUAL "QNX")
    add_executable(gaussian_filter_cuda_interop
        gaussian_filter_cuda_interop.c
    )

    target_link_libraries(gaussian_filter_cuda_interop
        PRIVATE cupva_host
        PRIVATE nvcv_types
        PRIVATE pva_operator
        PRIVATE gaussian_filter_ref
        PRIVATE imageIO_utils
        PRIVATE interop_utils
        PRIVATE CUDA::cudart_static
    )

    install(TARGETS gaussian_filter_cuda_interop
            RUNTIME DESTINATION ${PVASOL_BIN_INSTALL_PATH}
            COMPONENT ${PVASOL_SAMPLES_PACKAGE})

    message(STATUS "NVSCI_SUPPORTED: ${NVSCI_SUPPORTED}")
    if(NVSCI_SUPPORTED)
        add_executable(gaussian_filter_nvsci_raw_interop
            gaussian_filter_nvsci_raw_interop.c
        )

        target_link_libraries(gaussian_filter_nvsci_raw_interop
            PRIVATE cupva_host
            PRIVATE nvcv_types
            PRIVATE pva_operator
            PRIVATE gaussian_filter_ref
            PRIVATE imageIO_utils
            PRIVATE interop_utils
            PRIVATE nvscibuf
            PRIVATE nvscisync
            PRIVATE CUDA::cudart_static
        )

        install(TARGETS gaussian_filter_nvsci_raw_interop
                RUNTIME DESTINATION ${PVASOL_BIN_INSTALL_PATH}
                COMPONENT ${PVASOL_SAMPLES_PACKAGE})

        add_executable(gaussian_filter_nvsci_tensor_interop
            gaussian_filter_nvsci_tensor_interop.c
        )

        target_link_libraries(gaussian_filter_nvsci_tensor_interop
            PRIVATE cupva_host
            PRIVATE nvcv_types
            PRIVATE pva_operator
            PRIVATE gaussian_filter_ref
            PRIVATE imageIO_utils
            PRIVATE interop_utils
            PRIVATE nvscibuf
            PRIVATE nvscisync
            PRIVATE CUDA::cudart_static
        )

        install(TARGETS gaussian_filter_nvsci_tensor_interop
                RUNTIME DESTINATION ${PVASOL_BIN_INSTALL_PATH}
                COMPONENT ${PVASOL_SAMPLES_PACKAGE})
    endif()
endif()