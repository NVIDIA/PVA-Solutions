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

project(radar_range_fft)

add_subdirectory(ref)

add_executable(radar_range_fft
    range_fft.c
)

target_link_libraries(radar_range_fft
    PRIVATE cupva_host
    PRIVATE nvcv_types
    PRIVATE pva_operator
    PRIVATE range_fft_ref
    PRIVATE imageIO_utils
    PRIVATE radar_ref_utils
    PRIVATE CUDA::cudart_static
)

install(TARGETS radar_range_fft
        RUNTIME DESTINATION ${PVASOL_BIN_INSTALL_PATH}
        COMPONENT ${PVASOL_SAMPLES_PACKAGE})

if(CUPVA_BUILD_MODE STREQUAL "L4T" OR CUPVA_BUILD_MODE STREQUAL "QNX")
    message(STATUS "NVSCI_SUPPORTED: ${NVSCI_SUPPORTED}")
    if(NVSCI_SUPPORTED)
        add_executable(radar_range_fft_nvsci_raw_interop
            range_fft_nvsci_raw_interop.c
        )

        target_link_libraries(radar_range_fft_nvsci_raw_interop
            PRIVATE cupva_host
            PRIVATE nvcv_types
            PRIVATE pva_operator
            PRIVATE range_fft_ref
            PRIVATE imageIO_utils
            PRIVATE radar_ref_utils
            PRIVATE interop_utils
            PRIVATE nvscibuf
            PRIVATE nvscisync
            PRIVATE CUDA::cudart_static
        )

        install(TARGETS radar_range_fft_nvsci_raw_interop
                RUNTIME DESTINATION ${PVASOL_BIN_INSTALL_PATH}
                COMPONENT ${PVASOL_SAMPLES_PACKAGE})
    endif()
endif()
