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
project(radar_pipeline_4t4r VERSION 1.0.0)

# ============================================================================
# Radar Pipeline 4T4R Shared Library (.so) for e2e-solution
# ============================================================================
add_library(radar_pipeline_4t4r SHARED
    radar_pipeline_context.cpp
    radar_pipeline_tensors.cpp
    radar_pipeline_pva.cpp
    radar_pipeline_reference.cpp
    radar_pipeline_factory.cpp
)

# Hide all symbols by default; only radar_pipeline_4t4r() is exported.
# This prevents symbol conflicts when both 8t8r and 4t4r are linked to the same binary.
set_target_properties(radar_pipeline_4t4r PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Link radar pipeline 4T4R shared library
target_link_libraries(radar_pipeline_4t4r
    PRIVATE cupva_host
    PRIVATE CUDA::cudart_static
    PRIVATE nvcv_types
    PRIVATE pva_operator
    PRIVATE pva_operator_utils
    PRIVATE pva_range_fft_op
    PRIVATE pva_doppler_fft_op
    PRIVATE pva_nci_op
    PRIVATE pva_peak_detection_op
    PRIVATE pva_doa_da_fft_op
    PRIVATE imageIO_utils
    PRIVATE doa_target_processing_ref
    PRIVATE peak_detection_ref
    PRIVATE nci_ref
    PRIVATE doppler_fft_ref
    PRIVATE range_fft_ref
    PRIVATE radar_ref_utils
)

# Include directories for radar pipeline 4T4R
# public/pipelines/radar/include is 3 levels up from radar_advanced/lib/4t4r/
target_include_directories(radar_pipeline_4t4r
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../radar/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../>
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

# Install shared library
install(TARGETS radar_pipeline_4t4r
    LIBRARY DESTINATION ${PVASOL_LIB_INSTALL_PATH}
    COMPONENT ${PVASOL_RUNTIME_PACKAGE}
)
