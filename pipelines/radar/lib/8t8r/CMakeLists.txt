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
project(radar_pipeline VERSION 1.0.1)

# ============================================================================
# Radar Pipeline Advanced Shared Library (.so) for e2e-solution
# ============================================================================
add_library(radar_pipeline_8t8r SHARED
    radar_pipeline_context.cpp
    radar_pipeline_tensors.cpp
    radar_pipeline_pva.cpp
    radar_pipeline_reference.cpp
    radar_pipeline_factory.cpp
)

# Hide all symbols by default; only radar_pipeline_8t8r() is exported.
# This prevents symbol conflicts when both 8t8r and 4t4r are linked to the same binary.
set_target_properties(radar_pipeline_8t8r PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Link radar pipeline advanced shared library
target_link_libraries(radar_pipeline_8t8r
    PRIVATE cupva_host
    PRIVATE CUDA::cudart_static
    PRIVATE nvcv_types
    PRIVATE pva_operator
    PRIVATE pva_operator_utils
    PRIVATE imageIO_utils
    PRIVATE pva_snapshot_extraction_op
    PRIVATE pva_doa_angle_fft_op
    PRIVATE pva_bartlett_op
    PRIVATE radar_ref_utils
    PRIVATE target_processing_ref
    PRIVATE doa_angle_fft_ref
    PRIVATE doa_bartlett_beamforming_ref
    PRIVATE snapshot_extraction_ref
    PRIVATE nci_ref
    PRIVATE radar_cfar_ref
    PRIVATE doppler_fft_ref
    PRIVATE range_fft_ref
    # Capon: the steering-LUT builder (production), and the operator references validate() compares with.
    PRIVATE pva_capon_roi_beamspace
    PRIVATE capon_roi_preprocess_ref
)
if(CUPVA_LLVM_SUPPORTED) # doa_capon_roi_ref, like the Capon DoA operator, needs the LLVM front end
    target_link_libraries(radar_pipeline_8t8r PRIVATE doa_capon_roi_ref)
endif()

# Include directories for radar pipeline advanced
target_include_directories(radar_pipeline_8t8r
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../operators/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../>
)

# Install shared library
install(TARGETS radar_pipeline_8t8r
    LIBRARY DESTINATION ${PVASOL_LIB_INSTALL_PATH}
    COMPONENT ${PVASOL_RUNTIME_PACKAGE}
)
