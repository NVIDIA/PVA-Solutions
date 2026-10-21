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

# Create library for DOA Angle FFT 8x8 reference implementation
add_library(doa_angle_fft_ref STATIC
    doa_angle_fft_ref.cpp
)

target_include_directories(doa_angle_fft_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(doa_angle_fft_ref
    PRIVATE nvcv_types
    PRIVATE cupva_host
    PRIVATE pva_operator_includes
    PRIVATE pva_operator_priv_includes
    PRIVATE pva_operator_utils
    PRIVATE snapshot_extraction_ref_includes
)

add_library(doa_angle_fft_ref_includes INTERFACE)

target_include_directories(doa_angle_fft_ref_includes INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# On release builds, avoid fused multiply-add and other optimizations that diverge the reference from the PVA impl.
if (NOT CUPVA_BUILD_MODE STREQUAL "NATIVE")
    target_compile_options(doa_angle_fft_ref PRIVATE
        "$<$<CONFIG:Release,RelWithDebInfo>:-fno-fast-math>"
        "$<$<CONFIG:Release,RelWithDebInfo>:-ffp-contract=off>"
    )
endif()
