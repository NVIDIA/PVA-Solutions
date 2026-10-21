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

add_library(doppler_fft_ref OBJECT
    doppler_fft_ref.cpp
)

target_link_libraries(doppler_fft_ref
    PRIVATE nvcv_types
    PRIVATE pva_operator
    PRIVATE pva_operator_utils
    PRIVATE pva_operator_priv
    PRIVATE cupva_host
    PRIVATE radar_ref_utils
    PUBLIC imageIO_utils
    PRIVATE CUDA::cudart_static
)

target_include_directories(doppler_fft_ref
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src/operator/radar_processing/doppler_fft/device
)
