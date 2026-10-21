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

add_library(peak_detection_ref OBJECT
    peak_detection_ref.cpp
)

target_link_libraries(peak_detection_ref
    PRIVATE nvcv_types
    PRIVATE pva_operator
    PRIVATE cupva_host
    PRIVATE cupva_host_utils
    PRIVATE imageIO_utils
    PRIVATE pva_operator_includes
)

target_include_directories(peak_detection_ref
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(peak_detection_ref
    PUBLIC pva_peak_detection_op_includes
)
