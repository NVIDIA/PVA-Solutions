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

# Create library for Spatial Processing fixed-point reference implementation
add_library(snapshot_extraction_ref STATIC
    snapshot_extraction_ref.cpp
)

target_include_directories(snapshot_extraction_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(snapshot_extraction_ref
    PRIVATE nvcv_types
    PRIVATE cupva_host
    PRIVATE pva_operator_includes
    PRIVATE pva_operator_priv_includes
    PRIVATE pva_operator_utils
)

add_library(snapshot_extraction_ref_includes INTERFACE)

target_include_directories(snapshot_extraction_ref_includes INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)