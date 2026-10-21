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

# Object library: linked into radar_pipeline_advanced (.so) and exposes helpers
# (file parsing, JSON/radar assets, initSystemConfig) without duplicate symbols.
add_library(radar_pipeline_utils OBJECT
    radar_file_parser.cpp
    radar_pipeline_utils.cpp
)

set_target_properties(radar_pipeline_utils PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(radar_pipeline_utils
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../lib>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../operators/include>
)

target_link_libraries(radar_pipeline_utils
    PRIVATE
        cupva_host
        nvcv_types
        pva_operator
        nlohmann_json
)
