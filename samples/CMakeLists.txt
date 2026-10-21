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

if(CUPVA_BUILD_MODE STREQUAL "L4T" OR CUPVA_BUILD_MODE STREQUAL "QNX")
    find_package(nvsci QUIET)
    if(nvsci_FOUND)
        set(NVSCI_SUPPORTED ON CACHE BOOL "NVIDIA NvSci IPC library support")
        message(STATUS "nvsci found, NVSCI_SUPPORTED is ON")
    else()
        set(NVSCI_SUPPORTED OFF CACHE BOOL "NVIDIA NvSci IPC library support")
        message(WARNING "nvsci not found, NVSCI_SUPPORTED is OFF")
    endif()
else()
    set(NVSCI_SUPPORTED OFF CACHE BOOL "NVIDIA NvSci IPC library support")
    message(STATUS "CUPVA_BUILD_MODE is not L4T or QNX, NVSCI_SUPPORTED is OFF")
endif()

# Install assets
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/assets/
    DESTINATION share/sample_assets
    COMPONENT sample-assets)

project(pva_solutions_samples)

add_subdirectory(utils)
add_subdirectory(image_processing)
add_subdirectory(deep_learning)
add_subdirectory(lidar_processing)
add_subdirectory(linear_algebra)
add_subdirectory(radar_processing)
add_subdirectory(optimization)
