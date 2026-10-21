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
    DESTINATION share/pipeline_assets
    COMPONENT sample-assets)

# radar_advanced KPI tests expect radar_raw_8t8r_1.bin under pipeline_assets/radar_advanced/
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/assets/radar/8t8r/radar_raw_8t8r_1.bin
    DESTINATION share/pipeline_assets/radar_advanced
    COMPONENT sample-assets)

project(pva_solutions_pipelines)

add_subdirectory(radar)

option(PVA_SOLUTIONS_REQUIRE_TENSORRT
       "Fail L4T configuration when the TensorRT 10 development SDK is unavailable"
       OFF)

# The TensorRT pipelines build for Linux targets only, because they require the
# AArch64 TensorRT SDK. CUPVA_BUILD_MODE=L4T covers DRIVE OS Linux and Jetson Linux.
if(CUPVA_BUILD_MODE STREQUAL "L4T")
    if(PVA_SOLUTIONS_REQUIRE_TENSORRT)
        find_package(TensorRT 10 EXACT REQUIRED)
    endif()
    add_subdirectory(yolov5/utils)
    add_subdirectory(yolov5/lib)
    add_subdirectory(yolov5/gpu)
    add_subdirectory(yolov5/pva_gpu)
endif()
