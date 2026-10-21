#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#
# YOLOv5 pipeline utilities shared by the GPU-only and PVA+GPU pipelines:
# argument parsing, image I/O, COCO output, timing, detection checking,
# TensorRT helpers and INT8 calibration. Preprocessing is pipeline-specific and
# lives with each pipeline under gpu/ and pva_gpu/.
#

cmake_minimum_required(VERSION 3.20)

project(yolov5_pipeline_utils LANGUAGES CXX)

find_package(nvcv_types REQUIRED)
find_package(TensorRT 10 EXACT QUIET)

set(YOLOV5_UTILS_SRCS
    yolov5_common.cpp
    pipeline_args.cpp
    int8_preprocess.cpp
    image_io.cpp
    detection_check.cpp
)

if(TensorRT_FOUND)
    list(APPEND YOLOV5_UTILS_SRCS int8_calibrator.cpp trt_utils.cpp)
endif()

add_library(yolov5_pipeline_utils STATIC ${YOLOV5_UTILS_SRCS})

target_include_directories(yolov5_pipeline_utils
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(yolov5_pipeline_utils
    # Link cudart statically so the binaries do not depend on the target's CUDA runtime version.
    PUBLIC CUDA::cudart_static
    PRIVATE nvcv_types
    PRIVATE lodepng
)

if(TensorRT_FOUND)
    target_link_libraries(yolov5_pipeline_utils PRIVATE TensorRT::nvinfer)
endif()

install(TARGETS yolov5_pipeline_utils
        ARCHIVE DESTINATION ${PVASOL_LIB_INSTALL_PATH}
        COMPONENT ${PVASOL_SAMPLES_PACKAGE})
install(FILES yolov5_common.h pipeline_args.h detection_check.h int8_preprocess.h int8_calibrator.h
              image_io.h pipeline_error.h trt_utils.h
        DESTINATION ${PVASOL_INCLUDE_INSTALL_PATH}/pipelines/yolov5/utils
        COMPONENT ${PVASOL_SAMPLES_PACKAGE})
