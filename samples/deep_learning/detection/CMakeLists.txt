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

# Python3 is required for processing the ONNX model and building the sample.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
if(NOT Python3_Interpreter_FOUND)
    message(WARNING "Python3 interpreter not found, skipping building detection sample")
    return()
endif()

project(detection)

set(ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../assets/onnx")

set(MODEL_CONFIGS
    "yolov5s.optimized.onnx|yolov5s|CHW|HWC"
    "yolo26s.onnx|yolov26s|CHW|HWC|/model.23/TopK_1_output_0,/model.23/Mod_output_0,/model.23/GatherElements_1_output_0"
)

pva_compile_multiple_networks(COMPILED_MODELS "${MODEL_CONFIGS}" "${ASSETS_DIR}" "detection")

if(COMPILED_MODELS STREQUAL "")
    message(WARNING "No models were successfully compiled")
    return()
endif()

add_executable(detection
    detection.cpp
    coco_evaluation.cpp
    yolov5_postprocess.cpp
    ${COMPILED_MODELS}
)

target_link_libraries(detection
    PRIVATE cupva_host
    PRIVATE nvcv_types
    PRIVATE pva_operator
    PRIVATE pva_dl_inference_op
    PRIVATE imageIO_utils
    PRIVATE CUDA::cudart_static
)

target_link_options(detection PRIVATE "-Wl,--export-dynamic")

install(TARGETS detection
        RUNTIME DESTINATION ${PVASOL_BIN_INSTALL_PATH}
        COMPONENT ${PVASOL_SAMPLES_PACKAGE})
