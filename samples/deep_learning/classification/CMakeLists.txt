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
    message(WARNING "Python3 interpreter not found, skipping building classification sample")
    return()
endif()

project(classification)

# Define names and paths for the Swin-T model and related assets
set(ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../assets/onnx")

# Define models as a single structured list
set(MODEL_CONFIGS
    "swin_t_Opset18.onnx|swin_t|CHW|HWC"
    "resnet50_Opset18.onnx|resnet50|CHW|HWC"
    "convnext_tiny_Opset18_timm.onnx|convnext_tiny|CHW|HWC"
)

pva_compile_multiple_networks(COMPILED_MODELS "${MODEL_CONFIGS}" "${ASSETS_DIR}" "classification")

# =============================================================================
# Executable Creation
# =============================================================================
# Check if any models were successfully compiled
if(COMPILED_MODELS STREQUAL "")
    message(WARNING "No models were successfully compiled")
    return()
endif()

# Define classification executable
add_executable(classification
    classification.c
    ${COMPILED_MODELS}
)

# =============================================================================
# Linking and Installation
# =============================================================================
target_link_libraries(classification
    PRIVATE cupva_host
    PRIVATE nvcv_types
    PRIVATE pva_operator
    PRIVATE imageIO_utils
    PRIVATE CUDA::cudart_static
)

target_link_options(classification PRIVATE "-Wl,--export-dynamic")

install(TARGETS classification
        RUNTIME DESTINATION ${PVASOL_BIN_INSTALL_PATH}
        COMPONENT ${PVASOL_SAMPLES_PACKAGE})
