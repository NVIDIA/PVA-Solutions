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

cmake_minimum_required(VERSION 3.20)

project(yolov5_gpu LANGUAGES CXX CUDA)

# Find TensorRT and NVCV tensor types.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(TensorRT 10 EXACT)
find_package(nvcv_types REQUIRED)

if(NOT TensorRT_FOUND)
    message(WARNING "TensorRT 10 not found, skipping yolov5_gpu")
    return()
endif()

message(STATUS "yolov5_gpu: TensorRT ${TensorRT_VERSION} found")

# ============================================================================
# Pipeline Executable — TensorRT + CUDA
# ============================================================================

add_executable(yolov5_gpu
    gpu_pipeline.cpp
    preprocess_cuda.cu
    # CUDA letterbox: stages 1-2 of this backend. The PVA backend replaces
    # these with PVA ImageResize + NhwcToNchwLetterboxNormalize.
    letterbox_cuda.cu
    letterbox_gpu.cpp
)

set_target_properties(yolov5_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(yolov5_gpu PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
endif()

target_include_directories(yolov5_gpu
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(yolov5_gpu
    PRIVATE TensorRT::nvinfer
    PRIVATE TensorRT::nvinfer_plugin
    PRIVATE TensorRT::nvonnxparser
    PRIVATE CUDA::cudart_static
    PRIVATE yolov5_pipeline_utils
    PRIVATE yolov5_pipeline_lib
    PRIVATE nvcv_types
    PRIVATE lodepng
)

# No RPATH, applied to the build tree too, so build-tree binaries run from any directory.
set_target_properties(yolov5_gpu PROPERTIES
    INSTALL_RPATH ""
    BUILD_WITH_INSTALL_RPATH TRUE
)

# ============================================================================
# Installation
# ============================================================================

install(TARGETS yolov5_gpu
        RUNTIME DESTINATION ${PVASOL_BIN_INSTALL_PATH}
        COMPONENT ${PVASOL_SAMPLES_PACKAGE})

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../../samples/assets/onnx/prepare_yolov5_nms_onnx.py
        DESTINATION ${PVASOL_BIN_INSTALL_PATH}
        COMPONENT ${PVASOL_SAMPLES_PACKAGE})

# ============================================================================
# GPU-only model
# ============================================================================

# Generate yolov5s_672_nms.onnx (YOLOv5 decode + EfficientNMS) from the PVA+GPU
# model listed in networks.txt, and install it with the sample assets.
set(YOLOV5_ONNX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../samples/assets/onnx")
set(YOLOV5_NMS_PREPARE_SCRIPT "${YOLOV5_ONNX_DIR}/prepare_yolov5_nms_onnx.py")
set(YOLOV5_PVA_GPU_MODEL "${YOLOV5_ONNX_DIR}/yolov5s_trimmed_reshape_tranpose.onnx")
if(NOT EXISTS "${YOLOV5_PVA_GPU_MODEL}" AND EXISTS "${DEFAULT_ONNX_MODEL_DIR}/yolov5s_trimmed_reshape_tranpose.onnx")
    set(YOLOV5_PVA_GPU_MODEL "${DEFAULT_ONNX_MODEL_DIR}/yolov5s_trimmed_reshape_tranpose.onnx")
endif()

find_package(Python3 COMPONENTS Interpreter)
if(NOT EXISTS "${YOLOV5_PVA_GPU_MODEL}" OR NOT Python3_Interpreter_FOUND)
    message(WARNING "yolov5s_trimmed_reshape_tranpose.onnx or Python 3 not found, "
                    "skipping yolov5s_672_nms.onnx (see samples/assets/onnx/README.md)")
else()
    set(YOLOV5_GPU_ONLY_MODEL "${CMAKE_CURRENT_BINARY_DIR}/yolov5s_672_nms.onnx")
    add_custom_command(
        OUTPUT "${YOLOV5_GPU_ONLY_MODEL}"
        COMMAND "${Python3_EXECUTABLE}" "${YOLOV5_NMS_PREPARE_SCRIPT}"
                "${YOLOV5_PVA_GPU_MODEL}" "${YOLOV5_GPU_ONLY_MODEL}"
                --score-threshold 0.25 --iou-threshold 0.45
        DEPENDS "${YOLOV5_PVA_GPU_MODEL}" "${YOLOV5_NMS_PREPARE_SCRIPT}"
        COMMENT "Generating the GPU only YOLOv5 decode + EfficientNMS ONNX model"
        VERBATIM)
    add_custom_target(prepare_yolov5_nms_model ALL DEPENDS "${YOLOV5_GPU_ONLY_MODEL}")
    install(FILES "${YOLOV5_GPU_ONLY_MODEL}"
            DESTINATION share/sample_assets/onnx
            COMPONENT sample-assets)
endif()
