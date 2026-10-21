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
#
# 3-stage pipelined PVA + TensorRT YOLOv5 trimmed pipeline.
# Overlaps PVA pre-processing, GPU inference, and PVA post-processing
# across 3 frames for higher throughput.
#

cmake_minimum_required(VERSION 3.20)

project(yolov5_pva_gpu LANGUAGES CXX CUDA)

find_package(nvcv_types REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(TensorRT 10 EXACT)

if(NOT TensorRT_FOUND)
    message(WARNING "TensorRT 10 not found, skipping yolov5_pva_gpu pipeline")
    return()
endif()

message(STATUS "yolov5_pva_gpu: TensorRT ${TensorRT_VERSION} found")

add_executable(yolov5_pva_gpu
    pva_gpu_pipeline.cpp
)

set_target_properties(yolov5_pva_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(yolov5_pva_gpu PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
endif()

target_include_directories(yolov5_pva_gpu
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../samples/deep_learning/detection
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../samples/utils  # CudaInteropUtils.h for CUDA-PVA interop
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(yolov5_pva_gpu
    # PVA libraries
    PRIVATE cupva_host
    PRIVATE pva_operator          # OpImageResize, OpBatchedNMS, PvaAllocator
    PRIVATE pva_yolov5_decode_op  # OpYoloV5Decode (fused CHW transpose + decode)
    # CUDA-PVA interop utilities (nvcvImageWrapCudaBuf, nvcvTensorWrapCudaBuf)
    PRIVATE interop_utils
    PRIVATE pva_nhwc_to_nchw_letterbox_normalize_op  # OpNhwcToNchwLetterboxNormalize (fused letterbox+normalize+transpose)
    # NVCV types (for tensor/image handles)
    PRIVATE nvcv_types
    # TensorRT libraries
    PRIVATE TensorRT::nvinfer
    PRIVATE TensorRT::nvinfer_plugin
    PRIVATE TensorRT::nvonnxparser
    # CUDA runtime
    PRIVATE CUDA::cudart_static
    PRIVATE yolov5_pipeline_utils
    PRIVATE yolov5_pipeline_lib
)

# No RPATH, applied to the build tree too, so build-tree binaries run from any directory.
set_target_properties(yolov5_pva_gpu PROPERTIES
    INSTALL_RPATH ""
    BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS yolov5_pva_gpu
        RUNTIME DESTINATION ${PVASOL_BIN_INSTALL_PATH}
        COMPONENT ${PVASOL_SAMPLES_PACKAGE})
