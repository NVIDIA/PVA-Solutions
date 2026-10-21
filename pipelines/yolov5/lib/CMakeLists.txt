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
# Pipeline structure shared by the YOLOv5 pipelines: stage map, tensor
# ownership, CUDA/PVA interop and the three-frame schedule.
#

cmake_minimum_required(VERSION 3.20)

project(yolov5_pipeline_lib LANGUAGES CXX)

find_package(nvcv_types REQUIRED)

add_library(yolov5_pipeline_lib STATIC
    yolov5_pipeline_tensors.cpp
    yolov5_pipeline_interop.cpp
    yolov5_pipeline_iface.cpp
    yolov5_pipeline_schedule.cpp
)

target_include_directories(yolov5_pipeline_lib
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
    # nvcvTensorWrapCudaBuf, used to alias TensorRT outputs into PVA tensors.
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../samples/utils
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(yolov5_pipeline_lib
    PUBLIC nvcv_types
    PUBLIC pva_operator
    PUBLIC cupva_host
    PUBLIC CUDA::cudart_static
    PUBLIC interop_utils
)

install(TARGETS yolov5_pipeline_lib
        ARCHIVE DESTINATION ${PVASOL_LIB_INSTALL_PATH}
        COMPONENT ${PVASOL_SAMPLES_PACKAGE})
install(FILES yolov5_pipeline_tensors.hpp yolov5_pipeline_interop.hpp yolov5_pipeline_schedule.hpp
              ../include/yolov5_pipeline_iface.hpp
        DESTINATION ${PVASOL_INCLUDE_INSTALL_PATH}/pipelines/yolov5/lib
        COMPONENT ${PVASOL_SAMPLES_PACKAGE})
