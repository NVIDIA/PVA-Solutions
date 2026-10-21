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

#
# Function to setup CUDA Toolkit
# This function should be called after CUPVA_BUILD_MODE is set
#
function(setup_cuda_toolkit)
    if(CUPVA_BUILD_MODE STREQUAL QNX)
        if(NOT CUDA_TOOLKIT_DIR)
            message(FATAL_ERROR "CUDA_TOOLKIT_DIR is not set. Please use the -DCUDA_TOOLKIT_DIR=<path> option to specify the path to the CUDA Toolkit root directory.")
        endif()

        # For QNX, manually configure everything without calling find_package(CUDAToolkit)
        # because find_package auto-detects the wrong architecture (x86_64-linux instead of aarch64-qnx)

        # Set CUDA variables for QNX (PARENT_SCOPE for immediate parent, used in status messages)
        set(CUDAToolkit_FOUND TRUE PARENT_SCOPE)
        set(CUDAToolkit_ROOT "${CUDA_TOOLKIT_DIR}" PARENT_SCOPE)
        set(CUDAToolkit_BIN_DIR "${CUDA_TOOLKIT_DIR}/bin" PARENT_SCOPE)
        set(CUDAToolkit_INCLUDE_DIR "${CUDA_TOOLKIT_DIR}/targets/aarch64-qnx/include" PARENT_SCOPE)
        set(CUDAToolkit_TARGET_DIR "${CUDA_TOOLKIT_DIR}/targets/aarch64-qnx" PARENT_SCOPE)
        set(CUDA_cudart_LIBRARY "${CUDA_TOOLKIT_DIR}/targets/aarch64-qnx/lib/libcudart.so" PARENT_SCOPE)
        set(CUDA_cudart_static_LIBRARY "${CUDA_TOOLKIT_DIR}/targets/aarch64-qnx/lib/libcudart_static.a" PARENT_SCOPE)

        set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_DIR}/bin/nvcc" PARENT_SCOPE)
        set(CMAKE_CUDA_HOST_COMPILER "$ENV{QNX_HOST}/usr/bin/q++" PARENT_SCOPE)
        set(CMAKE_CUDA_COMPILER_FORCED TRUE PARENT_SCOPE)
        set(CMAKE_CUDA_FLAGS "--qpp-config 12.2.0,gcc_ntoaarch64le" PARENT_SCOPE)

        # C++ certified option is -xc++ instead of -lang-c++
        set(CMAKE_CXX_COMPILE_OBJECT
            "<CMAKE_CXX_COMPILER> -xc++ <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>" PARENT_SCOPE)

        # On QNX, explicitly link the C++ standard library (libc++) and math library for all executables
        # This is needed when C programs link with C++ object libraries or when using math functions
        link_libraries(c++ m)

        # Manually create GLOBAL imported targets for the correct QNX libraries
        # These override any find_package(CUDAToolkit) calls in subdirectories (e.g., CV-CUDA)
        if(NOT TARGET CUDA::cudart)
            add_library(CUDA::cudart SHARED IMPORTED GLOBAL)
            set_target_properties(CUDA::cudart PROPERTIES
                IMPORTED_LOCATION "${CUDA_TOOLKIT_DIR}/targets/aarch64-qnx/lib/libcudart.so"
                INTERFACE_INCLUDE_DIRECTORIES "${CUDA_TOOLKIT_DIR}/targets/aarch64-qnx/include"
            )
        endif()

        if(NOT TARGET CUDA::cudart_static)
            add_library(CUDA::cudart_static STATIC IMPORTED GLOBAL)
            set_target_properties(CUDA::cudart_static PROPERTIES
                IMPORTED_LOCATION "${CUDA_TOOLKIT_DIR}/targets/aarch64-qnx/lib/libcudart_static.a"
                INTERFACE_INCLUDE_DIRECTORIES "${CUDA_TOOLKIT_DIR}/targets/aarch64-qnx/include"
            )
        endif()
    else()
        # For non-QNX platforms, use standard find_package
        find_package(CUDAToolkit REQUIRED)

        # Propagate CUDAToolkit variables to parent scope
        set(CUDAToolkit_FOUND ${CUDAToolkit_FOUND} PARENT_SCOPE)
        set(CUDAToolkit_VERSION ${CUDAToolkit_VERSION} PARENT_SCOPE)
        set(CUDAToolkit_BIN_DIR ${CUDAToolkit_BIN_DIR} PARENT_SCOPE)
        set(CUDAToolkit_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS} PARENT_SCOPE)
        set(CUDAToolkit_TARGET_DIR ${CUDAToolkit_TARGET_DIR} PARENT_SCOPE)
        set(CUDA_cudart_LIBRARY ${CUDA_cudart_LIBRARY} PARENT_SCOPE)
        set(CUDA_cudart_static_LIBRARY ${CUDA_cudart_static_LIBRARY} PARENT_SCOPE)
    endif()
endfunction()

