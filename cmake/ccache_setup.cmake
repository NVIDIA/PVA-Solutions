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

# Auto-detect ccache and register it as the C/C++ compiler launcher.
# Must be included before project() so the launcher is active during compiler detection.
# Works transparently for both native and cross-compilation (L4T, QNX) builds.
# Override by passing -DCMAKE_CXX_COMPILER_LAUNCHER=<path> to disable or use a different tool.

# Include guard: this file may be included from both the repo-root CMakeLists.txt
# and public/CMakeLists.txt depending on the build entry point.
if(_PVA_CCACHE_SETUP_INCLUDED)
    return()
endif()
set(_PVA_CCACHE_SETUP_INCLUDED TRUE)

find_program(CCACHE_CMD ccache)
if(CCACHE_CMD)
    if(NOT CMAKE_CXX_COMPILER_LAUNCHER)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_CMD}" CACHE STRING "C++ compiler launcher")
    endif()
    if(NOT CMAKE_C_COMPILER_LAUNCHER)
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_CMD}" CACHE STRING "C compiler launcher")
    endif()
    if(NOT CMAKE_CUDA_COMPILER_LAUNCHER)
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_CMD}" CACHE STRING "CUDA compiler launcher")
    endif()
endif()
