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

add_library(pva_operator_priv_includes INTERFACE)
target_include_directories(pva_operator_priv_includes
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
    INTERFACE $<TARGET_PROPERTY:CUDA::cudart_static,INTERFACE_INCLUDE_DIRECTORIES>)

add_library(pva_operator_priv OBJECT
    PvaAllocator.cpp
    IOperator.cpp
    RadarUtils.cpp
)

target_link_libraries(pva_operator_priv
    PRIVATE cupva_host
    PRIVATE nvcv_types
    PRIVATE pva_operator_includes
    PUBLIC pva_operator_priv_includes
)

target_include_directories(pva_operator_priv
    PUBLIC .)

add_library(pva_operator_priv_submit_definitions STATIC
    PvaSubmitDefinitions.cpp
)

target_link_libraries(pva_operator_priv_submit_definitions
    PRIVATE cupva_host
    PRIVATE nvcv_types
    PRIVATE pva_operator_includes
)

target_include_directories(pva_operator_priv_submit_definitions
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if(PVA_BUILD_OP_FILTER)
    target_compile_definitions(pva_operator_priv_submit_definitions
        PRIVATE PVA_FILTER_OP=1)
    foreach(_op IN LISTS PVA_BUILD_OP_FILTER)
        string(TOUPPER "${_op}" _filter_upper)
        target_compile_definitions(pva_operator_priv_submit_definitions
            PRIVATE PVA_OP_FILTER_${_filter_upper}=1)
    endforeach()
endif()
